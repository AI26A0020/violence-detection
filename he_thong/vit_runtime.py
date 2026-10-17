"""Isolated research CLS adapter. Importing never loads Torch or a backbone.

Only infer_samples encodes frames; the shared worker supplies an opaque unique
stream_key for each monitor. This module does not enable/deploy any candidate.
"""
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np

from vit_causal_context import PROFILE


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _hash(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected an exact SHA256 backbone identity')
    return value


class _Profile:
    def as_dict(self):
        return dict(PROFILE)


def _profile(source):
    if hasattr(source, 'sampling_profile'):
        source = source.sampling_profile.as_dict()
    elif not isinstance(source, Mapping):
        source = getattr(source, 'profile', None)
    if isinstance(source, Mapping) and 'profile' in source:
        source = source['profile']
    if not isinstance(source, Mapping) or dict(source) != PROFILE:
        raise ValueError('Exact causal CLS preprocessing, clock and feature profile required')
    return dict(PROFILE)


class CLSContext:
    sample_fps, min_frames, min_span, window = 4, 4, .75, 4
    feature_dim, max_gap = 2304, .60
    max_prepared_frame_bytes = 3*224*224*4
    maxpreparedbytes = max_prepared_frame_bytes
    scene_cadence = 'native'
    source_clock = PROFILE['clock']

    def __init__(self, backbone_path, encoder=None, max_streams=4):
        if type(max_streams) is not int or not 4 <= max_streams <= 32:
            raise ValueError('CLS stream cache limit must be an integer from 4 to 32')
        path = Path(backbone_path).resolve()
        if not path.is_file():
            raise FileNotFoundError('Local CLS backbone weights required')
        self.weights_path, self.weights_sha256 = str(path), _sha(path)
        self.profile, self.sampling_profile = dict(PROFILE), _Profile()
        self.max_streams, self.cache_frames = max_streams, 8
        self._streams, self.closed = OrderedDict(), False
        if encoder is None:
            # The existing strict all-key mapping loads this cached path only.
            # A supplied copy is safe only when its bytes match that exact file.
            default = Path(__file__).resolve().parent/'models/vit-base-violence-detection/pytorch_model.bin'
            if not default.is_file() or (default.resolve() != path and _sha(default) != self.weights_sha256):
                raise ValueError('Supplied CLS weights differ from the explicitly mapped local backbone')
            from vit_causal_context import FrozenViTCLS
            encoder = FrozenViTCLS()
        declared = getattr(encoder, 'weights_sha256', self.weights_sha256)
        if declared != self.weights_sha256:
            raise ValueError('Injected/loaded encoder backbone identity mismatch')
        self.encoder = encoder
        if not callable(encoder) and not callable(getattr(encoder, 'encode_frame', None)):
            raise TypeError('Encoder must supply encode_frame(prepared)')

    def prepare_frame(self, rgb):
        """Exact existing PIL RGB bilinear224, mean/std .5; no CLS encoding."""
        from PIL import Image
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1):
            raise ValueError('Expected an actual RGB uint8 frame')
        resized = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR)).copy()
        return (resized.transpose(2, 0, 1).astype(np.float32)/255.-.5)/.5

    def infer_samples(self, samples, segment=0, *, stream_key):
        """Encode four real prior frames on the worker, reusing only exact CLS IDs."""
        if self.closed:
            raise RuntimeError('CLS context is closed')
        if type(segment) is not int or segment < 0 or len(samples) != 4:
            raise ValueError('CLS requires exactly four real observations and a valid segment')
        try:
            hash(stream_key)
        except TypeError as exc:
            raise ValueError('Expected an opaque hashable stream key') from exc
        if stream_key is None:
            raise ValueError('A unique per-monitor stream key is required')
        stamps, ids, prepared = [], [], []
        for stamp, identity, image in samples:
            if (isinstance(stamp, (bool, np.bool_)) or not isinstance(stamp, Real)
                    or not math.isfinite(stamp) or stamp < 0
                    or isinstance(identity, (bool, np.bool_)) or not isinstance(identity, (int, np.integer)) or identity < 0):
                raise ValueError('Invalid native observation identity')
            image = np.asarray(image)
            if image.dtype != np.float32 or image.shape != (3, 224, 224) or not np.isfinite(image).all():
                raise ValueError('Expected finite prepared float32 RGB224')
            stamps.append(float(stamp)); ids.append(int(identity)); prepared.append(image)
        if (any(b <= a for a, b in zip(ids, ids[1:]))
                or any(b <= a or b-a > self.max_gap+1e-6 for a, b in zip(stamps, stamps[1:]))
                or stamps[-1]-stamps[0]+1e-6 < self.min_span):
            raise ValueError('Repeated, gapped, unordered or short CLS window')
        state = self._streams.get(stream_key)
        if state is None:
            if len(self._streams) >= self.max_streams:
                self._streams.popitem(last=False)
            state = {'segment': segment, 'frames': OrderedDict(), 'last_id': -1, 'last_stamp': -math.inf}
            self._streams[stream_key] = state
        elif segment < state['segment']:
            raise ValueError('Stale CLS scene generation')
        elif segment != state['segment']:
            state = {'segment': segment, 'frames': OrderedDict(), 'last_id': -1, 'last_stamp': -math.inf}
            self._streams[stream_key] = state
        self._streams.move_to_end(stream_key)
        sequence = []
        encode = getattr(self.encoder, 'encode_frame', self.encoder)
        for stamp, identity, image in zip(stamps, ids, prepared):
            key = (identity, stamp)
            if key in state['frames']:
                feature = state['frames'][key]
            else:
                if identity <= state['last_id'] or stamp <= state['last_stamp']:
                    raise ValueError('Source identity reused with changed timestamp or evicted stale frame')
                feature = np.asarray(encode(image), dtype=np.float32)
                if feature.shape != (768,) or not np.isfinite(feature).all():
                    raise ValueError('Expected a finite encoded CLS768 vector')
                feature = feature.copy()
                state['frames'][key] = feature
                state['last_id'], state['last_stamp'] = identity, stamp
                while len(state['frames']) > self.cache_frames:
                    state['frames'].popitem(last=False)
            sequence.append(feature)
        sequence = np.stack(sequence)
        embedding = np.concatenate((sequence[-1], sequence.mean(0), np.abs(np.diff(sequence, axis=0)).mean(0)))
        if embedding.shape != (2304,) or not np.isfinite(embedding).all():
            raise ValueError('Invalid causal CLS context embedding')
        return {'embedding': embedding, 'second': stamps[-1], 'start_second': stamps[0],
                'frame_ids': ids, 'frame_seconds': stamps, 'real_frames': 4,
                'span': stamps[-1]-stamps[0], 'segment': segment}

    def clear_cache(self):
        """Call after joining the shared worker; retains the loaded encoder."""
        self._streams.clear()

    def close(self):
        self.clear_cache()
        self.closed = True


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value):
    value = _array(value)
    if value.shape != ():
        raise ValueError('Expected scalar checkpoint metadata')
    return value.item()


def _vector(value, shape, dtype=np.float64):
    value = _array(value)
    if (value.shape != shape or not (np.issubdtype(value.dtype, np.floating)
            or np.issubdtype(value.dtype, np.integer)) or not np.isfinite(value).all()):
        raise ValueError('Malformed/nonfinite CLS head parameter')
    with np.errstate(over='ignore', invalid='ignore'):
        converted = value.astype(dtype, copy=True)
    if not np.isfinite(converted).all():
        raise ValueError('CLS head parameter exceeds its inference dtype range')
    return converted


class FrozenCLSHead:
    """Validated experimental MLP or portable residual; no class-order guessing.

    Research artifacts fail closed unless allow_research=True is explicit.
    NumPy inference permits weightless tests and portable residual use without
    loading Torch. Torch is imported only to deserialize an actual .pt file.
    """
    sample_fps, min_frames, min_span, window, feature_dim = 4, 4, .75, 4, 2304

    def __init__(self, checkpoint_path, backbone_sha256, *, allow_research=False):
        path = Path(checkpoint_path)
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as source:
                checkpoint = {key: source[key].copy() for key in source.files}
        else:
            import torch
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self._initialize(checkpoint, backbone_sha256, allow_research)

    @classmethod
    def from_checkpoint(cls, checkpoint, backbone_sha256, *, allow_research=False):
        """Validate an already-loaded mapping; supports tiny NumPy-only tests."""
        result = cls.__new__(cls)
        result._initialize(checkpoint, backbone_sha256, allow_research)
        return result

    def _initialize(self, checkpoint, backbone_sha256, allow_research):
        if type(allow_research) is not bool or not isinstance(checkpoint, Mapping):
            raise ValueError('Expected explicit research permission and a checkpoint mapping')
        flag = _scalar(checkpoint.get('runtime_compatible', np.asarray(False)))
        if type(flag) is not bool or not flag and not allow_research:
            raise ValueError('Research CLS checkpoint requires explicit allow_research=True')
        self.runtime_compatible, self.research_only = flag, not flag
        self.backbone_sha256 = _hash(backbone_sha256)
        if _scalar(checkpoint['backbone_sha256']) != self.backbone_sha256:
            raise ValueError('CLS head/backbone SHA256 mismatch')
        architecture = _scalar(checkpoint['architecture'])
        if architecture not in ('vit_cls_context_mil_v1', 'vit_cls_residual_prior_mil_v1'):
            raise ValueError('Unsupported CLS head architecture')
        if _scalar(checkpoint['input_dim']) != 2304:
            raise ValueError('CLS head input dimension differs')
        profile = checkpoint.get('profile')
        if profile is None:
            profile = json.loads(_scalar(checkpoint['profile_json']))
        _profile(profile)
        self.profile, self.sampling_profile = dict(PROFILE), _Profile()
        self.architecture = architecture
        dtype = np.float32 if architecture == 'vit_cls_context_mil_v1' else np.float64
        self.mean = _vector(checkpoint['mean'], (2304,), dtype)
        self.scale = _vector(checkpoint['scale'], (2304,), dtype)
        self.offset = _scalar(checkpoint['offset'])
        if (isinstance(self.offset, bool) or not isinstance(self.offset, (int, float))
                or not math.isfinite(self.offset) or np.any(self.scale <= 0)):
            raise ValueError('Invalid CLS calibration/normalization')
        if dtype == np.float32 and abs(self.offset) > np.finfo(np.float32).max:
            raise ValueError('CLS offset exceeds its inference dtype range')
        if architecture == 'vit_cls_residual_prior_mil_v1':
            self.weight = _vector(checkpoint['weight'], (2304,))
            self.bias = _scalar(checkpoint['bias'])
            if not isinstance(self.bias, (int, float)) or isinstance(self.bias, bool) or not math.isfinite(self.bias):
                raise ValueError('Invalid residual CLS bias')
            for key, shape in (('prior_weight', (2304,)), ('prior_bias', ())):
                if key in checkpoint:
                    _vector(checkpoint[key], shape)
        else:
            hidden = checkpoint['hidden']
            if type(hidden) is not int or hidden not in (0, 64):
                raise ValueError('Unsupported CLS MLP width')
            state = checkpoint['state_dict']
            if not isinstance(state, Mapping):
                raise ValueError('Invalid CLS MLP state')
            shapes = ({'weight': (1, 2304), 'bias': (1,)} if hidden == 0 else
                      {'0.weight': (64, 2304), '0.bias': (64,), '3.weight': (1, 64), '3.bias': (1,)})
            if set(state) != set(shapes):
                raise ValueError('CLS MLP state keys differ')
            self.hidden = hidden
            self.state = {key: _vector(state[key], shape, np.float32) for key, shape in shapes.items()}

    def assert_sampling(self, context_or_provenance):
        _profile(context_or_provenance)
        return self.sampling_profile

    def score(self, embeddings):
        dtype = np.float32 if self.architecture == 'vit_cls_context_mil_v1' else np.float64
        x = np.asarray(embeddings, dtype=dtype)
        if x.ndim != 2 or x.shape[1] != 2304 or not np.isfinite(x).all():
            raise ValueError('Expected finite CLS context features [N,2304]')
        x = (x-self.mean)/self.scale
        if self.architecture == 'vit_cls_residual_prior_mil_v1':
            margins = x@self.weight+self.bias-self.offset
        elif self.hidden == 0:
            margins = (x@self.state['weight'].T+self.state['bias']).reshape(-1)-np.float32(self.offset)
        else:
            x = np.maximum(x@self.state['0.weight'].T+self.state['0.bias'], 0)
            margins = (x@self.state['3.weight'].T+self.state['3.bias']).reshape(-1)-np.float32(self.offset)
        if not np.isfinite(margins).all():
            raise ValueError('Nonfinite CLS head output')
        probabilities = np.empty_like(margins)
        positive = margins >= 0
        probabilities[positive] = 1/(1+np.exp(-margins[positive]))
        exp = np.exp(margins[~positive])
        probabilities[~positive] = exp/(1+exp)
        if self.architecture == 'vit_cls_residual_prior_mil_v1':
            # Portable training defines alert as margin >= 0. Preserve that
            # boundary even if sigmoid rounds a tiny negative margin to .5.
            rounded = (~positive) & (probabilities >= .5)
            probabilities[rounded] = np.nextafter(np.float64(.5), np.float64(0.))
        return probabilities
