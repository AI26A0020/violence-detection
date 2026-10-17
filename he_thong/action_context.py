"""Isolated causal action experiment. Importing this module changes no app state."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import math
import time

import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


GROUP_NAMES = {
    'person_contact': ('punching person (boxing)', 'slapping', 'wrestling'),
    'kicking': ('drop kicking', 'high kick', 'side kick'),
    'stage_combat': ('sword fighting', 'fencing'),
    'hugging': ('hugging',),
    'basketball': ('dribbling basketball', 'dunking basketball', 'playing basketball', 'shooting basketball'),
    'ball_sports': ('kicking soccer ball', 'kicking field goal', 'playing kickball', 'hitting baseball', 'playing tennis'),
    'exercise': ('punching bag', 'arm wrestling', 'push up', 'pull ups', 'situp'),
    'care_context': ('massaging person\'s head', 'massaging back', 'massaging legs', 'massaging feet'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ActionState:
    """Each video/camera owns its observations and timing."""
    frames: deque = field(default_factory=lambda: deque(maxlen=16))
    last_observed: float | None = None
    last_scored: float | None = None
    segment: int = 0

    def reset(self):
        self.frames.clear()
        self.last_observed = self.last_scored = None
        self.segment += 1


class CausalActionContext:
    """Frozen Kinetics features from real prior observations, never padded."""
    def __init__(self, weights=None, threads=2, min_frames=8, min_span=.875,
                 score_interval=.25, max_gap=.45, sample_fps=8):
        if (not 8 <= min_frames <= 16 or min_span < .875 or score_interval < .25
                or sample_fps not in (8, 16)
                or (sample_fps == 16 and (min_frames != 16 or min_span < .9375))):
            raise ValueError('Unsupported causal action sampling parameters')
        path = Path(weights) if weights else Path.home() / '.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth'
        if not path.is_file():
            raise FileNotFoundError('Local R3D weights required; this experiment never downloads weights')
        torch.set_num_threads(int(threads))
        self.weights_path, self.weights_sha256 = str(path.resolve()), sha256(path)
        self.model = r3d_18(weights=None).eval()
        self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.preprocess = R3D_18_Weights.KINETICS400_V1.transforms()
        self.categories = list(R3D_18_Weights.KINETICS400_V1.meta['categories'])
        self.groups = {key: [self.categories.index(name) for name in names if name in self.categories]
                       for key, names in GROUP_NAMES.items()}
        self.min_frames, self.min_span = min_frames, min_span
        self.score_interval, self.max_gap = score_interval, max_gap
        # observe accepts externally sampled real frames; the runtime uses this
        # declaration to select the identical capture rate and warmup profile.
        self.sample_fps = int(sample_fps)

    def prepare_frame(self, rgb):
        """Official per-frame preprocessing; safe to call outside capture callbacks."""
        image = np.ascontiguousarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected an actual RGB uint8 video frame')
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return self.preprocess(tensor)[:, 0]

    def infer_samples(self, samples, segment=0):
        """Score an immutable snapshot of prepared real past frames, without state mutation."""
        if not self.min_frames <= len(samples) <= 16:
            raise ValueError('Insufficient or excessive real action observations')
        stamps = [row[0] for row in samples]
        ids = [row[1] for row in samples]
        if (not all(math.isfinite(t) and t >= 0 for t in stamps)
                or any(b <= a or b-a > self.max_gap for a, b in zip(stamps, stamps[1:]))
                or any(b <= a for a, b in zip(ids, ids[1:]))
                or stamps[-1] - stamps[0] + 1e-8 < self.min_span):
            raise ValueError('Action snapshot must contain unique contiguous past observations')
        clip = torch.stack([row[2] for row in samples], dim=1).unsqueeze(0)
        began = time.perf_counter()
        with torch.inference_mode():
            features = self.model.stem(clip)
            for layer in (self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4):
                features = layer(features)
            embedding = self.model.avgpool(features).flatten(1)
            probabilities = self.model.fc(embedding).softmax(-1)[0].cpu().numpy()
        return {'second': stamps[-1], 'start_second': stamps[0],
                'frame_ids': ids, 'frame_seconds': stamps, 'real_frames': len(samples),
                'span': stamps[-1] - stamps[0], 'segment': segment,
                'sample_fps': getattr(self, 'sample_fps', 8),
                'min_real_frames': self.min_frames, 'min_span': self.min_span,
                'forward_seconds': time.perf_counter() - began,
                'probabilities': probabilities, 'embedding': embedding[0].cpu().numpy(),
                'scores': {key: float(probabilities[indices].sum()) for key, indices in self.groups.items()}}

    def observe(self, rgb, stamp, frame_id, state, scene_cut=False):
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError('A finite source timestamp is required')
        if not isinstance(frame_id, (int, np.integer)) or isinstance(frame_id, bool) or frame_id < 0:
            raise ValueError('A nonnegative integer source frame ID is required')
        if state.last_observed is not None:
            if stamp <= state.last_observed or frame_id <= state.frames[-1][1]:
                raise ValueError('Repeated, out-of-order or future-padded observations are forbidden')
            if scene_cut or stamp - state.last_observed > self.max_gap:
                state.reset()
        elif scene_cut:
            state.reset()
        processed = self.prepare_frame(rgb)
        state.frames.append((float(stamp), int(frame_id), processed))
        state.last_observed = float(stamp)
        if len(state.frames) < self.min_frames or stamp - state.frames[0][0] + 1e-8 < self.min_span:
            return None
        if state.last_scored is not None and stamp - state.last_scored + 1e-8 < self.score_interval:
            return None
        samples = list(state.frames)
        result = self.infer_samples(samples, state.segment)
        state.last_scored = float(stamp)
        return result

