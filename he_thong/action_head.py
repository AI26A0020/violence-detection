"""A calibrated causal-video hypothesis score, never a disciplinary conclusion."""
from pathlib import Path
import math

from action_sampling import assert_sampling, sampling_profile

import torch
from torch import nn


class ActionHead(nn.Module):
    def __init__(self, hidden=0):
        super().__init__()
        self.layers = (nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Dropout(.4), nn.Linear(hidden, 1))
                       if hidden else nn.Linear(512, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class FrozenActionHead:
    def __init__(self, checkpoint_path, backbone_sha256):
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        if (checkpoint.get('architecture') != 'r3d18_frozen_mil'
                or checkpoint.get('r3d_weights_sha256') != backbone_sha256
                or checkpoint.get('hidden') not in (0, 64)):
            raise ValueError('Incompatible causal-video checkpoint')
        self.sampling_profile = sampling_profile(checkpoint, allow_legacy_missing=True)
        self.sample_fps = self.sampling_profile.sample_fps
        self.min_frames = self.sampling_profile.min_frames
        self.min_span = self.sampling_profile.min_span
        self.legacy_sampling_metadata = not any(key in checkpoint for key in
            ('sample_fps', 'min_frames', 'min_real_frames', 'min_span'))
        self.model = ActionHead(checkpoint['hidden']).eval()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.mean = checkpoint['mean'].detach().cpu().float()
        self.scale = checkpoint['scale'].detach().cpu().float()
        self.offset = float(checkpoint['offset'])
        if (self.mean.shape != (512,) or self.scale.shape != (512,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or torch.any(self.scale <= 0) or not math.isfinite(self.offset)
                or not all(torch.isfinite(p).all() for p in self.model.parameters())):
            raise ValueError('Nonfinite or malformed causal-video calibration')

    def assert_sampling(self, context_or_provenance, *, allow_legacy_missing=False):
        """Verify the exact trained sampling profile before inference/deployment."""
        return assert_sampling(self.sampling_profile, context_or_provenance,
                               allow_legacy_missing=allow_legacy_missing)

    def score(self, embeddings):
        x = torch.as_tensor(embeddings, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != 512 or not torch.isfinite(x).all():
            raise ValueError('Expected finite causal video embeddings [N,512]')
        with torch.inference_mode():
            logits = self.model((x-self.mean)/self.scale)-self.offset
            return torch.sigmoid(logits).cpu().numpy()
