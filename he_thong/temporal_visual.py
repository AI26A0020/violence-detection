"""Small causal video head over frozen image features; no student identity data."""
import torch
from torch import nn
from collections import deque
import math
import time

WINDOW = 12
MIN_FRAMES = 6
SAMPLE_FPS = 5


class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 64), nn.ReLU())
        self.sequence = nn.GRU(64, 48, batch_first=True)
        self.classify = nn.Sequential(nn.Linear(48 + 64 * 3, 64), nn.ReLU(), nn.Dropout(.25), nn.Linear(64, 1))

    def forward(self, features):
        x = self.project(features)
        _, hidden = self.sequence(x)
        # Keep order in GRU, plus scene context and change over the same causal window.
        summary = torch.cat((hidden[-1], x.mean(1), x.std(1, unbiased=False), (x[:, 1:] - x[:, :-1]).abs().mean(1)), 1)
        return self.classify(summary).squeeze(-1)


def causal_windows(features, stride=2):
    """Only frames already seen enter a window. Warm-up is never fabricated."""
    if len(features) < MIN_FRAMES:
        return features.new_empty((0, WINDOW, 512))
    ends = list(range(MIN_FRAMES, len(features) + 1, stride))
    if ends[-1] != len(features):
        ends.append(len(features))
    windows = []
    for end in ends:
        seq = features[max(0, end - WINDOW):end]
        if len(seq) < WINDOW:
            seq = torch.cat((seq[:1].expand(WINDOW-len(seq), -1), seq))
        windows.append(seq)
    return torch.stack(windows)


class CausalScorer:
    """All temporal state belongs to the camera, never to the shared weights."""
    def __init__(self, head, offset, device='cpu'):
        if not math.isfinite(offset):
            raise ValueError('Invalid temporal calibration')
        self.head, self.offset, self.device = head.eval(), float(offset), device

    def score(self, feature, state, stamp=None):
        frames = getattr(state, 'temporal_frames', None)
        if frames is None:
            frames = state.temporal_frames = deque(maxlen=80)
        stamp = float(stamp) if stamp is not None else time.monotonic()
        if not math.isfinite(stamp):
            raise ValueError('Invalid video timestamp')
        if frames and (stamp <= frames[-1][0] or stamp-frames[-1][0] > 1.):
            frames.clear()
            state.smooth = None
            state.image_smooth = None
        # Cache and runtime use the same compact feature precision.
        feature = torch.as_tensor(feature).detach().reshape(512).to(torch.float16).to(torch.float32).cpu()
        if not torch.isfinite(feature).all():
            raise ValueError('Invalid video features')
        frames.append((stamp, feature))
        while frames and stamp-frames[0][0] > 2.6:
            frames.popleft()
        if len(frames) < MIN_FRAMES or stamp-frames[0][0]+1e-6 < (MIN_FRAMES-1)/SAMPLE_FPS:
            return None, None
        targets = [stamp-(WINDOW-1-i)/SAMPLE_FPS for i in range(WINDOW)]
        # Resampling uses only observations already received, including when inference FPS varies.
        sequence = torch.stack([min(frames, key=lambda r:abs(r[0]-t))[1] for t in targets])
        with torch.inference_mode():
            logit = self.head(sequence.unsqueeze(0).to(self.device))[0]
            raw = float(torch.sigmoid(logit-self.offset).cpu())
        if not math.isfinite(raw):
            raise ValueError('Invalid temporal prediction')
        state.smooth = raw if state.smooth is None else .4*raw+.6*state.smooth
        return raw, state.smooth
