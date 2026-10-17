"""Versioned image-head fallback over the same frozen visual features."""
import math
import torch
from torch import nn


class ConfidenceFallbackHead(nn.Module):
    """Use the adapted head at its confident ends and the old head in between.

    This is a development-calibrated ensemble, not a confidence guarantee.
    Both heads see only the current 512D image feature; source metadata and
    evaluation labels are never inputs. The fallback weights live in the same
    checkpoint, so changing another model file cannot alter this prediction.
    """
    def __init__(self, primary, specification):
        super().__init__()
        if not isinstance(specification, dict) or specification.get('kind') != 'confidence_fallback_v1':
            raise ValueError('Unsupported image-head selector')
        low, high = specification.get('low'), specification.get('high')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (low, high)) or not 0 <= low < high <= 1:
            raise ValueError('Invalid image-head confidence thresholds')
        weight, bias = specification.get('weight'), specification.get('bias')
        if (not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor)
                or weight.shape != (2, 512) or bias.shape != (2,)
                or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
            raise ValueError('Invalid fallback image-head weights')
        self.primary = primary
        self.fallback = nn.Linear(512, 2)
        with torch.no_grad():
            self.fallback.weight.copy_(weight)
            self.fallback.bias.copy_(bias)
        self.low, self.high = float(low), float(high)

    def forward(self, features):
        logits = self.primary(features)
        probability = logits.softmax(-1)[:, 1:2]
        selected = (probability <= self.low) | (probability >= self.high)
        return torch.where(selected, logits, self.fallback(features))
