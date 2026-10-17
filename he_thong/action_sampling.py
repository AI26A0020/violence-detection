"""Strict sampling-profile compatibility; standard library only, no model imports."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class SamplingProfile:
    sample_fps: int
    min_frames: int
    min_span: float

    def as_dict(self):
        return asdict(self)


LEGACY_PROFILE = SamplingProfile(8, 8, .875)
FIXED16_PROFILE = SamplingProfile(16, 16, .9375)
_MISSING = object()


def sampling_profile(source, *, allow_legacy_missing=False):
    """Read checkpoint/context/cache aliases; reject partial or unsupported metadata.

    All three fields absent may explicitly mean the original 8fps profile.
    min_frames (checkpoint/context) and min_real_frames (cache) are aliases;
    declaring both with conflicting values is an error.
    """
    def get(name):
        return source.get(name, _MISSING) if isinstance(source, Mapping) else getattr(source, name, _MISSING)
    fps, minimum, alias, span = (get(name) for name in ('sample_fps', 'min_frames', 'min_real_frames', 'min_span'))
    values = (fps, minimum, alias, span)
    if all(value is _MISSING for value in values):
        if allow_legacy_missing:
            return LEGACY_PROFILE
        raise ValueError('Missing causal action sampling metadata')
    if minimum is _MISSING:
        minimum = alias
    elif alias is not _MISSING and alias != minimum:
        raise ValueError('Conflicting minimum-frame sampling declarations')
    if any(value is _MISSING for value in (fps, minimum, span)):
        raise ValueError('Partial causal action sampling metadata')
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
           for value in (fps, minimum, span)):
        raise ValueError('Sampling metadata must contain finite numeric values')
    if fps != int(fps) or minimum != int(minimum):
        raise ValueError('Sampling rate and minimum real-frame count must be integers')
    candidate = SamplingProfile(int(fps), int(minimum), float(span))
    for supported in (LEGACY_PROFILE, FIXED16_PROFILE):
        if (candidate.sample_fps == supported.sample_fps and candidate.min_frames == supported.min_frames
                and abs(candidate.min_span-supported.min_span) <= 1e-10):
            if get('max_real_frames') is not _MISSING:
                maximum = get('max_real_frames')
                if isinstance(maximum, bool) or not isinstance(maximum, Real) or maximum != 16:
                    raise ValueError('Unsupported maximum real-frame count')
            return supported
    raise ValueError('Unsupported causal action sampling profile')


def assert_sampling(expected, actual, *, allow_legacy_missing=False):
    """Return the matching profile or reject a context/cache/head mismatch."""
    expected = sampling_profile(expected)
    actual = sampling_profile(actual, allow_legacy_missing=allow_legacy_missing)
    if expected != actual:
        raise ValueError('Causal action sampling profile does not match learned head')
    return actual
