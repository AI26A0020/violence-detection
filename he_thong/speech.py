"""
Distressed-speech detection -- PLUGGABLE INTERFACE, DELIBERATELY NOT TRAINED.

Why this file exists but contains no model
------------------------------------------
The requirement list asked for three kinds of audio signal. Two of them
(screaming/shouting/impact loudness, and temporal escalation) are learnable
from what is actually on disk, and both are trained in audio_model.py. The
third -- recognising *words* of distress ("help", "stop", "leave me alone") --
is not.

There is no speech transcript, no word-level annotation and no labelled
speech corpus anywhere in data_raw/. UCF-Crime and RPL ship video with an
audio track and a violence/non-violence label attached to the *clip*, not to
what was said in it. A word classifier cannot be fitted to a single binary
label per clip: there is no supervision signal that says which words occurred.

So this module does the one honest thing available. It defines the interface,
it refuses to produce a score, and it says so loudly. The alternative -- wiring
in an untrained network that emits a confident 0.0-1.0 number -- would be the
most dangerous kind of bug in this project, because a fabricated score gets
fed into fusion and looks like evidence. A missing capability that reports
itself is merely inconvenient; a fabricated one is silently wrong.

To make this real, implement the three marked hooks below and point
MODEL_PATH at a checkpoint trained on labelled speech. Nothing else changes;
fusion.py picks up the module automatically once TRAINED is True.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "speech_model.json"

# Flips only when a real checkpoint is trained. Never set by hand.
TRAINED = False

REASON = (
    "not trained - no labelled speech data available. No dataset in "
    "data_raw/ contains word-level or transcript annotations; the "
    "violence/non-violence label is attached to the clip, not to speech "
    "content. No accuracy figure is reported for this capability because "
    "none can be honestly computed."
)


@dataclass
class SpeechFeatures:
    """What a real speech backend would need to return."""

    # e.g. {"help": 0.31, "stop": 0.12, ...}
    distress_words: dict[str, float] = field(default_factory=dict)
    speech_ratio: float = 0.0
    n_speech_segments: int = 0
    model_version: str = "none"


@dataclass
class SpeechResult:
    """Uniform return type, so callers need no special-casing."""

    trained: bool
    score: float | None  # None == no claim being made. Never 0.0 as a stand-in.
    reason: str
    features: SpeechFeatures | None = None

    def usable(self) -> bool:
        return self.trained and self.score is not None


def unavailable() -> SpeechResult:
    """The single source of truth for 'this capability does not exist yet'."""
    return SpeechResult(trained=False, score=None, reason=REASON, features=None)


def is_trained() -> bool:
    """True only when a checkpoint genuinely exists on disk."""
    return TRAINED and MODEL_PATH.exists()


# --------------------------------------------------------------------------
# Hooks. Implement these against a real ASR backend to activate the path.
# --------------------------------------------------------------------------


def transcribe(audio_path: Path) -> list[tuple[float, float, str]]:
    """-> [(start_sec, end_sec, text), ...]

    Not implemented: requires an ASR backend (whisper / vosk) and Vietnamese
    or English acoustic models. Neither is installed and neither is implied by
    the data on disk.
    """
    raise NotImplementedError(
        "transcribe() requires an ASR backend. Install one (e.g. faster-whisper) "
        "and implement this hook. See REASON for why it is not active."
    )


def extract_distress_lexicon(texts: list[str]) -> dict[str, float]:
    """Map transcript segments to distress-word scores.

    Not implemented: needs a decision on which languages to cover and which
    words count as distress. Vietnamese has no public violence-speech lexicon;
    building one requires labelled audio, which is the thing we lack.
    """
    raise NotImplementedError(
        "extract_distress_lexicon() needs a curated distress lexicon and "
        "labelled data to validate it."
    )


def train(model_path: Path = MODEL_PATH) -> int:
    """Entry point for future training. Returns a process exit code.

    Refuses to run rather than emitting a model that cannot be validated.
    """
    print(REASON)
    print("\nNo speech model was written. To proceed you must:")
    print("  1. obtain labelled distress-speech audio (word or phrase level)")
    print("  2. implement transcribe() and extract_distress_lexicon()")
    print("  3. train and save to", model_path)
    print("  4. set TRAINED = True only after a leakage-free eval")
    return 1


def score(audio_path: Path | None) -> SpeechResult:
    """Public entry point used by fusion.

    Returns score=None while untrained. Fusion must treat None as 'no audio
    evidence available', NOT as 0.0 (no distress) and NOT as 'not violent'.
    """
    if not is_trained():
        return unavailable()
    try:  # pragma: no cover - unreachable until a checkpoint exists
        raise NotImplementedError("score() not wired to a trained checkpoint yet")
    except NotImplementedError as e:
        return SpeechResult(trained=False, score=None, reason=str(e))


def describe() -> str:
    status = "TRAINED" if is_trained() else "NOT TRAINED"
    return f"distressed-speech detection: {status}\n  {REASON}"


def _report() -> None:
    print("=" * 68)
    print(describe())
    r = score(None)
    print(f"  runtime result   : score={r.score!r} trained={r.trained}")
    print(f"  accuracy reported: NONE (nothing was trained)")
    print("=" * 68)


if __name__ == "__main__":
    raise SystemExit(train())
