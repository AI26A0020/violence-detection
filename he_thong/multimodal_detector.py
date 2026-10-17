"""
Unified multimodal violence detector -- the single inference entry point.

This is the R4 integration surface. `main.py` (person detection + optical flow)
calls one function and receives one decision per clip, rather than two
independent alerts that a human has to reconcile.

    from multimodal_detector import MultimodalDetector
    det = MultimodalDetector(mode="learned")
    result = det.analyze_clip("clip.mp4")   # -> Result
    print(result.is_violent, result.score, result.evidence)

Design notes
------------
* Lazy imports. Loading the ResNet at import time would slow down every
  `python main.py` start even when the violence model is never used, and would
  make the existing app fail to start if torch were missing. The 64 KB
  `main.py` is left untouched and keeps working on its own.
* Degrades honestly. No audio track -> visual score alone, flagged in the
  result. Untrained speech -> excluded, never imputed. A caller can always
  tell what evidence actually backed a decision.
* `result.evidence` records which modalities contributed, so an operator can
  see *why* an alert fired rather than receiving a bare boolean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

VISUAL_CKPT = ROOT / "models" / "violence_resnet18.pt"
AUDIO_MODEL = ROOT / "models" / "audio_model.json"
FUSION_MODEL = ROOT / "models" / "fusion_model.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Result:
    """One decision per clip, with the reasoning attached."""

    is_violent: bool
    score: float
    mode: str
    visual_score: float | None = None
    audio_score: float | None = None
    speech_score: float | None = None
    evidence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    available_modalities: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_violent": self.is_violent,
            "score": self.score,
            "mode": self.mode,
            "visual_score": self.visual_score,
            "audio_score": self.audio_score,
            "speech_score": self.speech_score,
            "evidence": self.evidence,
            "notes": self.notes,
            "available_modalities": self.available_modalities,
        }


def _finite(x: float) -> float:
    """NaN/inf would serialise to invalid JSON and poison a threshold compare."""
    return float(x) if x == x and abs(x) != float("inf") else 0.0


class MultimodalDetector:
    def __init__(
        self,
        mode: str = "learned",
        visual_threshold: float = 0.5,
        audio_threshold: float = 0.85,
        combine: str = "or",
        device: str = "cpu",
    ) -> None:
        if mode not in ("rule", "learned", "visual_only"):
            raise ValueError(f"unknown mode {mode!r}; expected rule|learned|visual_only")
        self.mode = mode
        self.visual_threshold = visual_threshold
        self.audio_threshold = audio_threshold
        self.combine = combine
        self.device = device
        self._model: Any = None
        self._tf: Any = None
        self._audio: Any = None
        self._fusion: Any = None
        self._fusion_keys: list[str] = []

    # ---------------- lazy loaders ----------------

    def _load_visual(self) -> bool:
        if self._model is not None:
            return True
        if not VISUAL_CKPT.exists():
            return False
        import torch
        from PIL import Image  # noqa: F401  (ensures PIL is a real dependency)
        from torchvision import transforms

        from train_model import build_model

        m = build_model()
        m.load_state_dict(torch.load(VISUAL_CKPT, map_location=self.device))
        m.to(self.device).eval()
        self._model = m
        self._tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return True

    def _load_audio(self) -> bool:
        if self._audio is not None:
            return True
        if not AUDIO_MODEL.exists():
            return False
        # AudioScorer lives in fusion.py; there is no AudioModel class.
        from fusion import AudioScorer

        self._audio = AudioScorer(AUDIO_MODEL)
        return True

    def _load_fusion(self) -> bool:
        if self._fusion is not None:
            return True
        if self.mode != "learned" or not FUSION_MODEL.exists():
            return False
        self._fusion = json.loads(FUSION_MODEL.read_text(encoding="utf-8"))
        self._fusion_keys = self._fusion.get("feature_names", [])
        return True

    # ---------------- scorers ----------------

    def score_frames(self, frame_paths: list[Path]) -> float | None:
        """Mean violence probability over representative frames of one clip."""
        if not self._load_visual() or not frame_paths:
            return None
        import torch
        from PIL import Image

        # Sample at most 3 frames: frames from one video are near-duplicates,
        # and scoring all of them costs CPU without adding information.
        step = max(1, len(frame_paths) // 3)
        sample = frame_paths[::step][:3]
        probs: list[float] = []
        with torch.no_grad():
            for p in sample:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                logits = self._model(self._tf(img).unsqueeze(0))
                probs.append(float(torch.softmax(logits, dim=1)[0, 1]))
        return _finite(sum(probs) / len(probs)) if probs else None

    def score_audio(self, audio_path: Path) -> float | None:
        """Violence probability from audio alone. None = no audio evidence."""
        if not self._load_audio():
            return None
        if not Path(audio_path).exists():
            return None
        try:
            from audio_features import compute_features, decode_audio

            audio, _sr = decode_audio(Path(audio_path))
            if audio is None:
                return None
            feats = compute_features(audio)
        except Exception:
            # Any decode/feature failure means "no audio evidence", never 0.0.
            return None
        if not feats:
            return None
        try:
            s = self._audio.score_row(feats)
        except Exception:
            return None
        return _finite(s) if s is not None else None

    # ---------------- combination ----------------

    def combine_scores(self, v: float | None, a: float | None) -> tuple[bool, float, list[str]]:
        """-> (is_violent, score, evidence). Never imputes a missing modality."""
        evidence: list[str] = []

        if self.mode == "visual_only":
            if v is None:
                return False, 0.0, ["no usable visual frames"]
            return v >= self.visual_threshold, v, [f"visual={v:.3f}"]

        if v is None and a is None:
            return False, 0.0, ["no usable visual or audio evidence"]

        if self.mode == "rule":
            # Handle the missing-modality cases BEFORE the OR/AND arithmetic.
            # Doing it after defaulted the absent modality to 0.0, which is a
            # safe value for OR but an accident rather than a decision.
            if self.combine == "and" and (v is None or a is None):
                present = v if v is not None else a
                return False, float(present), ["AND needs both modalities; one missing"]
            if v is None:
                return a >= self.audio_threshold, a, [f"audio={a:.3f} (no frames)"]
            if a is None:
                return v >= self.visual_threshold, v, [f"visual={v:.3f} (no audio)"]
            fired = ((v >= self.visual_threshold and a >= self.audio_threshold)
                     if self.combine == "and"
                     else (v >= self.visual_threshold or a >= self.audio_threshold))
            return fired, max(v, a), [
                f"rule({self.combine}) visual={v:.3f} audio={a:.3f}"]

        # learned
        if self._load_fusion() and v is not None and a is not None:
            # The persisted weights were fitted on *standardised* inputs, so the
            # scaler has to be applied; skipping it silently mis-scores every row.
            order = self._fusion.get("input", ["visual_score", "audio_score"])
            vals = {"visual_score": v, "audio_score": a}
            mean = self._fusion.get("scaler_mean", [0.0, 0.0])
            scale = self._fusion.get("scaler_scale", [1.0, 1.0])
            scale = [s if s else 1.0 for s in scale]
            row = [(vals.get(k, 0.0) - m) / s for k, m, s in zip(order, mean, scale)]
            w = self._fusion.get("coef") or []
            b = self._fusion.get("intercept", 0.0)
            if w and len(w) == len(row):
                z = sum(wi * ri for wi, ri in zip(w, row)) + float(b)
                import math

                s_score = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                evidence.append(f"learned-fusion visual={v:.3f} audio={a:.3f} -> {s_score:.3f}")
                return s_score >= 0.5, s_score, evidence
            evidence.append("fusion model malformed; falling back to visual")
            return v >= self.visual_threshold, v, evidence

        src = v if v is not None else a
        evidence.append(f"learned fusion unavailable; using {'visual' if v is not None else 'audio'}")
        return src >= self.visual_threshold, src, evidence

    # ---------------- public API ----------------

    def analyze_clip(self, clip_path: Path, frame_paths: list[Path] | None = None) -> Result:
        clip_path = Path(clip_path)
        if frame_paths is None:
            frame_paths = _frames_for(clip_path)

        v = self.score_frames(frame_paths)
        a = self.score_audio(clip_path)

        is_violent, score, evidence = self.combine_scores(v, a)
        avail = [n for n, s in (("visual", v), ("audio", a)) if s is not None]
        notes: list[str] = []
        if a is None:
            notes.append("no usable audio for this clip; decision rests on video alone")
        if not frame_paths:
            notes.append("no frames supplied for this clip")

        try:
            import speech as speech_mod

            if not speech_mod.is_trained():
                notes.append("distressed-speech modality: not trained, excluded")
        except ImportError:
            pass

        return Result(
            is_violent=is_violent,
            score=score,
            mode=self.mode,
            visual_score=v,
            audio_score=a,
            evidence=evidence,
            notes=notes,
            available_modalities=avail,
        )


def _frames_for(clip_path: Path) -> list[Path]:
    """Best-effort: match a clip to its extracted frames, if build_dataset ran."""
    img_dir = ROOT / "data" / "images"
    if not img_dir.exists():
        return []
    key = clip_path.stem
    out: list[Path] = []
    for cls in ("violence", "nonviolence"):
        d = img_dir / cls
        if not d.exists():
            continue
        for p in d.rglob(f"{key}_*.jpg"):
            out.append(p)
        if out:
            break
    return sorted(out)


def _selftest() -> int:
    print("=" * 68)
    for mode in ("rule", "learned", "visual_only"):
        det = MultimodalDetector(mode=mode)
        v, a = 0.90, 0.20
        fired, s, ev = det.combine_scores(v, a)
        print(f"{mode:12s} v=0.90 a=0.20 -> violent={fired} score={s:.3f} {ev}")
        fired2, s2, ev2 = det.combine_scores(v, None)
        print(f"{mode:12s} v=0.90 a=None -> violent={fired2} score={s2:.3f} {ev2}")
        fired3, s3, ev3 = det.combine_scores(None, None)
        print(f"{mode:12s} v=None a=None -> violent={fired3} score={s3:.3f} {ev3}")
    print("-" * 68)
    det = MultimodalDetector(mode="rule")
    r = det.analyze_clip(Path("nonexistent_clip.mp4"))
    print("missing-file clip ->", r.as_dict())
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
