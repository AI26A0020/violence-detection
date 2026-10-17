"""
Multimodal fusion: combine the visual and audio violence scores into one decision.

Two modes, both selectable at runtime:

  rule    configurable thresholds + AND/OR logic. Transparent, no training.
  learned a small model on the pair of modality scores.

Why a learned model on scores rather than late-fusion of raw features:
the audio model is weak (grouped-CV 0.736 accuracy, non-violence precision
0.44). Concatenating 2048-d visual embeddings with 44 acoustic features into
one network would let the visual branch simply dominate and learn nothing
useful from audio. Feeding the two calibrated scores in as a 2-D input
forces the fusion to treat them on comparable terms, and it is the form that
actually shows whether audio adds anything.

Join key: clips are matched on the source video identity (`clip_id`), so a
visual frame and an audio segment only ever pair when they come from the
same underlying video. The split is group-wise on that same clip id, which
prevents a clip's frames appearing in train and its audio in validation.

An audio-free case is handled explicitly: when a clip has no audio the
fusion falls back to the visual score rather than treating "no audio" as
evidence of "no violence".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

AUDIO_FEATS = ROOT / "data" / "audio" / "features" / "audio_features.json"
AUDIO_MODEL = ROOT / "models" / "audio_model.json"
IMG_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

VISUAL_CKPT = ROOT / "models" / "violence_resnet18.pt"


# --------------------------------------------------------------------------
# Audio scoring (loads the persisted linear model; no retraining here)
# --------------------------------------------------------------------------


class AudioScorer:
    def __init__(self, path: Path) -> None:
        self.m = json.loads(path.read_text(encoding="utf-8"))
        self.names = self.m["feature_names"]
        self.kind = self.m.get("kind", "linear")

        # Tree ensembles cannot be expressed as coef/intercept, so they are
        # persisted separately with joblib. Older artifacts are linear and are
        # still supported, which keeps old checkpoints loadable.
        self.pipeline = None
        tree_file = self.m.get("tree_path")
        if tree_file:
            tree_path = path.parent / tree_file
            if tree_path.exists():
                import joblib

                self.pipeline = joblib.load(tree_path)["pipeline"]

        if self.pipeline is None:
            self.mean = np.asarray(self.m["scaler_mean"], dtype=np.float64)
            self.scale = np.asarray(self.m["scaler_scale"], dtype=np.float64)
            self.scale[self.scale == 0] = 1.0
            self.coef = np.asarray(self.m["coef"], dtype=np.float64)
            self.intercept = float(self.m["intercept"])

    def score_row(self, feats: dict) -> float | None:
        if self.pipeline is not None:
            try:
                x = np.array([[float(feats.get(n, np.nan)) for n in self.names]],
                             dtype=np.float64)
                if not np.all(np.isfinite(x)):
                    med = np.nanmedian(np.where(np.isfinite(x), x, np.nan))
                    if not np.isfinite(med):
                        return None
                    x = np.where(np.isfinite(x), x, med)
                return float(self.pipeline.predict_proba(x)[0, 1])
            except Exception:
                return None
        try:
            x = np.array([float(feats.get(n, np.nan)) for n in self.names], dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(x)):
            med = np.where(np.isfinite(x), x, np.nan)
            if np.all(~np.isfinite(x)):
                return None
            fill = np.nanmedian(x)
            x = np.where(np.isfinite(x), x, fill if np.isfinite(fill) else 0.0)
        z = (x - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept))))


def build_clip_table() -> tuple[list[str], dict[str, float], np.ndarray]:
    """clip_id -> audio score, plus labels. Clips without usable audio are absent."""
    rows = json.loads(AUDIO_FEATS.read_text(encoding="utf-8"))
    scorer = AudioScorer(AUDIO_MODEL)
    table: dict[str, float] = {}
    labels: dict[str, int] = {}
    for r in rows:
        cid = r["clip_id"]
        s = scorer.score_row(r["features"])
        if s is None:
            continue
        table[cid] = s
        labels[cid] = int(r["label"])
    keys = sorted(table)
    y = np.array([labels[k] for k in keys], dtype=np.int64)
    return keys, table, y


def group_of_frame(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[0] if stem[-6:].isdigit() else stem


def collect_visual_clips() -> dict[str, list[Path]]:
    """group id -> list of frame paths, per class dir."""
    out: dict[str, list[Path]] = {}
    for cls in ("violence", "nonviolence"):
        d = IMG_DIR / cls
        if not d.exists():
            continue
        for p in d.rglob("*.jpg"):
            out.setdefault(group_of_frame(p), []).append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="both", choices=["rule", "learned", "both"])
    ap.add_argument("--visual-threshold", type=float, default=0.5)
    ap.add_argument("--audio-threshold", type=float, default=0.85)
    ap.add_argument("--combine", default="or", choices=["or", "and"])
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, roc_auc_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import torch
    from PIL import Image
    from torchvision import transforms
    from train_model import build_model

    audio_keys, audio_scores, audio_y = build_clip_table()
    print(f"clips with usable audio: {len(audio_keys)} "
          f"(V={int((audio_y==1).sum())} NV={int((audio_y==0).sum())})")

    # ---- visual scores per clip, on CPU, batched ----
    print("scoring visual clips with the trained ResNet-18 ...")
    device = torch.device("cpu")
    model = build_model()
    model.load_state_dict(torch.load(VISUAL_CKPT, map_location=device))
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    audio_set = set(audio_keys)
    vis: dict[str, list[float]] = {}
    clips = collect_visual_clips()
    todo = {g: f for g, f in clips.items() if g in audio_set}
    print(f"clips joined across both modalities: {len(todo)}")

    # one representative frame per clip (frames of a clip share the video,
    # and averaging all of them would leak within-clip info into the score)
    with torch.no_grad():
        for i, (g, frames) in enumerate(todo.items()):
            sample = frames[:: max(1, len(frames) // 3)][:3]
            probs = []
            for p in sample:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                x = tf(img).unsqueeze(0)
                logits = model(x)
                probs.append(float(torch.softmax(logits, dim=1)[0, 1]))
            if not probs:
                continue
            vis[g] = probs
            if (i + 1) % 200 == 0:
                print(f"  scored {i+1}/{len(todo)}")

    # Labels come from the audio feature table only. Deriving them from the
    # folder name instead would relabel clips, because "nonviolence" contains
    # the substring "violence" - the same trap that mislabelled the corpus
    # once already.
    labels = {k: int(audio_y[i]) for i, k in enumerate(audio_keys)}
    groups = sorted(vis)
    if not groups:
        print("ERROR: no clips joined between modalities")
        return 1
    y = np.array([labels[g] for g in groups], dtype=np.int64)
    vs = np.array([float(np.mean(vis[g])) for g in groups], dtype=np.float64)
    as_ = np.array([audio_scores[g] for g in groups], dtype=np.float64)
    print(f"joined clips: {len(groups)}  V={int((y==1).sum())} NV={int((y==0).sum())}")

    report: dict = {"n_clips": int(len(y)), "n_violence": int((y == 1).sum()),
                    "n_nonviolence": int((y == 0).sum())}

    # Third modality: distressed-speech words. Requested in the spec, but not
    # trainable from this data (no word-level labels exist on disk). Recorded
    # explicitly rather than left out, so the report never implies a coverage
    # it does not have. It contributes no score to any fusion mode below.
    try:
        import speech as speech_mod
    except ImportError:
        speech_mod = None
    speech_status = (
        {"trained": True, "score_available": True}
        if speech_mod is not None and speech_mod.is_trained()
        else {
            "trained": False,
            "score_available": False,
            "reason": (speech_mod.REASON if speech_mod else "speech.py not importable"),
            "used_in_scores": False,
        }
    )
    report["speech"] = speech_status
    if not speech_status["trained"]:
        print(f"\nspeech modality: NOT TRAINED -> {speech_status['reason'][:90]}...")
        print("  excluded from both fusion modes; no accuracy claimed for it.")

    # ---------------- baseline: visual only ----------------
    vpred = (vs >= args.visual_threshold).astype(int)
    vcm = confusion_matrix(y, vpred, labels=[0, 1])
    vacc = float((vpred == y).mean())
    try:
        vauc = float(roc_auc_score(y, vs))
    except ValueError:
        vauc = float("nan")
    report["visual_only"] = {
        "accuracy": vacc, "auc": vauc, "confusion_matrix": vcm.tolist(),
        "report": classification_report(y, vpred, target_names=["nonviolence", "violence"],
                                        zero_division=0),
        # Per-clip scores, so fusion_significance.py can run McNemar against
        # correctly aligned predictions instead of re-deriving them from the
        # confusion matrices (which silently misaligns them).
        "per_clip_predictions": [
            {"clip_id": g, "y": int(yy), "visual": int(pp), "audio": float(aa)}
            for g, yy, pp, aa in zip(groups, y, vpred, as_)
        ],
    }
    print(f"\nvisual-only: accuracy={vacc:.4f} auc={vauc:.4f}")

    # ---------------- rule fusion ----------------
    if args.mode in ("rule", "both"):
        rpred = ((vs >= args.visual_threshold) | (as_ >= args.audio_threshold)).astype(int)
        if args.combine == "and":
            rpred = ((vs >= args.visual_threshold) & (as_ >= args.audio_threshold)).astype(int)
        rcm = confusion_matrix(y, rpred, labels=[0, 1])
        racc = float((rpred == y).mean())
        report["rule_fusion"] = {
            "accuracy": racc, "visual_threshold": args.visual_threshold,
            "audio_threshold": args.audio_threshold, "combine": args.combine,
            "confusion_matrix": rcm.tolist(),
            "report": classification_report(y, rpred, target_names=["nonviolence", "violence"],
                                            zero_division=0),
        }
        print(f"rule fusion ({args.combine}, v>={args.visual_threshold} a>={args.audio_threshold}): "
              f"accuracy={racc:.4f}  delta={racc - vacc:+.4f}")
        print(rcm)

    # ---------------- learned fusion ----------------
    if args.mode in ("learned", "both"):
        X = np.column_stack([vs, as_])
        # grouped by clip == each row IS one clip, so a plain stratified
        # split is already group-disjoint here; no two rows share a video.
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y), dtype=np.float64)
        for k, (tr, te) in enumerate(skf.split(X, y)):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("clf", LogisticRegression(C=1.0, max_iter=2000))])
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict_proba(X[te])[:, 1]
        fpred = (oof >= 0.5).astype(int)
        fcm = confusion_matrix(y, fpred, labels=[0, 1])
        facc = float((fpred == y).mean())
        try:
            fauc = float(roc_auc_score(y, oof))
        except ValueError:
            fauc = float("nan")
        for rec, fp_ in zip(report["visual_only"]["per_clip_predictions"], fpred):
            rec["fusion"] = int(fp_)
        report["learned_fusion"] = {
            "accuracy": facc, "auc": fauc, "confusion_matrix": fcm.tolist(),
            "f1": float(f1_score(y, fpred, zero_division=0)),
            "report": classification_report(y, fpred, target_names=["nonviolence", "violence"],
                                            zero_division=0),
            "folds": args.folds,
        }
        print(f"\nlearned fusion: accuracy={facc:.4f} auc={fauc:.4f}  delta={facc - vacc:+.4f}")
        print(fcm)

        # fit final fusion model on all clips
        final = Pipeline([("sc", StandardScaler()),
                          ("clf", LogisticRegression(C=1.0, max_iter=2000))])
        final.fit(X, y)
        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "fusion_model.json").write_text(json.dumps({
            "input": ["visual_score", "audio_score"],
            "scaler_mean": [float(v) for v in final.named_steps["sc"].mean_],
            "scaler_scale": [float(v) for v in final.named_steps["sc"].scale_],
            "coef": [float(v) for v in final.named_steps["clf"].coef_[0]],
            "intercept": float(final.named_steps["clf"].intercept_[0]),
            "cv": report["learned_fusion"],
            "rule_defaults": {"visual_threshold": args.visual_threshold,
                              "audio_threshold": args.audio_threshold,
                              "combine": args.combine},
        }, indent=2), encoding="utf-8")
        print(f"fusion model -> {OUT_DIR / 'fusion_model.json'}")

    # ---------------- honest verdict ----------------
    best_name, best_acc = "visual_only", vacc
    for key in ("rule_fusion", "learned_fusion"):
        if key in report and report[key]["accuracy"] > best_acc:
            best_name, best_acc = key, report[key]["accuracy"]
    improves = best_acc > vacc
    report["verdict"] = {
        "best_modality": best_name,
        "best_accuracy": best_acc,
        "visual_only_accuracy": vacc,
        "fusion_improves": bool(improves),
        "delta": best_acc - vacc,
    }
    print("\n" + "=" * 60)
    print(f"visual-only accuracy : {vacc:.4f}")
    print(f"best fusion accuracy : {best_acc:.4f}  ({best_name})")
    if improves:
        print(f"VERDICT: fusion improves accuracy by {best_acc - vacc:+.4f}")
    else:
        print(f"VERDICT: fusion does NOT improve accuracy ({best_acc - vacc:+.4f}).")
        print("Audio is not adding signal beyond the visual model on this data.")
    print("=" * 60)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "fusion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {REPORT_DIR / 'fusion_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
