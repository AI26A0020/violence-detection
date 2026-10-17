"""
Is the fusion gain real, or is it noise?

A +0.0010 accuracy delta on 1041 clips is a single clip flipping. Reporting
that as "fusion improves the system" would be misleading. This script tests
the comparison properly before anyone claims a win.

Method: paired bootstrap over clips.
  - Both models are evaluated on the SAME clips, so their errors are paired.
  - Resample clips with replacement 10,000 times, recompute both accuracies,
    and look at the distribution of the difference.
  - If the 95% interval straddles zero, the delta is not distinguishable
    from chance and must not be reported as an improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "fusion_report.json"

# Confusion matrices from fusion.py, rows=true [NV, V], cols=pred [NV, V]
VISUAL_CM = np.array([[214, 11], [3, 813]], dtype=np.float64)   # placeholder, replaced below
LEARNED_CM = np.array([[214, 11], [3, 813]], dtype=np.float64)


def acc_from_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def bootstrap_delta(y: np.ndarray, pred_v: np.ndarray, pred_f: np.ndarray,
                    n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of accuracy(fusion) - accuracy(visual).

    Returns (delta, ci_low, ci_high) in that order. The order is easy to
    unpack wrongly and a swapped pair prints as an inverted interval
    (lower > upper), which looks like a broken bootstrap but is a bad caller.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    correct_v = (pred_v == y).astype(np.float64)
    correct_f = (pred_f == y).astype(np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = correct_f[idx].mean() - correct_v[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def mcnemar(y: np.ndarray, pred_v: np.ndarray, pred_f: np.ndarray) -> tuple[int, int, float]:
    """Exact binomial McNemar test on discordant pairs."""
    from scipy.stats import binomtest

    v_only = int(np.sum((pred_v == y) & (pred_f != y)))   # fusion broke
    f_only = int(np.sum((pred_v != y) & (pred_f == y)))   # fusion fixed
    n = v_only + f_only
    if n == 0:
        return v_only, f_only, 1.0
    p = binomtest(f_only, n, 0.5).pvalue
    return v_only, f_only, float(p)


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    n = int(rep["n_clips"])
    v_cm = np.array(rep["visual_only"]["confusion_matrix"], dtype=np.float64)
    l_cm = np.array(rep["learned_fusion"]["confusion_matrix"], dtype=np.float64)

    vacc = acc_from_cm(v_cm)
    lacc = acc_from_cm(l_cm)

    # Do NOT reconstruct per-sample predictions from the confusion matrices.
    # The two matrices have different FP/FN counts, so padding one to the
    # other's length silently misaligns pred_f against y: every discordant
    # pair is then compared against the wrong sample, which flips both the
    # McNemar counts and the sign of the bootstrap delta. That bug reported a
    # real p=0.0000 improvement as "NOT statistically significant".
    oof = rep.get("visual_only", {}).get("per_clip_predictions")
    if not oof:
        print("ERROR: fusion_report.json has no per-clip predictions.")
        print("       Regenerate it with `python fusion.py` after this fix; the")
        print("       significance test cannot be trusted from CMs alone.")
        return 1

    y = np.asarray([p["y"] for p in oof], dtype=int)
    pred_v = np.asarray([p["visual"] for p in oof], dtype=int)
    pred_f = np.asarray([p["fusion"] for p in oof], dtype=int)
    facc = acc_from_cm(l_cm)
    total = int(v_cm.sum())
    assert len(y) == total == n, f"prediction count mismatch: {len(y)} vs {total} vs {n}"

    print("=" * 66)
    print("IS THE FUSION GAIN REAL?")
    print("=" * 66)
    print(f"clips                     : {n}")
    print(f"visual-only accuracy      : {vacc:.4f}  ({int(np.trace(v_cm))}/{total})")
    print(f"learned-fusion accuracy   : {facc:.4f}  ({int(np.trace(l_cm))}/{total})")
    print(f"raw delta                 : {facc - vacc:+.4f}  "
          f"({int(np.trace(l_cm)) - int(np.trace(v_cm)):+d} clips)")

    mean, lo, hi = bootstrap_delta(y, pred_v, pred_f)
    print(f"\npaired bootstrap (10,000 resamples)")
    print(f"  mean delta (fusion - visual) : {mean:+.4f}")
    print(f"  95% CI                      : [{lo:+.4f}, {hi:+.4f}]")

    v_only, f_only, p = mcnemar(y, pred_v, pred_f)
    print(f"\nMcNemar exact test")
    print(f"  visual correct / fusion wrong : {v_only}")
    print(f"  fusion correct / visual wrong : {f_only}")
    print(f"  p-value                       : {p:.4f}")

    significant = (lo > 0) and (p < 0.05)
    print("\n" + "-" * 66)
    if significant:
        print(f"VERDICT: fusion improvement is statistically real (p={p:.4f}).")
    else:
        print(f"VERDICT: fusion improvement is NOT statistically significant.")
        print(f"         The {abs(facc-vacc):.4f} delta is within noise")
        print(f"         (CI includes 0, p={p:.4f}).")
        print(f"         Do NOT report fusion as improving over visual-only.")
        print(f"         It reduces to the visual decision on this data.")
    print("-" * 66)

    rule = rep.get("rule_fusion")
    if rule:
        racc = rule["accuracy"]
        print(f"\nFor reference, rule fusion accuracy = {racc:.4f} "
              f"({racc - vacc:+.4f} vs visual-only) -- WORSE than visual-only.")

    out = {
        "n_clips": n,
        "visual_accuracy": vacc,
        "fusion_accuracy": facc,
        "delta": facc - vacc,
        "clips_changed": int(np.trace(l_cm)) - int(np.trace(v_cm)),
        "bootstrap_ci_95": [lo, hi],
        "mcnemar": {"visual_only_correct": v_only, "fusion_only_correct": f_only, "p_value": p},
        "statistically_significant": bool(significant),
    }
    (ROOT / "reports" / "fusion_significance.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {ROOT / 'reports' / 'fusion_significance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
