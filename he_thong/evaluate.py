"""
Independent evaluation of a saved checkpoint.

Deliberately separate from train_model.py: it rebuilds the SAME group-wise
split (same seed) and scores the saved weights, so the numbers are
reproducible and not merely copied from the training run's own log.

This is the artifact you use to check a claim. If training printed "f1=0.97",
this recomputes it from the weights on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_model import (  # noqa: E402
    IMG_DIR,
    METRICS_DIR,
    build_model,
    collect_samples,
)


class EvalDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths = list(paths)
        self.labels = list(labels)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.tf(img), int(self.labels[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "violence_resnet18.pt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from sklearn.model_selection import GroupShuffleSplit

    paths, labels, groups = collect_samples()
    if not paths:
        print("ERROR: no dataset found at", IMG_DIR)
        return 1
    print(f"samples={len(paths)} violence={int((labels==1).sum())} nonviolence={int((labels==0).sum())}")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    _, va_idx = next(gss.split(np.zeros(len(paths)), labels, groups))
    va_paths = [paths[i] for i in va_idx]
    va_lab = labels[va_idx]

    overlap = set(groups[[i for i in range(len(paths)) if i not in set(va_idx)]]) & set(groups[va_idx])
    print(f"val={len(va_paths)} (V={int((va_lab==1).sum())}, NV={int((va_lab==0).sum())}) group_overlap={len(overlap)}")
    assert not overlap, "GROUP LEAKAGE in evaluation split"

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ld = DataLoader(EvalDataset(va_paths, va_lab, tf), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    probs, targets, correct, total = [], [], 0, 0
    with torch.no_grad():
        for x, y in ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            targets.append(y.cpu().numpy())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

    p = np.concatenate(probs)
    t = np.concatenate(targets)
    pred = (p >= 0.5).astype(int)
    acc = correct / max(total, 1)
    cm = confusion_matrix(t, pred, labels=[0, 1])
    rep = classification_report(t, pred, target_names=["nonviolence", "violence"], zero_division=0)
    try:
        auc = float(roc_auc_score(t, p))
    except ValueError:
        auc = float("nan")

    print(f"\nACCURACY = {acc:.4f}")
    print(f"AUC      = {auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(rep)

    METRICS_DIR.mkdir(exist_ok=True)
    out = METRICS_DIR / "independent_eval.json"
    out.write_text(json.dumps({
        "checkpoint": args.ckpt,
        "seed": args.seed,
        "n_val": int(total),
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "report": rep,
        "group_overlap": len(overlap),
    }, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
