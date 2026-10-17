"""
Train a violence / non-violence image classifier on the collected corpus.

Design notes
------------
* Split is done at the VIDEO level, never the frame level. Frames from one clip
  are near-duplicates; a frame-level split leaks the val set into train and
  inflates accuracy to ~99% on data that never generalises. This is the single
  most common way violence classifiers "work" in demos and fail in the field.
* We therefore group frames by their source clip (`<source>_<stem>_<idx>.jpg`)
  and use GroupShuffleSplit on those groups.
* A pretrained ResNet-18 backbone is fine-tuned; the head is replaced for
  binary output. Pretraining matters here because the corpus is small and
  CCTV footage is far from ImageNet's distribution.
* Class weights counteract any residual imbalance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "data" / "images"
MODEL_DIR = ROOT / "models"
METRICS_DIR = ROOT / "reports"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The retrain writes to its own file and only replaces the live checkpoint if it
# actually wins. Overwriting in place would let a worse run destroy a good model.
BASELINE_F1 = 0.9300  # 93.0% F1 held by models/violence_resnet18.pt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples() -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Return (paths, labels, groups). Group = source clip id."""
    paths: list[Path] = []
    labels: list[int] = []
    groups: list[str] = []

    for cls_name, lab in (("violence", 1), ("nonviolence", 0)):
        d = IMG_DIR / cls_name
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(lab)
                # build_dataset writes "<sourcefolder>_<stem>_<index>.jpg";
                # strip the trailing _NNNNNN index to recover the clip id.
                stem = p.stem
                gid = stem.rsplit("_", 1)[0] if stem[-6:].isdigit() else stem
                groups.append(gid)
    return paths, np.asarray(labels, dtype=np.int64), np.asarray(groups)


class FrameDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, tf) -> None:
        self.paths = paths
        self.labels = labels
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            # corrupt file -> return a black image rather than kill the batch
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        return self.tf(img), int(self.labels[i])


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, val_tf


def build_model() -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 2))
    return m


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs: list[np.ndarray] = []
    targets: list[int] = []
    losses = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        losses += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        targets.append(y.cpu().numpy())
    p = np.concatenate(probs) if probs else np.array([])
    t = np.concatenate(targets) if targets else np.array([])
    return float(losses / max(n, 1)), p, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--candidate-name", default="violence_resnet18_candidate.pt",
                    help="staging filename; promoted to violence_resnet18.pt only if it beats baseline")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch intra-op threads; 0 = auto (cpu_count - dataloader workers)")
    ap.add_argument("--smoke", action="store_true",
                    help="cap data to ~512 train / 256 val and run 1 epoch, to verify "
                         "the save/promote path before committing to a long run")
    args = ap.parse_args()

    # Leave headroom for dataloader workers. Left at torch's default (14 here)
    # the intra-op and worker pools fight over the same 20 cores.
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    else:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - args.workers))
    print(f"torch threads={torch.get_num_threads()} "
          f"(cpu_count={os.cpu_count()}, workers={args.workers})")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}")

    paths, labels, groups = collect_samples()
    print(f"samples={len(paths)} violence={int((labels==1).sum())} nonviolence={int((labels==0).sum())}")
    if len(paths) < 50:
        print("ERROR: not enough samples to train (need >=50). Run build_dataset.py first.")
        return 1

    # ---- group-wise split (prevents frame leakage) ----
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(np.zeros(len(paths)), labels, groups))
    tr_paths = [paths[i] for i in tr_idx]
    va_paths = [paths[i] for i in va_idx]
    tr_lab = labels[tr_idx]
    va_lab = labels[va_idx]
    tr_grp = groups[tr_idx]
    va_grp = groups[va_idx]
    overlap = set(tr_grp) & set(va_grp)
    print(
        f"train={len(tr_paths)} (V={int((tr_lab==1).sum())}, NV={int((tr_lab==0).sum())}) "
        f"val={len(va_paths)} (V={int((va_lab==1).sum())}, NV={int((va_lab==0).sum())}) "
        f"group_overlap={len(overlap)}"
    )
    assert not overlap, "GROUP LEAKAGE: same clip in train and val"

    if args.smoke:
        # Cap AFTER the group split so leakage assertions still hold on the
        # capped subsets. Stride keeps both classes represented.
        tr_stride = max(1, len(tr_paths) // 512)
        va_stride = max(1, len(va_paths) // 256)
        tr_paths, tr_lab = tr_paths[::tr_stride][:512], tr_lab[::tr_stride][:512]
        va_paths, va_lab = va_paths[::va_stride][:256], va_lab[::va_stride][:256]
        args.epochs = 1
        args.candidate_name = "_smoke_candidate.pt"
        print(f"SMOKE MODE: train={len(tr_paths)} val={len(va_paths)} epochs=1")

    train_tf, val_tf = build_transforms()
    tr_ds = FrameDataset(tr_paths, tr_lab, train_tf)
    va_ds = FrameDataset(va_paths, va_lab, val_tf)

    # class weights from the training split
    n1 = float((tr_lab == 1).sum())
    n0 = float((tr_lab == 0).sum())
    w = torch.tensor([len(tr_lab) / (2 * max(n0, 1)), len(tr_lab) / (2 * max(n1, 1))],
                     dtype=torch.float32, device=device)
    print(f"class weights={w.tolist()}")

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, persistent_workers=args.workers > 0)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=w)
    candidate_name = args.candidate_name

    best = {"f1": -1.0}
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        vloss, p, t = evaluate(model, va_ld, device)
        pred = (p >= 0.5).astype(int)
        f1 = f1_score(t, pred, zero_division=0)
        rec = recall_score(t, pred, zero_division=0)
        prec = precision_score(t, pred, zero_division=0)
        try:
            auc = roc_auc_score(t, p)
        except ValueError:
            auc = float("nan")
        print(
            f"epoch {ep+1}/{args.epochs} train_loss={tot/max(nb,1):.4f} "
            f"val_loss={vloss:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} auc={auc:.4f} "
            f"({time.time()-t0:.0f}s)"
        )
        hist.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_loss": vloss,
                     "f1": f1, "precision": prec, "recall": rec, "auc": auc})
        # Save to a candidate file, never straight to the live checkpoint: `best`
        # starts at -1.0, so without this every run overwrote the live model on
        # epoch 1 and a retrain that ended worse than the baseline destroyed it.
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": ep + 1, "precision": prec, "recall": rec, "auc": auc}
            MODEL_DIR.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / candidate_name)

    print(f"BEST epoch={best['epoch']} f1={best['f1']:.4f} auc={best.get('auc', float('nan')):.4f}")

    # Promote only on a genuine, statistically-supported improvement. Comparing
    # against a hardcoded F1 constant is wrong: it once promoted a candidate
    # whose real F1 on this split (0.9347) was WORSE than the live model's
    # (0.9352). The live model is scored on the SAME split here, so the
    # comparison is apples-to-apples.
    candidate = MODEL_DIR / candidate_name
    live = MODEL_DIR / "violence_resnet18.pt"
    promoted = False
    promote_reason = ""

    cand_pred = (p >= 0.5).astype(int) if p is not None and len(p) else np.array([])
    live_f1 = None
    live_p_value = 1.0

    if live.exists():
        prev = build_model().to(device)
        prev.load_state_dict(torch.load(live, map_location=device))
        _vl, lp, lt = evaluate(prev, va_ld, device)
        if lp is not None and len(lp) == len(t):
            live_pred = (lp >= 0.5).astype(int)
            live_f1 = float(f1_score(lt, live_pred, zero_division=0))
            # Exact McNemar: only claim an improvement if the discordant pairs
            # are significantly lopsided, not merely numerous.
            try:
                from fusion_significance import mcnemar as _mcnemar

                _only, cand_only, live_p_value = _mcnemar(lt, cand_pred, live_pred)
            except Exception:
                live_p_value = 1.0

    if not live.exists():
        promoted = True
        promote_reason = "no live checkpoint existed"
    elif live_f1 is None:
        promoted = True
        promote_reason = "live model produced no comparable scores"
    elif best["f1"] > live_f1 and live_p_value < 0.05:
        promoted = True
        promote_reason = (f"f1 {best['f1']:.4f} > live {live_f1:.4f} "
                          f"and McNemar p={live_p_value:.4f} < 0.05")
    else:
        promote_reason = (f"f1 {best['f1']:.4f} vs live {live_f1:.4f}, "
                          f"McNemar p={live_p_value:.4f} - not a proven improvement")

    if promoted:
        os.replace(candidate, live)
        candidate = live  # the file no longer exists at its old path
        print(f"PROMOTED: {candidate_name} -> violence_resnet18.pt ({promote_reason})")
    else:
        os.replace(candidate, MODEL_DIR / "violence_resnet18_rejected.pt")
        candidate = MODEL_DIR / "violence_resnet18_rejected.pt"
        print(f"NOT PROMOTED ({promote_reason}). Live checkpoint left untouched.")

    # ---- final report on the held-out split ----
    # Reload `candidate`, not the live path: on a rejected run the live file is
    # still the baseline, and reloading it would report baseline metrics as ours.
    model.load_state_dict(torch.load(candidate, map_location=device))
    vloss, p, t = evaluate(model, va_ld, device)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(t, pred, labels=[0, 1])
    rep = classification_report(t, pred, target_names=["nonviolence", "violence"], zero_division=0)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)
    print(rep)

    METRICS_DIR.mkdir(exist_ok=True)
    metrics = {
        "seed": args.seed,
        "promoted": promoted,
        "baseline_f1": BASELINE_F1,
        "candidate": candidate.name,
        "n_train": len(tr_paths),
        "n_val": len(va_paths),
        "n_groups": len(set(groups)),
        "group_overlap": len(overlap),
        "best": best,
        "history": hist,
        "confusion_matrix": cm.tolist(),
        "report": rep,
        "final": {
            "f1": float(f1_score(t, pred, zero_division=0)),
            "precision": float(precision_score(t, pred, zero_division=0)),
            "recall": float(recall_score(t, pred, zero_division=0)),
        },
    }
    (METRICS_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"metrics -> {METRICS_DIR / 'training_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
