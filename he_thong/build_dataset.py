"""
Build a training-ready dataset from raw videos/images.

Inputs
------
data_raw/<source>/**            downloaded archives (videos and/or images)
Violence/**, NonViolence/**     the pre-existing RPL corpus (videos)

Outputs
-------
data/images/violence/**         extracted frames labelled violence
data/images/nonviolence/**      extracted frames labelled non-violence
data/video_lists/{violence,nonviolence}.txt   video-level labels
dataset_manifest.json           provenance + counts for reproducibility

Every sample carries provenance so we can later de-duplicate by source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMG_OUT = DATA / "images"
LIST_OUT = DATA / "video_lists"
MANIFEST = ROOT / "dataset_manifest.json"

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Label inference -------------------------------------------------------
# Folder names that unambiguously mean "violence".
VIOLENCE_TOKENS = {
    "violence", "violent", "fight", "fighting", "fights", "punch", "punching",
    "kick", "kicking", "hit", "hitting", "assault", "attack", "beating",
    "riot", "abuse", "aggressive", "aggression", "abnormal", "brawl",
    "shoplifting", "stealing", "vandalism",
}
# Folder names that unambiguously mean "non-violence".
NONVIOLENCE_TOKENS = {
    "nonviolence", "normal", "calm", "peace", "nofight", "benign",
    "walking", "running", "sleeping", "talking", "playing",
}


def _norm(part: str) -> str:
    """Normalise a folder name for token matching."""
    return (
        part.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


VIOLENCE_LOOKUP = {_norm(t): 1 for t in VIOLENCE_TOKENS}
NONVIOLENCE_LOOKUP = {_norm(t): 0 for t in NONVIOLENCE_TOKENS}


def infer_label(path: Path) -> int | None:
    """Return 1 for violence, 0 for non-violence, None if ambiguous.

    Ambiguity is deliberate: guessing is how label noise gets in. Anything we
    cannot classify confidently is skipped rather than silently mislabelled.

    Exact match wins over substring match, in both directions. Substring
    matching alone is wrong in both directions here:
      * "normal" is a substring of "abnormal"  -> would flip violence to calm
      * "violence" is a substring of "nonviolence" -> would flip calm to violence
    """
    parts = [p.lower() for p in path.parts]
    # deepest-first so a specific leaf folder beats a generic parent
    for part in reversed(parts[:-1]):
        n = _norm(part)
        # 1) exact match on the whole folder name
        if n in NONVIOLENCE_LOOKUP:
            return 0
        if n in VIOLENCE_LOOKUP:
            return 1
        # 2) fallback: substring, longest token first so "nonviolence" beats
        #    the bare "violence" it contains
        for tok in sorted(NONVIOLENCE_TOKENS, key=len, reverse=True):
            if tok in n:
                return 0
        for tok in sorted(VIOLENCE_TOKENS, key=len, reverse=True):
            if tok in n:
                return 1
    return None


# Helpers ---------------------------------------------------------------


def video_to_frames(
    src: Path,
    out_dir: Path,
    prefix: str,
    frames_per_video: int,
    min_side: int,
    max_side: int,
) -> int:
    """Uniformly sample `frames_per_video` frames across the clip.

    Uniform (not leading-frames) sampling matters: violence events are often
    mid-clip, and sampling only the head biases the class toward "nothing
    happening yet".
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    written = 0
    try:
        if total > 0:
            idxs = {
                int(i * (total - 1) / max(frames_per_video - 1, 1))
                for i in range(frames_per_video)
            }
        else:
            idxs = None  # unknown length -> decode sequentially

        requested = frames_per_video if idxs is not None else 1 << 30
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idxs is None or i in idxs:
                if frame is None or frame.size == 0:
                    i += 1
                    continue
                h, w = frame.shape[:2]
                if min(h, w) < min_side:
                    i += 1
                    continue
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    frame = cv2.resize(
                        frame,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                dst = out_dir / f"{prefix}_{i:06d}.jpg"
                if cv2.imwrite(str(dst), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                    written += 1
                if written >= requested:
                    break
            i += 1
    finally:
        cap.release()
    return written


def hash_file(p: Path, tail_bytes: int = 0) -> str:
    """Content hash for dedup.

    For large files we hash the head and tail rather than the whole body:
    two byte-identical videos will always match on those, and it keeps the
    cost of hashing a 16 GB corpus to seconds instead of minutes.
    """
    h = hashlib.blake2b(digest_size=16)
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as fh:
        h.update(fh.read(1 << 20))
        if tail_bytes:
            fh.seek(max(0, size - tail_bytes))
            h.update(fh.read(tail_bytes))
    return h.hexdigest()


def collect_files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--max-side", type=int, default=360)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--max-videos-per-source", type=int, default=100000)
    ap.add_argument("--hash-tail-bytes", type=int, default=1 << 20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="raw roots to include; defaults to data_raw/* plus Violence/ NonViolence/",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    DATA.mkdir(exist_ok=True)
    LIST_OUT.mkdir(parents=True, exist_ok=True)
    for cls in ("violence", "nonviolence"):
        (IMG_OUT / cls).mkdir(parents=True, exist_ok=True)

    roots: list[Path] = []
    if args.sources:
        roots = [Path(s) for s in args.sources]
    else:
        raw = ROOT / "data_raw"
        if raw.exists():
            roots += sorted([p for p in raw.iterdir() if p.is_dir()])
        for legacy in ("Violence", "NonViolence"):
            if (ROOT / legacy).exists():
                roots.append(ROOT / legacy)

    files: list[tuple[Path, int]] = []
    seen_paths: set[str] = set()
    skipped_unlabelled = 0
    dupes_by_hash = 0
    seen_hashes: set[str] = set()
    for root in roots:
        fs = collect_files(root)
        # cap per source so one huge repo cannot starve the others
        if len(fs) > args.max_videos_per_source:
            random.shuffle(fs)
            fs = fs[: args.max_videos_per_source]
            random.shuffle(fs)
        for f in fs:
            key = str(f.resolve()).lower()
            if key in seen_paths:
                continue
            lab = infer_label(f)
            if lab is None:
                skipped_unlabelled += 1
                continue
            # Content hash: the same clip can arrive twice under different
            # paths (RPL ships as loose files AND inside the 34data zips).
            # Counting it twice leaks train frames into val.
            if f.stat().st_size <= args.hash_tail_bytes * 2:
                fh = hash_file(f)
            else:
                fh = hash_file(f, args.hash_tail_bytes)
            if fh in seen_hashes:
                dupes_by_hash += 1
                continue
            seen_hashes.add(fh)
            seen_paths.add(key)
            files.append((f, lab))
        log(f"source {root.name}: {len(fs)} media files")

    videos = [(f, l) for f, l in files if f.suffix.lower() in VIDEO_EXT]
    images = [(f, l) for f, l in files if f.suffix.lower() in IMAGE_EXT]
    log(f"labelled: {len(videos)} videos, {len(images)} images, {skipped_unlabelled} unlabelled skipped")

    video_lists: dict[int, list[str]] = defaultdict(list)
    frame_counts = {1: 0, 0: 0}
    errors = 0

    import concurrent.futures as cf

    def work(item: tuple[Path, int]) -> tuple[int, int, str | None, str | None]:
        src, lab = item
        cls = "violence" if lab == 1 else "nonviolence"
        out_dir = IMG_OUT / cls
        prefix = f"{src.parent.name[:24]}_{src.stem[:40]}"
        try:
            if src.suffix.lower() in VIDEO_EXT:
                n = video_to_frames(
                    src, out_dir, prefix, args.frames_per_video, args.min_side, args.max_side
                )
                return lab, n, str(src), None
            # already an image -> normalise size, keep as one sample
            img = cv2.imread(str(src))
            if img is None or img.size == 0:
                return lab, 0, str(src), "unreadable image"
            h, w = img.shape[:2]
            if max(h, w) > args.max_side:
                scale = args.max_side / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            dst = out_dir / f"{prefix}.jpg"
            cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return lab, 1, str(src), None
        except Exception as exc:  # never let one bad file kill the run
            return lab, 0, str(src), f"{type(exc).__name__}: {exc}"

    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for lab, n, src, err in ex.map(work, files):
            done += 1
            if err:
                errors += 1
            frame_counts[lab] += n
            if src and src.lower().endswith(tuple(VIDEO_EXT)):
                video_lists[lab].append(src)
            if done % 200 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {done}/{len(files)} processed ({rate:.1f}/s) frames={frame_counts}")

    for lab, name in ((1, "violence.txt"), (0, "nonviolence.txt")):
        paths = sorted(video_lists[lab])
        (LIST_OUT / name).write_text("\n".join(paths), encoding="utf-8")

    manifest = {
        "created": time.time(),
        "roots": [str(r) for r in roots],
        "frames_per_video": args.frames_per_video,
        "max_side": args.max_side,
        "counts": {
            "violence_images": frame_counts[1],
            "nonviolence_images": frame_counts[0],
            "violence_videos": len(video_lists[1]),
            "nonviolence_videos": len(video_lists[0]),
            "errors": errors,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"DONE frames V={frame_counts[1]} NV={frame_counts[0]} errors={errors}")
    log(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
