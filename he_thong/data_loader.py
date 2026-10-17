"""
data_loader.py — Video Data Loader (tối ưu + sửa bug augmentation)
====================================================================
- Augmentation được thực hiện trên RAW FRAMES trước khi extract feature
- Batch feature extraction (32 frames/batch) → nhanh
- Feature cache trên disk → lần 2 < 1 phút
"""

from __future__ import annotations
import os, cv2, random, hashlib, logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

log = logging.getLogger("data_loader")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".ts"}
LABELS     = {"nonviolence": 0, "violence": 1}
CACHE_DIR  = "feature_cache"


@dataclass
class VideoSample:
    path: str
    label: int
    label_name: str


# ─────────────────────────────────────────────────────────────
#  Quét dataset
# ─────────────────────────────────────────────────────────────
def scan_dataset(data_dir: str) -> Tuple[List[VideoSample], dict]:
    samples, stats = [], {}
    data_path = Path(data_dir)
    for label_name, label_idx in LABELS.items():
        folder = data_path / label_name
        if not folder.exists():
            log.warning(f"Không tìm thấy folder: {folder}"); continue
        count = 0
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS:
                samples.append(VideoSample(str(f), label_idx, label_name))
                count += 1
        stats[label_name] = count
        log.info(f"  {label_name:15s}: {count} videos")
    if not samples:
        raise ValueError(
            f"Không tìm thấy video!\n"
            f"  {data_dir}/nonviolence/*.mp4\n"
            f"  {data_dir}/violence/*.mp4"
        )
    random.shuffle(samples)
    log.info(f"Tổng: {len(samples)} | {stats}")
    return samples, stats


def train_val_split(samples, val_ratio=0.2):
    by_class: dict = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)
    train, val = [], []
    for lst in by_class.values():
        random.shuffle(lst)
        cut = max(1, int(len(lst) * val_ratio))
        val.extend(lst[:cut]); train.extend(lst[cut:])
    random.shuffle(train); random.shuffle(val)
    log.info(f"Train: {len(train)} | Val: {len(val)}")
    return train, val


# ─────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────
def _cache_key(video_path: str, seq_len: int, input_size: int,
               augmented: bool = False) -> str:
    mtime = str(os.path.getmtime(video_path))
    raw   = f"{video_path}|{mtime}|{seq_len}|{input_size}|{'aug' if augmented else 'orig'}"
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.npy")


# ─────────────────────────────────────────────────────────────
#  Trích xuất frames từ video
# ─────────────────────────────────────────────────────────────
def sample_video_frames(video_path: str, max_frames: int = 80,
                         input_size: int = 96) -> List[np.ndarray]:
    """Sample đều max_frames từ video, trả về list float32 [0,1]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw   = []

    if total > 0:
        n    = min(max_frames, total)
        idxs = np.linspace(0, total - 1, n, dtype=int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret: raw.append(frame)
    else:
        # Fallback: đọc tuần tự
        while len(raw) < max_frames:
            ret, f = cap.read()
            if not ret: break
            raw.append(f)
    cap.release()

    if not raw: return []

    out = []
    for frame in raw:
        h, w  = frame.shape[:2]
        m     = min(h, w)
        ax, ay = (w - m) // 2, (h - m) // 2
        crop  = frame[ay:ay+m, ax:ax+m]
        small = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(rgb)
    return out


# ─────────────────────────────────────────────────────────────
#  Augmentation trên raw frames (4D: N×H×W×3)
# ─────────────────────────────────────────────────────────────
def augment_frames(frames: np.ndarray) -> np.ndarray:
    """
    frames: shape (N, H, W, 3) float32 [0,1]
    Augment nhất quán toàn bộ sequence.
    """
    aug = frames.copy()

    # Horizontal flip
    if random.random() < 0.5:
        aug = aug[:, :, ::-1, :].copy()

    # Brightness ±20%
    if random.random() < 0.5:
        aug = np.clip(aug + random.uniform(-0.20, 0.20), 0.0, 1.0)

    # Contrast ×0.8-1.2
    if random.random() < 0.4:
        factor = random.uniform(0.8, 1.2)
        mean   = aug.mean(axis=(1, 2, 3), keepdims=True)
        aug    = np.clip((aug - mean) * factor + mean, 0.0, 1.0)

    # Temporal reverse
    if random.random() < 0.2:
        aug = aug[::-1].copy()

    return aug.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Batch feature extraction
# ─────────────────────────────────────────────────────────────
def extract_features_batch(frames_list: List[np.ndarray],
                            feature_extractor,
                            batch_size: int = 32) -> np.ndarray:
    """
    frames_list: list of (H,W,3) float32 arrays
    Trả về (N, feat_dim) float32
    """
    import tensorflow as tf
    arr  = np.array(frames_list, dtype=np.float32)  # (N, H, W, 3)
    feats = []
    for i in range(0, len(arr), batch_size):
        batch = tf.constant(arr[i:i + batch_size])
        f     = feature_extractor(batch, training=False).numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)  # (N, feat_dim)


# ─────────────────────────────────────────────────────────────
#  Get features cho 1 video (với cache)
# ─────────────────────────────────────────────────────────────
def get_video_features(video_path: str,
                        feature_extractor,
                        seq_len: int,
                        stride: int,
                        input_size: int,
                        augmented: bool = False,
                        use_cache: bool = True) -> np.ndarray:
    """
    Trả về shape (num_seqs, seq_len, feat_dim).
    augmented=True: áp dụng augmentation trên raw frames trước khi extract.
    """
    key   = _cache_key(video_path, seq_len, input_size, augmented)
    cpath = _cache_path(key)

    if use_cache and os.path.exists(cpath):
        return np.load(cpath)

    # Lấy raw frames
    max_f  = max(seq_len * 6, 80)
    frames = sample_video_frames(video_path, max_frames=max_f, input_size=input_size)
    if not frames:
        return np.empty((0, seq_len, 1280), dtype=np.float32)

    # Augment raw frames nếu cần
    frame_arr = np.array(frames, dtype=np.float32)   # (N, H, W, 3)
    if augmented:
        frame_arr = augment_frames(frame_arr)

    # Extract features theo batch
    all_feats = extract_features_batch(list(frame_arr), feature_extractor)  # (N, feat_dim)

    # Tạo sequences bằng sliding window
    n    = len(all_feats)
    seqs = []
    if n < seq_len:
        padded_idx = [i % n for i in range(seq_len)]
        seqs.append(all_feats[padded_idx])
    else:
        for start in range(0, n - seq_len + 1, stride):
            seqs.append(all_feats[start:start + seq_len])
        if not seqs:
            seqs.append(all_feats[:seq_len])

    result = np.array(seqs, dtype=np.float32)   # (num_seqs, seq_len, feat_dim)

    if use_cache:
        np.save(cpath, result)
    return result


# ─────────────────────────────────────────────────────────────
#  Build dataset X, y
# ─────────────────────────────────────────────────────────────
def build_dataset(samples: List[VideoSample],
                  feature_extractor,
                  seq_len: int  = 10,
                  stride: int   = 5,
                  input_size: int = 96,
                  training: bool  = True,
                  use_cache: bool = True,
                  desc: str = "Dataset") -> Tuple[np.ndarray, np.ndarray]:
    """
    Xây dựng (X, y).
    Training=True: thêm augmented version của mỗi video.
    X shape: (total_seqs, seq_len, feat_dim)
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(samples, desc=f"  {desc}", ncols=80, unit="video")
    except ImportError:
        iterator = samples
        log.info(f"  Đang xử lý {desc} ({len(samples)} videos)...")

    all_X, all_y = [], []
    cache_hits   = 0

    for sample in iterator:
        # ── Original sequences ──────────────────────────────
        cpath = _cache_path(_cache_key(sample.path, seq_len, input_size, False))
        was_cached = use_cache and os.path.exists(cpath)

        feats = get_video_features(
            sample.path, feature_extractor,
            seq_len=seq_len, stride=stride,
            input_size=input_size,
            augmented=False, use_cache=use_cache,
        )
        if feats.shape[0] > 0:
            all_X.append(feats)
            all_y.extend([sample.label] * len(feats))
            if was_cached: cache_hits += 1

        # ── Augmented sequences (chỉ khi training) ──────────
        if training:
            feats_aug = get_video_features(
                sample.path, feature_extractor,
                seq_len=seq_len, stride=stride,
                input_size=input_size,
                augmented=True, use_cache=use_cache,
            )
            if feats_aug.shape[0] > 0:
                all_X.append(feats_aug)
                all_y.extend([sample.label] * len(feats_aug))

    if not all_X:
        raise ValueError("Không đọc được video nào!")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.array(all_y, dtype=np.int32)

    nc = {0: int((y==0).sum()), 1: int((y==1).sum())}
    log.info(f"  {desc}: {len(X)} sequences | "
             f"NonViolence={nc[0]} Violence={nc[1]} | "
             f"cache={cache_hits}/{len(samples)}")
    return X, y
