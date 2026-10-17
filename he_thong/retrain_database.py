"""Reproducible video-level retraining of SchoolGuard, with a separate test set.

The source database is read-only. Frames, manifests, candidates and progress are
written to a versioned run directory. No external notifications are sent.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def stamp():
    from datetime import datetime, timezone, timedelta
    return datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def say(message):
    print(f"[{stamp()}] {message}", flush=True)


def save_torch(value, path):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(value, temporary)
    os.replace(temporary, path)


def digest_file(path):
    before = Path(path).stat()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError("source media changed while being read")
    return digest.hexdigest()


class UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, left, right):
        self.parents[self.find(right)] = self.find(left)


class HashTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if not distance:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = (value, [index], {})
                return
            node = node[2][distance]

    def search(self, value, radius):
        pending = [self.root] if self.root else []
        result = []
        while pending:
            node = pending.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                result.extend(node[1])
            pending.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)
        return result


def original_group(row):
    stem = Path(row["relative_path"]).stem
    if row["source"] == "RWF-2000":
        stem = re.sub(r"^rwf2000_(?:train|val)_(?:v|nv)_", "", stem)
        stem = re.sub(r"_\d+$", "", stem)
    elif row["source"] == "UCF-Crime":
        stem = re.sub(r"^ucf_(?:abuse|crime_(?:train|val|test))_(?:v|nv)_", "", stem)
    elif row["source"] == "Weapon-Violence-Dataset-2.0":
        # Lighting/weather variants of the same action are one source group.
        stem = re.sub(r"-(?:Afternoon|Dusk|Midday|MidNight|Morning|SunSet).*$", "", stem)
    return row["source"] + ":" + stem


def list_media(database):
    metadata = {}
    csv_path = database / "metadata.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            for item in csv.DictReader(stream):
                metadata[item["relative_path"].replace("\\", "/")] = item
    records, exclusions = [], []
    for label, folder in ((0, "nonviolence"), (1, "violence")):
        for path in sorted((database / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXT | IMAGE_EXT:
                continue
            relative = path.relative_to(database).as_posix()
            item = metadata.get(relative, {})
            source = item.get("source_dataset", "User-added")
            reason = ""
            if item and (int(item["class_id"]) != label or item["label"] != folder):
                reason = "metadata and parent-folder labels disagree"
            if source == "UCF-Crime" and label == 1:
                category = re.sub(r"^ucf_(?:abuse|crime_(?:train|val|test))_v_", "", path.stem)
                if not category.startswith(("Abuse", "Assault", "Fighting")):
                    reason = "crime/anomaly label does not establish interpersonal violence"
            row = {"relative_path": relative, "label": label, "source": source,
                   "bytes": path.stat().st_size, "kind": "image" if path.suffix.lower() in IMAGE_EXT else "video"}
            if reason:
                exclusions.append(dict(row, reason=reason))
            else:
                row["original_group"] = original_group(row)
                records.append(row)
    return records, exclusions


def frame_hash(frame):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.dct(gray)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    return sum(int(bit) << index for index, bit in enumerate(bits))


def decode_record(row, database, run, frame_count):
    import cv2
    import numpy as np
    path = database / row["relative_path"]
    folder = run / "frames" / row["sha256"][:24]
    cached = folder / "decoded.json"
    if cached.is_file():
        data = json.loads(cached.read_text(encoding="utf-8"))
        if data.get("sha256") == row["sha256"] and data.get("requested_frames") == frame_count:
            if all((run / item).is_file() for item in data.get("frames", [])):
                return dict(row, **{k: v for k, v in data.items() if k != "sha256"})
    folder.mkdir(parents=True, exist_ok=True)
    frames, hashes, times = [], [], []
    cap = None
    try:
        if row["kind"] == "image":
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            images = [(0., image)]
            duration, total, fps = 0., 1, 0.
        else:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise ValueError("cannot open video")
            total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
            if total < 2 or not 0 < fps <= 240:
                raise ValueError("invalid frame count or FPS")
            duration = total / fps
            images = []
            indices = sorted(set(int((i+.5)*total/frame_count) for i in range(frame_count)))
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(index, total-1))
                ok, image = cap.read()
                if ok and image is not None:
                    images.append((index/fps, image))
        for second, image in images:
            if image is None or min(image.shape[:2]) < 48:
                continue
            hashes.append(f"{frame_hash(image):016x}")
            scale = min(1., 384 / max(image.shape[:2]))
            if scale < 1:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            target = folder / f"f{len(frames):02d}.jpg"
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                continue
            encoded.tofile(target)
            frames.append(target.relative_to(run).as_posix())
            times.append(round(second, 4))
        if len(frames) < (1 if row["kind"] == "image" else min(4, frame_count)):
            raise ValueError("too few valid decoded frames")
        data = {"sha256": row["sha256"], "requested_frames": frame_count, "frames": frames,
                "frame_seconds": times, "phashes": hashes, "duration_seconds": round(duration, 4),
                "total_frames": total, "fps": fps}
        write_json(cached, data)
        return dict(row, **{k: v for k, v in data.items() if k != "sha256"})
    except Exception as error:
        return dict(row, error=type(error).__name__ + ": " + str(error))
    finally:
        if cap is not None:
            cap.release()


def split_records(records, seed):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    union = UnionFind(len(records))
    originals, tree, similar = {}, HashTree(), []
    for index, row in enumerate(records):
        key = row["original_group"]
        if key in originals:
            union.join(index, originals[key])
        originals[key] = index
        hashes = [int(value, 16) for value in row["phashes"]]
        probes = [hashes[0], hashes[len(hashes)//2], hashes[-1]]
        for candidate in tree.search(probes[1], 4):
            other = records[candidate]
            if row["kind"] != other["kind"]:
                continue
            if row["kind"] == "video" and abs(row["duration_seconds"]-other["duration_seconds"]) > max(.25, .05*row["duration_seconds"]):
                continue
            other_hashes = [int(value, 16) for value in other["phashes"]]
            other_probes = [other_hashes[0], other_hashes[len(other_hashes)//2], other_hashes[-1]]
            distances = [(a ^ b).bit_count() for a, b in zip(probes, other_probes)]
            if max(distances) <= 6 and sum(distances) <= 12:
                union.join(index, candidate)
                similar.append({"a": row["relative_path"], "b": other["relative_path"], "hamming": distances,
                                "different_labels": row["label"] != other["label"]})
        tree.add(probes[1], index)
    groups = np.asarray([f"group-{union.find(i):05d}" for i in range(len(records))])
    counts = Counter(row["source"] + ":" + str(row["label"]) for row in records)
    strata = np.asarray([row["source"] + ":" + str(row["label"]) if counts[row["source"]+":"+str(row["label"])] >= 10 else "rare:"+str(row["label"]) for row in records])
    rare = Counter(strata)
    strata = np.asarray([value if rare[value] >= 10 else "RLVS:"+str(records[i]["label"]) for i,value in enumerate(strata)])
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    assigned = {}
    for fold, (_, indices) in enumerate(folds.split(np.zeros(len(records)), strata, groups)):
        for index in indices:
            assigned[groups[index]] = "test" if fold == 0 else "validation" if fold == 1 else "train"
    # Preserve official held-out footage, and move related train clips with it.
    official_test, official_val = set(), set()
    for i, row in enumerate(records):
        name = Path(row["relative_path"]).name
        if name.startswith("ucf_crime_test_") or (row["source"] == "SCVD-SmartCity" and "_test_" in name):
            official_test.add(groups[i])
        elif name.startswith("ucf_crime_val_"):
            official_val.add(groups[i])
        elif name.startswith("rwf2000_val_"):
            bucket = int(hashlib.sha256((str(seed)+groups[i]).encode()).hexdigest()[:8], 16) % 2
            (official_test if bucket else official_val).add(groups[i])
    for group in official_val:
        assigned[group] = "validation"
    for group in official_test:
        assigned[group] = "test"
    for i, row in enumerate(records):
        row["group"] = str(groups[i])
        row["split"] = assigned[groups[i]]
    split_groups = {name: {row["group"] for row in records if row["split"] == name} for name in ("train", "validation", "test")}
    assert not (split_groups["train"] & split_groups["validation"] or split_groups["train"] & split_groups["test"] or split_groups["validation"] & split_groups["test"])
    for name in split_groups:
        assert {row["label"] for row in records if row["split"] == name} == {0, 1}
    return similar, {name: len(value) for name, value in split_groups.items()}


def final_holdout_manifests(args):
    from evaluation_workflow import inventory_json_files, read
    paths = set(inventory_json_files(ROOT / "evaluation"))
    explicit = getattr(args, "holdout", None)
    if explicit:
        paths.add(Path(explicit).resolve())
    for path in sorted(paths):
        if path.name.endswith(".used.json"):
            # A used marker protects its parent even when its purpose/state is old.
            parent = path.with_name(path.name[:-len(".used.json")])
            if not parent.is_file():
                raise ValueError("Thiếu manifest của tập cuối đã dùng: " + str(parent))
            continue
        data = read(path)
        if not isinstance(data, dict):
            if explicit and path.resolve() == Path(explicit).resolve() and isinstance(data, list):
                data = {"clips": data}
            else:
                continue
        if (path == explicit or path.resolve() == (Path(explicit).resolve() if explicit else None)
                or data.get("purpose") in ("unseen_final_test", "final_holdout")
                or path.with_suffix(path.suffix + ".used.json").exists()):
            yield path, data


def holdout_identity(args):
    from evaluation_workflow import identity_rows, row_identity
    hashes, groups = set(), set()
    for path, data in final_holdout_manifests(args):
        for row in identity_rows(data.get("clips", data.get("records", data.get("samples", [])))):
            row_hashes, row_groups = row_identity(row, path.parent)
            hashes.update(row_hashes)
            groups.update(row_groups)
    return hashes, groups


def sample_protected_phashes(path, frame_count=8):
    """Read old holdout footage without changing its historical manifest/cache."""
    import cv2
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    hashes = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or total < 1:
            raise ValueError("Không đọc được video tập cuối để bảo vệ bản cắt/đổi mã: " + str(path))
        for index in sorted({min(total - 1, int((i + .5) * total / frame_count)) for i in range(frame_count)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                hashes.append(frame_hash(frame))
        if len(hashes) < min(4, total):
            raise ValueError("Thiếu khung đọc được trong video tập cuối: " + str(path))
        return hashes
    finally:
        capture.release()


def holdout_phashes(args):
    from evaluation_workflow import identity_rows, resolve
    hashes, sampled = set(), set()
    for manifest, data in final_holdout_manifests(args):
        for row in identity_rows(data.get("clips", data.get("records", data.get("samples", [])))):
            values = row.get("phashes", [])
            if values:
                hashes.update(int(value, 16) for value in values)
                continue
            value = row.get("path", row.get("relative_path"))
            if not value:
                raise ValueError("Tập cuối thiếu cả phashes lẫn đường dẫn video: " + str(manifest))
            source = resolve(value, manifest.parent)
            if not source.is_file():
                raise ValueError("Thiếu video tập cuối để kiểm tra gần trùng: " + str(source))
            if source not in sampled:
                hashes.update(sample_protected_phashes(source))
                sampled.add(source)
    return hashes


def protected_hash_tree(values):
    tree = HashTree()
    for index, value in enumerate(set(values)):
        tree.add(value, index)
    return tree


def near_protected(row, tree):
    return any(tree.search(int(value, 16), 4) for value in row.get("phashes", []))


def verify_final_holdout_separation(args, records):
    """Recheck an old prepared run before training; never silently edit its data."""
    from evaluation_workflow import GROUP_KEYS, HASH_KEYS
    hashes, groups = holdout_identity(args)
    tree = protected_hash_tree(holdout_phashes(args))
    for row in records:
        if ({str(row[k]).lower() for k in HASH_KEYS if row.get(k)} & hashes
                or {str(row[k]) for k in GROUP_KEYS if row.get(k)} & groups):
            raise ValueError("Manifest huấn luyện chứa dữ liệu/nguồn tập cuối đã khóa hoặc đã dùng; tạo run mới bằng prepare.")
        if tree.root is not None and not row.get("phashes"):
            raise ValueError("Manifest huấn luyện thiếu phashes để loại bản gần trùng tập cuối; tạo run mới bằng prepare.")
        if near_protected(row, tree):
            raise ValueError("Manifest huấn luyện chứa khung gần giống tập cuối đã khóa hoặc đã dùng; tạo run mới bằng prepare.")


def add_reviewed_hard_negatives(args, records):
    """Append only human-approved, train-only negatives; protect all held-out data."""
    manifest = getattr(args, "hard_negatives", None)
    if manifest is None:
        return records, {"added": 0}
    from evaluation_workflow import approved_training_samples
    held_out = [r for r in records if r["split"] in ("test", "validation")]
    forbidden_hashes = {r["sha256"] for r in records}
    forbidden_groups = {r[k] for r in held_out for k in ("group", "original_group") if r.get(k)}
    holdout_hashes, holdout_groups = holdout_identity(args)
    forbidden_hashes.update(holdout_hashes)
    forbidden_groups.update(holdout_groups)
    approved = approved_training_samples(manifest, forbidden_hashes, forbidden_groups)
    protected_phashes = {int(h, 16) for r in held_out for h in r.get("phashes", [])}
    protected_phashes.update(holdout_phashes(args))
    protected_tree = protected_hash_tree(protected_phashes)
    added = []
    for sample in approved:
        source = Path(sample["path"])
        row = {"relative_path": str(source), "label": 0, "source": "Reviewed-Hard-Negatives",
               "original_group": sample["source_group"], "group": "reviewed:" + sample["source_group"],
               "split": "train", "bytes": source.stat().st_size, "kind": "video", "sha256": sample["sha256"],
               "reviewer": sample["reviewer"], "category": sample["category"]}
        decoded = decode_record(row, args.database.resolve(), args.run.resolve(), args.frames)
        if "error" in decoded:
            raise ValueError("Không đọc được clip báo nhầm đã duyệt: " + decoded["error"])
        if near_protected(decoded, protected_tree):
            raise ValueError("Clip báo nhầm có khung gần giống validation/test/tập cuối; cần kiểm tra nguồn trước.")
        added.append(decoded)
    return records + added, {"added": len(added), "manifest": str(Path(manifest).resolve()),
                             "manifest_sha256": digest_file(manifest), "split": "train"}


def prepare(args):
    import cv2
    cv2.setNumThreads(1)
    database, run = args.database.resolve(), args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    records, exclusions = list_media(database)
    say(f"Inventory: {len(records)} eligible media; {len(exclusions)} uncertain labels excluded")
    started = time.perf_counter()
    hashes, duplicates, conflicts = defaultdict(list), [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(digest_file, database / row["relative_path"]): row for row in records}
        for completed, future in enumerate(as_completed(future_map), 1):
            row = future_map[future]
            try:
                row["sha256"] = future.result()
                hashes[row["sha256"]].append(row)
            except OSError as error:
                exclusions.append(dict(row, reason="cannot read media: "+type(error).__name__))
            if completed % 1000 == 0:
                say(f"SHA256 {completed}/{len(records)}")
    unique = []
    holdout_hashes, holdout_groups = holdout_identity(args)
    for key, rows in sorted(hashes.items()):
        if key in holdout_hashes or any(r.get("original_group") in holdout_groups for r in rows):
            exclusions.extend(dict(row, reason="reserved unseen final evaluation footage") for row in rows)
            continue
        if len({row["label"] for row in rows}) > 1:
            conflicts.extend(dict(row, reason="byte-identical media has conflicting class labels") for row in rows)
            continue
        # Prefer official held-out aliases when duplicated media has multiple names.
        rows.sort(key=lambda row: (0 if "_test_" in row["relative_path"] else 1 if "_val_" in row["relative_path"] else 2, row["relative_path"]))
        unique.append(rows[0])
        duplicates.extend(dict(row, duplicate_of=rows[0]["relative_path"]) for row in rows[1:])
    say(f"Decode {len(unique)} unique media; duplicates={len(duplicates)}, conflicting labels={len(conflicts)}")
    decoded, bad = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(decode_record, row, database, run, args.frames) for row in unique]
        for completed, future in enumerate(as_completed(tasks), 1):
            row = future.result()
            (bad if "error" in row else decoded).append(row)
            if completed % 100 == 0:
                say(f"Decode {completed}/{len(unique)}; errors={len(bad)}")
                write_json(run / "progress.json", {"stage": "prepare", "completed": completed, "total": len(unique), "errors": len(bad), "updated": stamp()})
    decoded.sort(key=lambda row: row["relative_path"])
    protected_tree = protected_hash_tree(holdout_phashes(args))
    separated = []
    for row in decoded:
        if near_protected(row, protected_tree):
            exclusions.append(dict(row, reason="sampled-frame near duplicate of reserved/consumed final evaluation footage"))
        else:
            separated.append(row)
    decoded = separated
    similar, group_counts = split_records(decoded, args.seed)
    decoded, hard_negative_report = add_reviewed_hard_negatives(args, decoded)
    summary = {"created": stamp(), "database": str(database), "seed": args.seed,
               "eligible_before_dedup": len(records), "used_media": len(decoded),
               "sampled_frames": sum(len(row["frames"]) for row in decoded),
               "excluded_labels": len(exclusions), "exact_duplicates": len(duplicates),
               "conflicting_duplicates": len(conflicts), "decode_errors": len(bad),
               "similar_pairs_grouped": len(similar), "groups_by_split": group_counts,
               "group_overlap": 0, "hard_negatives": hard_negative_report,
               "seconds": round(time.perf_counter()-started, 2),
               "counts_by_split": {name: dict(Counter(str(row["label"]) for row in decoded if row["split"] == name)) for name in ("train", "validation", "test")},
               "counts_by_source": dict(Counter(row["source"] for row in decoded))}
    manifest = dict(summary, records=decoded)
    write_json(run / "manifest.json", manifest)
    write_json(run / "audit.json", dict(summary, exclusions=exclusions, exact_duplicates=duplicates,
                                        conflicting_duplicates=conflicts, unreadable=bad, similar_pairs=similar))
    write_json(run / "progress.json", {"stage": "prepared", "summary": summary, "updated": stamp()})
    say(f"Prepared {len(decoded)} media / {summary['sampled_frames']} frames; group overlap=0")
    return manifest


class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        from PIL import Image, ImageOps
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR,
                            color=(124, 116, 104), centering=(.5, .5))


def transform(size, augment=False):
    from torchvision import transforms
    steps = []
    if augment:
        steps += [transforms.RandomHorizontalFlip(),
                  transforms.ColorJitter(brightness=.22, contrast=.20, saturation=.15, hue=.025),
                  transforms.RandomGrayscale(p=.08)]
    return transforms.Compose(steps + [Letterbox(size), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Frames:
    def __init__(self, run, records, size):
        self.paths = [run / name for row in records for name in row["frames"]]
        self.tf = transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            return self.tf(image.convert("RGB"))


class Videos:
    def __init__(self, run, records, size, count=4, augment=True):
        self.run, self.records, self.count = run, records, count
        self.tf = transform(size, augment)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from PIL import Image
        row = self.records[index]
        available = row["frames"]
        chosen = random.sample(available, min(self.count, len(available)))
        while len(chosen) < self.count:
            chosen.append(random.choice(available))
        tensors = []
        for name in chosen:
            with Image.open(self.run / name) as image:
                tensors.append(self.tf(image.convert("RGB")))
        source_weights = {"Hockey-Fights": .7, "RWF-2000": 1.2, "VID-Dataverse": 1.2,
                          "UCF-Crime": .8, "RLVS": 1., "UniDataPro-Fights": 1.2,
                          "SCVD-SmartCity": 1.2, "Weapon-Violence-Dataset-2.0": .5}
        return torch.stack(tensors), float(row["label"]), source_weights.get(row["source"], 1.)


def build_model():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(512, 2))
    return model


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pooled_logits(logits):
    import torch
    margins = logits[..., 1] - logits[..., 0]
    # Positive videos need an event, not a positive label for every single frame.
    return torch.topk(margins, max(1, margins.shape[-1] // 2), dim=-1).values.mean(-1)


def video_loss(logits, labels, weights=None):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits
    margins = logits[..., 1] - logits[..., 0]
    main = binary_cross_entropy_with_logits(pooled_logits(logits), labels, reduction="none")
    negative = binary_cross_entropy_with_logits(margins, torch.zeros_like(margins), reduction="none").mean(-1)
    loss = main + .25 * negative * (1 - labels)
    return (loss * weights).mean() if weights is not None else loss.mean()


def metrics(labels, probabilities, threshold=.5):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predicted = probabilities >= threshold
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"samples": len(labels), "threshold": float(threshold), "accuracy": float(accuracy_score(labels, predicted)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "false_positive_rate": float(fp / (tn+fp)) if tn+fp else None,
            "false_negative_rate": float(fn / (fn+tp)) if fn+tp else None,
            "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else None,
            "average_precision": float(average_precision_score(labels, probabilities)) if len(set(labels)) > 1 else None,
            "confusion_matrix": matrix.tolist()}


def select_threshold(labels, probabilities):
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    beta2 = .7**2
    score = (1+beta2)*precision[:-1]*recall[:-1] / np.maximum(beta2*precision[:-1]+recall[:-1], 1e-12)
    feasible = recall[:-1] >= .82
    ranking = np.where(feasible, score, -1.) if feasible.any() else score
    index = int(np.argmax(ranking))
    chosen = float(thresholds[index])
    return chosen, float(score[index])


def feature_offsets(records):
    result, position = [], 0
    for row in records:
        result.append((position, position+len(row["frames"])))
        position += len(row["frames"])
    return result


def extract_features(args, model, records, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    run = args.run.resolve()
    manifest_hash = digest_file(run / "manifest.json")
    signature = {"manifest_sha256": manifest_hash, "image_size": args.image_size,
                 "backbone": "resnet18-imagenet1k-v1", "preprocessing": "letterbox-meanpad-v1"}
    info_path, target = run / "feature_cache.json", run / "imagenet_features.npy"
    previous = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    frames = Frames(run, records, args.image_size)
    reusable = previous.get("signature") == signature and target.is_file()
    mask_path = run / "feature_valid.npy"
    cache = np.lib.format.open_memmap(target, mode="r+" if reusable else "w+", dtype=np.float32, shape=(len(frames), 512))
    valid = np.load(mask_path) if reusable and mask_path.is_file() else np.zeros(len(frames), dtype=bool)
    if reusable and not mask_path.is_file():
        valid[:int(previous.get("completed", 0))] = True
    legacy_manifest = run / "manifest.pre_additions.json"
    legacy_metadata = run / "feature_cache.pre_additions.json"
    legacy_features = run / "imagenet_features.pre_additions.npy"
    if not reusable and legacy_manifest.is_file() and legacy_metadata.is_file() and legacy_features.is_file():
        old = json.loads(legacy_manifest.read_text(encoding="utf-8"))["records"]
        info = json.loads(legacy_metadata.read_text(encoding="utf-8"))
        if info["signature"]["image_size"] == args.image_size:
            source = np.load(legacy_features, mmap_mode="r")
            old_map = {row["sha256"]: bounds for row,bounds in zip(old, feature_offsets(old)) if bounds[1] <= info["completed"]}
            for row, (first,last) in zip(records,feature_offsets(records)):
                if row["sha256"] in old_map:
                    a,b = old_map[row["sha256"]]
                    if last-first == b-a:
                        cache[first:last] = source[a:b]
                        valid[first:last] = True
            say(f"Reused {int(valid.sum())} cached ImageNet feature vectors")
    missing = np.flatnonzero(~valid)
    if not len(missing):
        say("ImageNet feature cache already complete")
        return cache
    extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    loader = DataLoader(Subset(frames, missing.tolist()), batch_size=args.batch_size,
                        num_workers=2, persistent_workers=True)
    position, last = 0, time.perf_counter()
    with torch.inference_mode():
        for tensors in loader:
            features = extractor(tensors.to(device)).flatten(1).cpu().numpy()
            indices = missing[position:position+len(features)]
            cache[indices] = features
            valid[indices] = True
            position += len(features)
            if time.perf_counter()-last > 20 or position == len(missing):
                cache.flush()
                np.save(mask_path, valid)
                completed = int(valid.sum())
                write_json(info_path, {"signature": signature, "completed": completed, "total": len(frames)})
                write_json(run / "progress.json", {"stage": "feature_extraction", "completed": completed, "total": len(frames), "updated": stamp()})
                say(f"ImageNet features {completed}/{len(frames)}")
                last = time.perf_counter()
    cache.flush()
    return cache


def evaluate_head(head, features, indices, records, offsets, device):
    import numpy as np
    import torch
    head.eval()
    logits = []
    with torch.inference_mode():
        for index in indices:
            first, last = offsets[index]
            output = head(torch.from_numpy(np.array(features[first:last])).to(device))
            logits.append(float(pooled_logits(output.unsqueeze(0)).item()))
    return np.asarray(logits), np.asarray([records[i]["label"] for i in indices])


def warm_head(args, model, features, records, device):
    import numpy as np
    import torch
    run = args.run.resolve()
    target = run / "warm_head.pt"
    if target.is_file():
        model.fc.load_state_dict(torch.load(target, map_location=device, weights_only=True))
        say("Warm-up classifier restored")
        return
    offsets = feature_offsets(records)
    training = [i for i, row in enumerate(records) if row["split"] == "train"]
    validation = [i for i, row in enumerate(records) if row["split"] == "validation"]
    head = model.fc.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=.02)
    best, stale, history = -1., 0, []
    for epoch in range(1, 61):
        random.shuffle(training)
        head.train()
        losses = []
        for begin in range(0, len(training), 128):
            batch = training[begin:begin+128]
            bags = []
            for index in batch:
                first, last = offsets[index]
                chosen = list(range(first, last))
                if len(chosen) < args.frames:
                    chosen += random.choices(chosen, k=args.frames-len(chosen))
                bags.append(np.array(features[chosen]))
            tensor = torch.from_numpy(np.stack(bags)).to(device)
            labels = torch.tensor([records[i]["label"] for i in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(tensor)
            loss = video_loss(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        logits, labels = evaluate_head(head, features, validation, records, offsets, device)
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        threshold, quality = select_threshold(labels, probability)
        result = metrics(labels, probability, threshold)
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), selection_score=quality, **result))
        say(f"Head {epoch}/60 validation F1={result['f1']:.4f} precision={result['precision']:.4f} recall={result['recall']:.4f} FPR={result['false_positive_rate']:.4f}")
        if quality > best + .0005:
            best, stale = quality, 0
            save_torch(head.state_dict(), target)
        else:
            stale += 1
        write_json(run / "warmup_history.json", history)
        write_json(run / "progress.json", {"stage": "classifier_warmup", "epoch": epoch,
                   "validation": result, "updated": stamp()})
        if stale >= 7 and epoch >= 12:
            break
    head.load_state_dict(torch.load(target, map_location=device, weights_only=True))


def checkpoint_payload(model, args, metadata=None):
    return {"schema": 2, "architecture": "resnet18", "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "preprocessing": {"mode": "letterbox", "size": args.image_size, "pad_rgb": [124, 116, 104], "mean": MEAN, "std": STD},
            "metadata": metadata or {}}


def evaluate_model(args, model, rows, device):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(Frames(args.run.resolve(), rows, args.image_size), batch_size=args.batch_size,
                        num_workers=2, persistent_workers=True)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for tensors in loader:
            scores = model(tensors.to(device))
            outputs.extend((scores[:, 1]-scores[:, 0]).cpu().numpy().tolist())
    outputs = np.asarray(outputs)
    video, labels = [], []
    for row, (first, last) in zip(rows, feature_offsets(rows)):
        margin = outputs[first:last]
        video.append(float(np.sort(margin)[-max(1,len(margin)//2):].mean()))
        labels.append(row["label"])
    return np.asarray(video), np.asarray(labels), outputs


def fit_calibration(logits, labels):
    import torch
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_temp = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp, bias], lr=.3, max_iter=80, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(.35, 4.)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / temperature + bias.clamp(-3., 3.), y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temp.exp().clamp(.35, 4.).item()), float(bias.clamp(-3., 3.).item())


def train(args):
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    verify_final_holdout_separation(args, records)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    say(f"Train: {len(records)} media, CPU threads={args.threads}, device={device}")
    baseline = run / "baseline.pt"
    if not baseline.is_file():
        import shutil
        import main1
        current = main1.load_config()["model"]
        if current["backend"] not in ("auto", "resnet"):
            raise ValueError("The database trainer compares ResNet checkpoints; select a ResNet baseline first")
        shutil.copy2(main1.resolve_path(current["path"]), baseline)
    model = build_model().to(device)
    features = extract_features(args, model, records, device)
    warm_head(args, model, features, records, device)
    del features
    training = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    testing = [row for row in records if row["split"] == "test"]
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.layer4, model.fc):
        for parameter in module.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW([{"params": model.layer4.parameters(), "lr": 1e-4},
                                   {"params": model.fc.parameters(), "lr": 3e-4}], weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(Videos(run, training, args.image_size, args.train_frames),
                        batch_size=max(2, args.batch_size//args.train_frames), shuffle=True,
                        num_workers=2, persistent_workers=True)
    best_path, resume_path = run / "best_uncalibrated.pt", run / "resume.pt"
    history, start_epoch, best, stale = [], 0, -1., 0
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if resume["manifest_sha256"] != digest_file(run / "manifest.json") or resume["image_size"] != args.image_size:
            raise ValueError("Resume checkpoint belongs to a different dataset or image size")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history, start_epoch, best, stale = resume["history"], resume["epoch"], resume["best"], resume["stale"]
        say(f"Resume fine-tuning after epoch {start_epoch}")
    else:
        logits, labels, _ = evaluate_model(args, model, validation, device)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        threshold, best = select_threshold(labels, probabilities)
        result = metrics(labels, probabilities, threshold)
        save_torch(checkpoint_payload(model, args, {"epoch": 0, "validation": result}), best_path)
        say(f"Warm-up validation F1={result['f1']:.4f}, FPR={result['false_positive_rate']:.4f}")
    for epoch in range(start_epoch, args.epochs):
        model.eval()
        model.layer4.train()
        model.fc.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        total, count, since = 0., 0, time.perf_counter()
        for step, (images, labels, weights) in enumerate(loader, 1):
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.flatten(0, 1)).reshape(images.shape[0], images.shape[1], 2)
            loss = video_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
            optimizer.step()
            total += float(loss.item()) * len(labels)
            count += len(labels)
            if step % 50 == 0 or step == len(loader):
                progress = {"stage": "fine_tuning", "epoch": epoch+1, "epochs": args.epochs,
                            "batch": step, "batches": len(loader), "media_seen": count,
                            "training_loss": round(total / max(count,1), 5), "updated": stamp()}
                write_json(run / "progress.json", progress)
                say(f"Epoch {epoch+1}/{args.epochs} batch {step}/{len(loader)}, loss={total/count:.4f}")
        scheduler.step()
        logits, labels, _ = evaluate_model(args, model, validation, device)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        threshold, quality = select_threshold(labels, probabilities)
        result = metrics(labels, probabilities, threshold)
        row = dict(epoch=epoch+1, training_loss=total/count, seconds=round(time.perf_counter()-since, 1),
                   selection_score=quality, **result)
        history.append(row)
        say(f"Epoch {epoch+1}: F1={result['f1']:.4f}, precision={result['precision']:.4f}, recall={result['recall']:.4f}, FPR={result['false_positive_rate']:.4f}")
        if quality > best + .0005:
            best, stale = quality, 0
            save_torch(checkpoint_payload(model, args, {"epoch": epoch+1, "validation": result}), best_path)
        else:
            stale += 1
        save_torch({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "epoch": epoch+1, "best": best, "stale": stale,
                    "manifest_sha256": digest_file(run / "manifest.json"), "image_size": args.image_size}, resume_path)
        write_json(run / "fine_tuning_history.json", history)
        if stale >= 2 and epoch >= 2:
            say("Early stopping: validation selection score has stopped improving")
            break
    saved = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    validation_logits, validation_labels, _ = evaluate_model(args, model, validation, device)
    temperature, bias = fit_calibration(validation_logits, validation_labels)
    with torch.no_grad():
        model.fc[1].weight.div_(temperature)
        model.fc[1].bias.div_(temperature)
        model.fc[1].bias[1].add_(bias)
    val_probability = 1 / (1 + np.exp(-np.clip(validation_logits / temperature + bias, -40, 40)))
    threshold, _ = select_threshold(validation_labels, val_probability)
    metadata = {"created": stamp(), "manifest_sha256": digest_file(run / "manifest.json"),
                "source_database": str(args.database.resolve()), "classes": ["nonviolence", "violence"],
                "best_epoch": saved["metadata"]["epoch"], "calibration_baked_into_weights": {"temperature": temperature, "bias": bias},
                "validation_video_threshold": threshold, "group_overlap": 0,
                "pooling": "mean top-half frame logit margins; video labels are weak frame labels"}
    candidate = run / "violence_resnet18_database.pt"
    save_torch(checkpoint_payload(model, args, metadata), candidate)
    say("Best model selected on validation only; evaluating untouched test groups once")
    test_logits, test_labels, frame_logits = evaluate_model(args, model, testing, device)
    test_probability = 1 / (1 + np.exp(-np.clip(test_logits, -40, 40)))
    frame_probability = 1 / (1 + np.exp(-np.clip(frame_logits, -40, 40)))
    frame_labels = np.asarray([row["label"] for row in testing for _ in row["frames"]])
    validation_result = metrics(validation_labels, val_probability, threshold)
    test_result = metrics(test_labels, test_probability, threshold)
    by_source = {}
    for source in sorted({row["source"] for row in testing}):
        indices = [i for i,row in enumerate(testing) if row["source"] == source]
        by_source[source] = metrics(test_labels[indices], test_probability[indices], threshold)
    predictions = [{"relative_path": row["relative_path"], "group": row["group"], "source": row["source"],
                    "label": row["label"], "probability": float(probability)} for row, probability in zip(testing, test_probability)]
    write_json(run / "test_predictions.json", predictions)
    write_json(run / "validation_predictions.json", [{"relative_path": row["relative_path"], "group": row["group"],
               "source": row["source"], "label": row["label"], "probability": float(probability)}
               for row,probability in zip(validation,val_probability)])
    report = {"created": stamp(), "run": str(run), "candidate": str(candidate), "candidate_sha256": digest_file(candidate),
              "classification_unit": "source media: video pooled over sampled frames; an image is a one-frame bag",
              "baseline": str(baseline), "baseline_sha256": digest_file(baseline),
              "data": {key: value for key,value in manifest.items() if key != "records"},
              "training": {"initialization": "ImageNet ResNet18, independent of the old violence checkpoint",
                           "fine_tuned_layers": ["layer4", "fc"], "epochs_completed": len(history),
                           "best_epoch": saved["metadata"]["epoch"], "device": str(device),
                           "image_size": args.image_size, "history": history, "calibration": metadata["calibration_baked_into_weights"]},
              "validation": validation_result, "test": test_result, "test_by_source": by_source,
              "test_video_only": metrics(test_labels[np.asarray([row['kind']=='video' for row in testing])],
                                         test_probability[np.asarray([row['kind']=='video' for row in testing])], threshold),
              "weak_frame_test": metrics(frame_labels, frame_probability, threshold),
              "activated": False,
              "limits": ["Grouped holdout from supplied benchmarks, not a school field accuracy measurement.",
                         "Video labels do not annotate the start/end of violence; weak frame metrics are secondary.",
                         "Some related scenes may remain unidentified despite source-ID and perceptual grouping.",
                         "Historical training provenance for the baseline is incomplete; baseline holdout results may be optimistic."]}
    write_json(run / "report.json", report)
    write_json(run / "progress.json", {"stage": "trained", "validation": validation_result, "test": test_result, "updated": stamp()})
    say(f"Test source-media F1={test_result['f1']:.4f}, precision={test_result['precision']:.4f}, recall={test_result['recall']:.4f}, FPR={test_result['false_positive_rate']:.4f}")
    return report


def evaluate_checkpoint(args, path, rows):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import main1
    cfg = copy.deepcopy(main1.DEFAULT_CONFIG["model"])
    cfg.update(path=str(path), backend="resnet", threads=args.threads)
    engine = main1.VisualEngine(cfg)
    dataset = Frames(args.run.resolve(), rows, args.image_size)
    dataset.tf = engine.transform
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    outputs = []
    last = time.perf_counter()
    with torch.inference_mode():
        for images in loader:
            logits = engine.model(images.to(engine.device))
            outputs.extend((logits[:,1]-logits[:,0]).cpu().numpy().tolist())
            if time.perf_counter()-last > 30:
                say(f"Baseline evaluation {len(outputs)}/{len(dataset)} frames")
                last = time.perf_counter()
    probabilities = []
    for first,last in feature_offsets(rows):
        margins = np.sort(outputs[first:last])
        pooled = margins[-max(1,len(margins)//2):].mean()
        probabilities.append(float(1/(1+np.exp(-np.clip(pooled,-40,40)))))
    return np.asarray(probabilities)


def operational_regression(args, path, detection):
    import cv2
    import numpy as np
    import main1
    cfg = copy.deepcopy(main1.DEFAULT_CONFIG["model"])
    cfg.update(path=str(path), backend="resnet", threads=2)
    engine = main1.VisualEngine(cfg)
    cache = json.loads((ROOT / "reports/main1_precision_features.json").read_text(encoding="utf-8"))["cases"]
    results = []
    for case in cache:
        filename = case["video"]
        source = args.run.resolve() / "references" / filename
        previous_reference = ROOT / "schoolguard_data/retraining/database_20261011/references" / filename
        if not source.is_file() and previous_reference.is_file():
            source = previous_reference
        if not source.is_file():
            matches = list((ROOT / "database").glob("*/rlvs_*_" + filename))
            source = matches[0] if matches else ROOT / filename
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            results.append({"case": case["case"], "passed": False, "error": "reference media unavailable"})
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        verifier, gate, state = main1.InteractionVerifier(), main1.AlertGate(), main1.VisualState()
        alerts, timeline, latencies = [], [], []
        try:
            for sample in case["rows"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(sample["second"] * fps))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("failed to read reference frame")
                if sample["cut"]:
                    state = main1.VisualState()
                    verifier.reset()
                    gate.reset()
                began = time.perf_counter()
                raw, smooth = engine.score(frame, state)
                latencies.append(1000*(time.perf_counter()-began))
                boxes = []
                for values in sample["boxes"]:
                    values = copy.deepcopy(values)
                    if values.get("keypoints") is not None:
                        values["keypoints"] = np.asarray(values["keypoints"])
                    boxes.append(main1.PersonBox(**values))
                evidence = verifier.update(sample["second"], boxes, sample["shape"], smooth, detection=detection)
                _, fire = gate.update(sample["second"], smooth, evidence["supported"], detection, evidence["confirmed_since"])
                if fire:
                    alerts.append(sample["second"])
                timeline.append({"second": sample["second"], "raw": raw, "smooth": smooth,
                                 "supported": evidence["supported"], "reason": evidence["reason"]})
        except Exception as error:
            results.append({"case": case["case"], "passed": False, "error": type(error).__name__+": "+str(error)})
            continue
        finally:
            cap.release()
        passed = bool(alerts) == case["expected_alert"]
        results.append({"case": case["case"], "video": str(source), "expected_alert": case["expected_alert"],
                        "passed": passed, "alerts": alerts, "samples": len(timeline),
                        "average_visual_latency_ms": float(np.mean(latencies)), "timeline": timeline})
        say(f"Operational {case['case']}: {'PASS' if passed else 'FAIL'}, alerts={len(alerts)}")
    return {"created": stamp(), "checkpoint_sha256": digest_file(path), "detection": detection,
            "scope": "Development regression clips with cached pose/motion; visual scores recomputed using new weights.",
            "passed": sum(row["passed"] for row in results), "total": len(results), "results": results}


def assess(args):
    import numpy as np
    from scipy.stats import binomtest
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    validation = [row for row in manifest["records"] if row["split"] == "validation"]
    testing = [row for row in manifest["records"] if row["split"] == "test"]
    baseline, candidate = Path(report["baseline"]), Path(report["candidate"])
    candidate_validation = json.loads((run / "validation_predictions.json").read_text(encoding="utf-8"))
    candidate_test = json.loads((run / "test_predictions.json").read_text(encoding="utf-8"))
    yv = np.asarray([row["label"] for row in validation])
    yt = np.asarray([row["label"] for row in testing])
    say("Comparing saved baseline on the same validation and test source media")
    bv = evaluate_checkpoint(args, baseline, validation)
    bt = evaluate_checkpoint(args, baseline, testing)
    baseline_threshold, _ = select_threshold(yv, bv)
    baseline_val = metrics(yv, bv, baseline_threshold)
    baseline_test = metrics(yt, bt, baseline_threshold)
    cp = np.asarray([row["probability"] for row in candidate_test])
    cb = cp >= report["test"]["threshold"]
    bb = bt >= baseline_threshold
    candidate_only = int(((cb == yt) & (bb != yt)).sum())
    baseline_only = int(((cb != yt) & (bb == yt)).sum())
    pvalue = float(binomtest(candidate_only, candidate_only+baseline_only, .5).pvalue) if candidate_only+baseline_only else 1.
    write_json(run / "baseline_predictions.json", {"threshold_selected_on_validation": baseline_threshold,
               "validation": [{"relative_path": row["relative_path"], "probability": float(probability)} for row,probability in zip(validation,bv)],
               "test": [{"relative_path": row["relative_path"], "probability": float(probability)} for row,probability in zip(testing,bt)]})
    import main1
    config = main1.load_config()
    detection = copy.deepcopy(config["detection"])
    interaction = round(max(.5, min(.82, report["validation"]["threshold"])), 3)
    detection.update(require_interaction=True, interaction_threshold=interaction,
                     threshold=round(max(.75,min(.95,interaction+.15)), 3), release_threshold=min(.45, interaction-.15))
    regression = operational_regression(args, candidate, detection)
    write_json(run / "operational_regression.json", regression)
    cv, ct = report["validation"], report["test"]
    val_better = cv["f1"] >= baseline_val["f1"]-.005 and cv["false_positive_rate"] <= baseline_val["false_positive_rate"]+.01
    test_holds = ct["f1"] >= baseline_test["f1"]-.01 and ct["recall"] >= baseline_test["recall"]-.03 and ct["false_positive_rate"] <= baseline_test["false_positive_rate"]+.01
    meaningful = ct["f1"] > baseline_test["f1"]+.005 or ct["false_positive_rate"] < baseline_test["false_positive_rate"]-.02
    regression_holds = regression["passed"] == regression["total"] and regression["total"] >= 10
    eligible = bool(val_better and test_holds and meaningful and regression_holds)
    report["comparison"] = {"baseline_validation": baseline_val, "baseline_test": baseline_test,
                            "candidate_only_correct": candidate_only, "baseline_only_correct": baseline_only,
                            "mcnemar_exact_pvalue": pvalue,
                            "baseline_provenance_caveat": "The baseline may have seen some held-out clips during historical training.",
                            "validation_better": bool(val_better), "test_holds": bool(test_holds),
                            "meaningful_gain": bool(meaningful), "operational_regression_holds": bool(regression_holds),
                            "eligible_for_activation": eligible}
    report["operational_regression"] = {"passed": regression["passed"], "total": regression["total"], "report": str(run / "operational_regression.json")}
    if args.promote and eligible:
        import shutil
        date_tag = stamp()[:10].replace("-", "")
        destination = ROOT / "models" / ("violence_resnet18_database_"+date_tag+"_"+report["candidate_sha256"][:8]+".pt")
        shutil.copy2(candidate, destination)
        assert digest_file(destination) == report["candidate_sha256"]
        config_backup = run / "main1_config_before_activation.json"
        if not config_backup.exists():
            shutil.copy2(ROOT / "main1_config.json", config_backup)
        config["model"].update(path=destination.relative_to(ROOT).as_posix(), backend="resnet", input_size=args.image_size)
        config["detection"].update(detection)
        main1.validate_config(config)
        write_json(ROOT / "main1_config.json", config)
        report["activated"] = True
        report["deployment"] = {"created": stamp(), "model": str(destination), "configuration_backup": str(config_backup),
                                "detection": detection, "external_alerts_sent": False}
        say("New checkpoint activated in main1_config.json; restart main1 to use it")
    elif args.promote:
        say("Candidate retained for review; activation gates not met")
    write_json(run / "report.json", report)
    write_json(run / "progress.json", {"stage": "complete", "activated": report["activated"], "comparison": report["comparison"], "updated": stamp()})
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "database")
    tag = stamp().replace("-", "").replace(":", "").replace("T", "_")[:15]
    parser.add_argument("--run", type=Path, default=ROOT / ("schoolguard_data/retraining/database_"+tag))
    parser.add_argument("--stage", choices=("prepare", "train", "assess", "all"), default="all")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--hard-negatives", type=Path, help="Manifest mẫu báo nhầm đã được người phụ trách duyệt; chỉ thêm vào train ở stage prepare/all")
    parser.add_argument("--holdout", type=Path, help="Manifest tập kiểm tra riêng cần loại khỏi các mẫu huấn luyện bổ sung")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.hard_negatives and args.stage not in ("prepare", "all"):
        raise SystemExit("--hard-negatives chỉ dùng với --stage prepare/all trong run mới; không tự sửa manifest của run cũ.")
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("train", "all"):
        train(args)
    if args.stage == "assess" or args.promote and args.stage in ("train", "all"):
        assess(args)
