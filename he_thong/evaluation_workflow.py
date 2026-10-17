"""Freeze a genuinely separate 20–30 video test; curate false alarms for training.

python evaluation_workflow.py init --folder NEW_VIDEOS --output evaluation/holdout.json
python evaluation_workflow.py lock --manifest evaluation/holdout.json
python evaluation_workflow.py run --manifest evaluation/holdout.json --output reports/holdout_run
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}
CATEGORIES = {"fight", "play", "sports", "crowd", "fall"}
OPTIONAL_CATEGORIES = {"walking", "social_contact"}
REQUIRED_STRATA = {"low_motion", "occluded"}
HASH_KEYS = ("sha256", "original_sha256", "source_sha256", "clip_sha256")
GROUP_KEYS = ("group", "original_group", "source_group", "source_id", "original_source_group")
MEDIA_PATH_KEYS = ("path", "relative_path", "original_path", "video", "clip")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, data):
    from main2 import atomic_json
    atomic_json(Path(path), data)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(value, base=ROOT):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def identity_rows(data):
    """Read records, clips, samples and nested result/source inventories alike."""
    if isinstance(data, dict):
        if (any(data.get(k) for k in HASH_KEYS + GROUP_KEYS)
                or any(isinstance(data.get(k), str) and Path(data[k]).suffix.lower() in VIDEO_EXT for k in MEDIA_PATH_KEYS)):
            yield data
        for value in data.values():
            if isinstance(value, (dict, list)):
                yield from identity_rows(value)
    elif isinstance(data, list):
        for value in data:
            yield from identity_rows(value)


def row_identity(row, base=ROOT):
    hashes = {str(row[k]).strip().lower() for k in HASH_KEYS if row.get(k)}
    groups = {str(row[k]).strip() for k in GROUP_KEYS if row.get(k)}
    # Older source inventories have paths and source groups, but no digest.
    for key in MEDIA_PATH_KEYS:
        value = row.get(key)
        if not isinstance(value, str) or Path(value).suffix.lower() not in VIDEO_EXT:
            continue
        candidates = [resolve(value, base), resolve(value, ROOT)]
        for path in dict.fromkeys(candidates):
            if path.is_file():
                if not row.get("sha256") or key == "original_path":
                    hashes.add(digest(path))
                break
    return hashes, groups


def discover_exposure_manifests(exclude=()):
    """Conservatively include existing development, review and consumed test data."""
    excluded = {Path(p).resolve() for p in exclude}
    paths = set()
    for folder in (ROOT / "evaluation", ROOT / "schoolguard_data/retraining", ROOT / "schoolguard_data/hard_negatives", ROOT / "reports"):
        if not folder.exists():
            continue
        for path in inventory_json_files(folder):
            if path.resolve() in excluded or "before_changes" in path.parts or path.name.endswith(".used.json"):
                continue
            # Per-clip inference timelines are large and duplicate results.json.
            if re.fullmatch(r"[0-9a-fA-F]{64}\.json", path.name):
                continue
            if folder == ROOT / "reports" and not ("manifest" in path.stem or "source" in path.stem
                    or "development" in path.stem or "result" in path.stem or "prediction" in path.stem):
                continue
            data = read(path)
            if any(True for _ in identity_rows(data)):
                paths.add(path.resolve())
    return sorted(paths)


def inventory_json_files(folder):
    """Skip frame/image caches: their parent manifests retain media identities."""
    skipped = {"frames", "references", "clips", "quick_clips", "source_videos", "cache", "__pycache__", ".git", "before_changes"}
    for parent, directories, files in os.walk(folder):
        directories[:] = [name for name in directories if name not in skipped]
        for name in files:
            if name.endswith(".json") and not name.endswith(".info.json"):
                yield Path(parent) / name


def provisional_review(data):
    """A generic reviewed flag cannot turn source/AI labels into staff approval."""
    if data.get("human_review_pending") is True or data.get("human_verified") is False:
        return True
    if data.get("review_type") in ("source", "ai", "automated") or data.get("annotation_origin") in ("source", "ai", "automated"):
        return True
    text = " ".join(str(data.get(k, "")) for k in ("reviewer", "annotation_status", "annotation", "review_type", "annotation_origin"))
    text = text.casefold().replace("_", " ")
    return bool(re.search(r"\b(codex|ai|provisional|preliminary|pending|unreviewed)\b|source labels?|not human|no human|no staff", text))


def coverage_report(clips):
    categories = {category: sum(r.get("category") == category for r in clips) for category in sorted(CATEGORIES | OPTIONAL_CATEGORIES)}
    strata = {}
    for name in sorted(REQUIRED_STRATA):
        rows = [r for r in clips if name in r.get("strata", [])]
        strata[name] = {"clips": len(rows), "positive": sum(r.get("label") is True for r in rows),
                        "negative": sum(r.get("label") is False for r in rows),
                        "source_groups": len({r.get("source_group") for r in rows if r.get("source_group")})}
    missing = [name for name, support in strata.items() if not support["positive"]]
    return {"categories": categories, "required_strata": sorted(REQUIRED_STRATA), "strata": strata,
            "missing_positive_strata": missing, "complete": not missing and all(categories[c] for c in CATEGORIES),
            "note": "Counts describe supplied footage only. Each required difficult stratum needs positive footage; sample counts do not establish CCTV generalization."}


def init_manifest(folder, output):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Danh sách đã tồn tại; không ghi đè tập kiểm tra.")
    if not folder.is_dir():
        raise ValueError("Thư mục video không tồn tại.")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    data = {"schema": 2, "state": "draft", "purpose": "unseen_final_test", "reviewer": "",
            "annotation_status": "pending_human_review", "human_review_pending": True,
            "known_manifests": [str(p) for p in discover_exposure_manifests(exclude=(output,))],
            "note": "Chọn 20–30 video mới, thêm đủ fight/play/sports/crowd/fall. Nhãn phải do người xem xác nhận.",
            "clips": [{"path": str(p), "sha256": digest(p), "label": None, "category": "",
                       "source_group": "", "previously_unused": False, "reviewed": False,
                       "strata": []} for p in files]}
    data["coverage"] = coverage_report(data["clips"])
    save(output, data)
    return {"manifest": str(output), "videos": len(files), "state": "draft", "ready": False}


def known_exposure(paths, base=ROOT):
    hashes, groups = set(), set()
    for value in paths:
        path = resolve(value, base)
        if not path.is_file():
            raise ValueError("Thiếu danh sách dữ liệu đã dùng: " + str(path))
        for row in identity_rows(read(path)):
            row_hashes, row_groups = row_identity(row, path.parent)
            hashes.update(row_hashes)
            groups.update(row_groups)
    return hashes, groups


def runtime_signature(cfg):
    # No tokens or notification endpoints enter evaluation artifacts.
    frozen = {key: copy.deepcopy(cfg[key]) for key in ("model", "person", "detection", "school_policy")}
    models = {}
    fields = [("model", "path"), ("model", "keras_path"), ("model", "feature_weights"), ("person", "path")]
    if cfg['model'].get('video_context_enabled'):
        fields += [("model", "video_context_head"), ("model", "video_context_backbone")]
    for section, key in fields:
        value = frozen[section].get(key)
        if value:
            path = resolve(value)
            if path.is_file():
                models[str(path)] = digest(path)
    return {"config": frozen, "models": models,
            "implementation": {name: digest(ROOT / name) for name in ("main2.py", "evaluation_workflow.py", "school_policy.py", "temporal_visual.py", "short_contact.py", "confidence_head.py")}}


def lock_manifest(path, cfg=None, allow_provisional=False):
    from main2 import load_config, now_iso
    path = Path(path).resolve()
    data = read(path)
    cfg=cfg or load_config()
    if cfg['school_policy']['mode']=='auto':
        raise ValueError('Chọn chế độ cố định để khóa phép thử; không để lịch thực thay đổi chính sách trong lúc chạy.')
    if data.get("state") != "draft":
        raise ValueError("Chỉ khóa danh sách draft; tập đã khóa không chỉnh lại.")
    clips = data.get("clips", [])
    if not 20 <= len(clips) <= 30:
        raise ValueError("Cần từ 20 đến 30 video riêng cho phép thử cuối.")
    if not isinstance(data.get("reviewer"), str) or not data["reviewer"].strip():
        raise ValueError("Cần người phụ trách xác nhận nhãn và nguồn video.")
    provisional = provisional_review(data) or any(provisional_review(r) for r in clips)
    if provisional and not allow_provisional:
        raise ValueError("Nhãn nguồn/AI hoặc chờ người duyệt không phải nhãn người đã xác nhận; chỉ chạy sơ bộ với --allow-provisional.")
    if not data.get("known_manifests"):
        raise ValueError("Cần khai báo các manifest huấn luyện/validation/test đã từng dùng.")
    seen, groups = known_exposure(data["known_manifests"], path.parent)
    unique, covered = set(), set()
    for clip in clips:
        if clip.get("reviewed") is not True or clip.get("previously_unused") is not True:
            raise ValueError("Mỗi clip cần được xem, gán nhãn và xác nhận chưa dùng để chỉnh hệ thống.")
        if type(clip.get("label")) is not bool or clip.get("category") not in CATEGORIES | OPTIONAL_CATEGORIES:
            raise ValueError("label cần true/false và category thuộc fight/play/sports/crowd/fall.")
        if not isinstance(clip.get("strata", []), list) or any(s not in REQUIRED_STRATA for s in clip.get("strata", [])):
            raise ValueError("strata cần là danh sách low_motion/occluded đã được xem xác nhận.")
        group = str(clip.get("source_group", "")).strip()
        if not group or group in groups:
            raise ValueError("Nguồn/nhóm video cần tách biệt khỏi dữ liệu đã dùng.")
        source = resolve(clip["path"], path.parent)
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXT:
            raise ValueError("Không tìm thấy video: " + str(source))
        actual = digest(source)
        if actual != clip.get("sha256") or actual in seen or actual in unique:
            raise ValueError("Video thay đổi, trùng nhau hoặc đã xuất hiện trong dữ liệu cũ: " + source.name)
        clip["path"] = str(source)
        unique.add(actual)
        covered.add(clip["category"])
    if not CATEGORIES <= covered or {r["label"] for r in clips} != {True, False}:
        raise ValueError("Tập kiểm tra cần đủ năm cảnh và cả nhãn có/không xô xát.")
    data["coverage"] = coverage_report(clips)
    if data.get("schema", 1) >= 2 and not data["coverage"]["complete"] and not allow_provisional:
        raise ValueError("Thiếu video dương tính cho tầng khó: " + ", ".join(data["coverage"]["missing_positive_strata"]))
    data.update(annotation_status="provisional_source_or_ai_review" if provisional else "human_reviewed",
                human_review_pending=provisional,
                evaluation_kind="provisional" if provisional or not data["coverage"]["complete"] else "final_holdout")
    data.update(state="locked", locked_at=now_iso(), runtime=runtime_signature(cfg or load_config()))
    data["known_manifests"] = [str(resolve(p, path.parent)) for p in data["known_manifests"]]
    data["exposure_files"] = {p: digest(p) for p in data["known_manifests"]}
    data["lock_sha256"] = fingerprint(data)
    save(path, data)
    return {"manifest": str(path), "state": "locked", "clips": len(clips), "lock_sha256": data["lock_sha256"]}


def score_predictions(rows):
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for row in rows:
        if row.get("error"):
            continue
        if type(row.get("label")) is not bool or type(row.get("alert")) is not bool:
            raise ValueError("Kết quả phải chứa nhãn và cảnh báo thực.")
        counts[("true_positive" if row["alert"] else "false_negative") if row["label"] else
               ("false_positive" if row["alert"] else "true_negative")] += 1
    tp, tn, fp, fn = (counts[k] for k in ("true_positive", "true_negative", "false_positive", "false_negative"))
    return {**counts, "correct": tp + tn, "false_alarms": fp, "missed": fn,
            "scored": tp + tn + fp + fn, "errors": sum(bool(r.get("error")) for r in rows),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "unit": "video; phát hiện khi có ít nhất một cảnh báo tự động trong video"}


def run_evaluation(manifest, output, cfg=None, emit=print, cancel=None, allow_provisional=False):
    import main2 as app
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    data, cfg = read(manifest), cfg or app.load_config()
    expected = data.pop("lock_sha256", None)
    if data.get("state") != "locked" or expected != fingerprint(data):
        raise ValueError("Danh sách chưa khóa hoặc đã sửa sau khi khóa.")
    provisional = provisional_review(data) or any(provisional_review(r) for r in data["clips"])
    if provisional and data.get("evaluation_kind") != "provisional" and not allow_provisional:
        raise ValueError("Manifest cũ có nhãn nguồn/AI; cần --allow-provisional để chạy và ghi rõ trạng thái sơ bộ.")
    if data["runtime"] != runtime_signature(cfg):
        raise ValueError("Mô hình, mã hoặc cấu hình đã đổi sau khi khóa; không dùng tập cuối để chỉnh tiếp.")
    for path, sha in data["exposure_files"].items():
        if digest(path) != sha:
            raise ValueError("Danh sách dữ liệu cũ thay đổi sau khi khóa.")
    for clip in data["clips"]:
        if digest(clip["path"]) != clip["sha256"]:
            raise ValueError("Video đã thay đổi sau khi khóa.")
    if output.exists():
        raise ValueError("Thư mục kết quả đã tồn tại; giữ nguyên bằng chứng lần thử trước.")
    output.mkdir(parents=True)
    # Mark this test as used BEFORE inference. It cannot be rerun for threshold selection.
    marker = manifest.with_suffix(manifest.suffix + ".used.json")
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"started": app.now_iso(), "lock_sha256": expected, "output": str(output)}, stream)
    report = {"state": "running", "started": app.now_iso(), "lock_sha256": expected, "results": [],
              "annotation_status": "provisional_source_or_ai_review" if provisional else data.get("annotation_status", "unknown_review_status"),
              "human_review_pending": provisional or data.get("human_review_pending", True),
              "evaluation_kind": "provisional" if provisional else data.get("evaluation_kind", "historical_unspecified"),
              "coverage": data.get("coverage", coverage_report(data["clips"])),
              "planned": len(data["clips"]), "scope": "video-level final test; không xác định người có lỗi", "runtime": data["runtime"]}
    save(output / "results.json", report)
    app.runtime_imports()
    visual, people = app.VisualEngine(cfg["model"]), app.PeopleEngine(cfg["person"])
    if cfg["detection"]["require_interaction"] and people.model is None:
        raise ValueError("Thiếu mô hình người; chưa thể đo hệ thống cảnh báo hoàn chỉnh.")
    for clip in data["clips"]:
        if cancel is not None and cancel.is_set():
            break
        row = {k: clip[k] for k in ("path", "sha256", "label", "category", "source_group")}
        row["strata"] = clip.get("strata", [])
        try:
            result = app.analyze_clip(Path(clip["path"]), cfg, engine=visual, people_engine=people, cancel=cancel)
            if result["cancelled"] or not result["timeline"]:
                raise ValueError("Phân tích bị dừng hoặc không có kết quả AI.")
            save(output / (clip["sha256"] + ".json"), result)
            row.update(alert=bool(result["automatic_alerts"]), alerts=result["automatic_alerts"])
        except Exception as exc:
            row["error"] = app.redact(exc)
        report["results"].append(row)
        save(output / "results.json", report)
        emit(f"Đã thử {len(report['results'])}/{report['planned']}: {Path(clip['path']).name}")
    report.update(state="complete" if len(report["results"]) == report["planned"] and not any(r.get("error") for r in report["results"]) else "incomplete",
                  ended=app.now_iso(), metrics=score_predictions(report["results"]))
    report["by_category"] = {c: score_predictions([r for r in report["results"] if r["category"] == c]) for c in sorted(CATEGORIES | OPTIONAL_CATEGORIES)}
    report["by_stratum"] = {s: score_predictions([r for r in report["results"] if s in r.get("strata", [])]) for s in sorted(REQUIRED_STRATA)}
    save(output / "results.json", report)
    m = report["metrics"]
    (output / "RESULTS.md").write_text(
        f"# Đánh giá tập video riêng\n\nTrạng thái: {report['state']}. Đã xử lý {len(report['results'])}/{report['planned']} video.\n\n"
        f"Đúng: {m['correct']}; báo nhầm: {m['false_alarms']}; bỏ sót: {m['missed']}; lỗi: {m['errors']}.\n\n"
        f"Nhãn: {report['annotation_status']}. Còn cần người phụ trách duyệt: {report['human_review_pending']}.\n\n"
        f"Phạm vi: {report['evaluation_kind']}. Đủ tầng khó: {report['coverage']['complete']}; thiếu: {', '.join(report['coverage']['missing_positive_strata']) or 'không'}.\n\n"
        "Đơn vị là video, không phải từng hành vi/thời điểm. Xem results.json cho từng cảnh. "
        "Nguồn mới do người phụ trách xác nhận; kiểm tra SHA256 phát hiện trùng nguyên tệp, không chứng minh mọi bản cắt/đổi mã đều khác nguồn. "
        "Nếu dùng kết quả để chỉnh tiếp, phải thu một tập kiểm tra cuối mới.\n", encoding="utf-8")
    return report


def export_false_alarms(store, output):
    """Export review candidates only. No unreviewed sample can enter training."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Thư mục xuất đã tồn tại.")
    output.mkdir(parents=True)
    samples, skipped = [], []
    for row in store.rows(limit=None):
        if row["status"] not in ("Báo nhầm", "Báo giả"):
            continue
        source = Path(row.get("clip", ""))
        if not source.is_file() or source.suffix.lower() not in VIDEO_EXT:
            skipped.append({"event_id": row["id"], "reason": "Chưa có clip hoàn chỉnh; cần lấy clip từ camera."})
            continue
        # Ensure evidence belongs to this store, never follow arbitrary database paths.
        if not source.resolve().is_relative_to(store.folder):
            skipped.append({"event_id": row["id"], "reason": "Clip nằm ngoài thư mục bằng chứng."})
            continue
        target = output / "clips" / (row["id"] + source.suffix.lower())
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        samples.append({"event_id": row["id"], "path": str(target), "sha256": digest(target),
                        "label": False, "category": "", "source_group": "camera:" + row["camera"] + ":" + row["created"][:10],
                        "reviewed": False, "reviewer": "", "approved_for_training": False,
                        "note": "Xem toàn clip; chỉ duyệt khi toàn bộ clip không có xô xát, hoặc cắt đoạn và cập nhật hash."})
    manifest = {"schema": 1, "purpose": "hard_negative_review", "samples": samples, "skipped": skipped}
    save(output / "manifest.json", manifest)
    return {"manifest": str(output / "manifest.json"), "candidates": len(samples), "skipped": len(skipped),
            "note": "Chưa đưa vào huấn luyện; cần người kiểm tra toàn clip, category và approved_for_training."}


def approved_training_samples(manifest, forbidden_hashes=(), forbidden_groups=()):
    data = read(manifest)
    if data.get("purpose") != "hard_negative_review":
        raise ValueError("Không phải danh sách mẫu báo nhầm.")
    result, seen = [], set(forbidden_hashes)
    for row in data.get("samples", []):
        if row.get("approved_for_training") is not True:
            continue
        if provisional_review(data) or provisional_review(row):
            raise ValueError("Nhãn nguồn/AI hoặc chờ người duyệt không được coi là phê duyệt huấn luyện của người phụ trách.")
        if (row.get("reviewed") is not True or row.get("label") is not False or not str(row.get("reviewer", "")).strip()
                or row.get("category") not in (CATEGORIES | OPTIONAL_CATEGORIES) - {"fight"} or not row.get("source_group")):
            raise ValueError("Mẫu duyệt thiếu người kiểm tra, loại cảnh hoặc nhãn không xô xát.")
        path = resolve(row["path"], Path(manifest).resolve().parent)
        sha = digest(path)
        if sha != row["sha256"] or sha in seen or row["source_group"] in forbidden_groups:
            raise ValueError("Mẫu trùng tập kiểm tra/nhóm dữ liệu hoặc nội dung đã thay đổi.")
        seen.add(sha)
        result.append({**row, "path": str(path)})
    if not result:
        raise ValueError("Chưa có mẫu báo nhầm nào được duyệt để huấn luyện.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "lock", "run", "export-false-alarms"))
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-provisional", action="store_true", help="Cho phép nhãn sơ bộ/tầng khó còn thiếu; kết quả luôn ghi rõ chưa đủ điều kiện phép thử cuối.")
    args = parser.parse_args(argv)
    if args.action == "init":
        if not args.folder or not args.output:
            parser.error("init cần --folder và --output")
        result = init_manifest(args.folder, args.output)
    elif args.action == "lock":
        if not args.manifest:
            parser.error("lock cần --manifest")
        result = lock_manifest(args.manifest, allow_provisional=args.allow_provisional)
    elif args.action == "run":
        if not args.manifest or not args.output:
            parser.error("run cần --manifest và --output")
        result = run_evaluation(args.manifest, args.output, allow_provisional=args.allow_provisional)
    else:
        import main2 as app
        if not args.output:
            parser.error("export-false-alarms cần --output")
        store = app.EventStore(resolve(app.load_config()["storage"]["folder"]))
        try:
            result = export_false_alarms(store, args.output)
        finally:
            store.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
