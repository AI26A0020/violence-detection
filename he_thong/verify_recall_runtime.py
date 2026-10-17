"""Measure actual main2.analyze_clip outputs on consumed development manifests.

No inference occurs on import. Engines are reused, but analyze_clip resets all
per-clip state. Context must be explicitly enabled. Interval cases are decoded
to a pixel-verified FFV1 crop before detection, excluding earlier context.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "evaluation/requested_revision_20261015/holdout.json"
DEFAULT_BASELINES = (
    ROOT / "reports/requested_revision_20261015/holdout_new/results.json",
    ROOT / "reports/false_alarms_20261015/regression_results.json",
)
SCOPE = ("Consumed DEVELOPMENT regression; provisional source/AI labels, human "
         "review pending. No independent holdout or real-school accuracy claim.")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    temp.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def redact_config(value):
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if any(word in key.lower() for word in
                     ("token", "password", "secret", "webhook", "credential"))
                     else redact_config(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def manifest_records(manifest):
    records = manifest.get("records", manifest.get("clips"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest requires nonempty records or clips")
    normalized, ids = [], set()
    for index, original in enumerate(records):
        row = copy.deepcopy(original)
        row["id"] = str(row.get("id", f"latest25_{index:03d}"))
        if row["id"] in ids:
            raise ValueError("Duplicate case ID: " + row["id"])
        ids.add(row["id"])
        if not isinstance(row.get("label"), bool):
            raise ValueError("Case label must be a boolean: " + row["id"])
        if len(row.get("sha256", "")) != 64:
            raise ValueError("Case requires source SHA256: " + row["id"])
        row["path"] = str(resolve(row["path"]))
        start, end = row.get("score_start"), row.get("score_end")
        if (start is None) != (end is None):
            raise ValueError("Both score_start and score_end are required")
        if start is not None and (not all(isinstance(x, (float, int)) and
                not isinstance(x, bool) and math.isfinite(x) for x in (start, end))
                or not 0 <= start < end):
            raise ValueError("Invalid scoring interval: " + row["id"])
        normalized.append(row)
    return normalized


def baseline_rows(path):
    data = read_json(path)
    rows = []
    for row in data.get("results", data.get("rows", [])):
        case = row.get("case", row)
        if row.get("error") or row.get("status") == "error":
            continue
        alerts = row.get("alerts", row.get("after", []))
        rows.append({"case": case, "alert": bool(row.get("alert", alerts)),
                     "alerts": alerts, "baseline_path": str(path)})
    return rows


def find_baseline(case, baselines):
    matches = []
    for item in baselines:
        previous = item["case"]
        same_path = previous.get("path") and resolve(previous["path"]) == Path(case["path"])
        same_sha = previous.get("sha256") == case["sha256"]
        same_interval = all(previous.get(key) == case.get(key)
                            for key in ("score_start", "score_end"))
        if (same_path or same_sha) and same_interval:
            if previous.get("sha256") and not same_sha:
                raise ValueError("Baseline path has different content: " + case["id"])
            if previous.get("label") is not case["label"]:
                raise ValueError("Baseline label differs: " + case["id"])
            matches.append(item)
    if len({item["alert"] for item in matches}) > 1:
        raise ValueError("Conflicting baselines; select one with --baseline: " + case["id"])
    return matches[0] if matches else None


def strict_interval_crop(cv2, source, target, start, end):
    """Copy native CFR BGR frames in [start,end), then verify decoded pixels.

    Refuse VFR/unstable timestamps instead of silently retiming evidence. Use
    the same timestamp fallback as analyze_clip. No interpolation or padding.
    """
    cap = cv2.VideoCapture(str(source))
    writer = None
    frames, expected, native_shape = [], hashlib.sha256(), None
    last, index, first_time, fps = -1., 0, None, cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not math.isfinite(fps) or not 1 <= fps <= 120:
        cap.release()
        raise ValueError("Interval crop needs an opened source with valid native FPS")
    tolerance = max(.002, .02 / fps)
    observed_end = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            second = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(second) or second <= last:
                second = max(index / fps, last + 1 / fps)
            if first_time is None:
                first_time = second
            if abs(second - (first_time + index / fps)) > tolerance:
                raise ValueError("VFR/unstable source timestamp: strict interval crop refused")
            last = second
            if second >= end:
                observed_end = True
                break
            if second >= start:
                if writer is None:
                    height, width = frame.shape[:2]
                    native_shape = tuple(frame.shape)
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"FFV1"),
                                             fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("FFV1 unavailable; refusing lossy interval crop")
                if tuple(frame.shape) != native_shape:
                    raise ValueError("Source geometry changed inside scoring interval")
                expected.update(frame.tobytes())
                writer.write(frame)
                frames.append({"source_frame_id": index, "source_second": second})
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not frames or frames[0]["source_second"] - start > 1 / fps + tolerance:
        raise ValueError("No complete native frame coverage for interval start")
    if not observed_end and last + 1 / fps < end - tolerance:
        raise ValueError("Requested interval extends beyond source end")
    verify = cv2.VideoCapture(str(target))
    actual, count, local_last = hashlib.sha256(), 0, -1.
    try:
        if not verify.isOpened():
            raise ValueError("Cannot decode FFV1 interval crop")
        while True:
            ok, frame = verify.read()
            if not ok:
                break
            if count >= len(frames):
                raise ValueError("Interval crop contains extra frames")
            if tuple(frame.shape) != native_shape:
                raise ValueError("FFV1 crop changed native frame geometry")
            second = verify.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(second) or second <= local_last:
                second = max(count / fps, local_last + 1 / fps)
            if abs(second - count / fps) > tolerance:
                raise ValueError("FFV1 decoder timestamp changed native timing")
            local_last = second
            frames[count]["local_second"] = second
            actual.update(frame.tobytes())
            count += 1
    finally:
        verify.release()
    if count != len(frames) or actual.hexdigest() != expected.hexdigest():
        raise ValueError("FFV1 crop failed pixel-identical native-frame verification")
    return {"method": "native_cfr_ffv1_pixel_verified", "path": str(target),
            "sha256": digest(target), "fps": fps, "score_start": start,
            "score_end": end, "native_shape": native_shape,
            "frames": frames, "pixel_sha256": actual.hexdigest(),
            "fresh_state": True, "no_pre_interval_context": True}


def assess_alerts(raw, interval):
    """Labels are not used here or passed to detection; map real event times."""
    alerts = copy.deepcopy(raw["automatic_alerts"])
    if interval is None:
        return alerts
    frames = interval["frames"]
    for alert in alerts:
        local = float(alert["second"])
        native = min(frames, key=lambda frame: abs(frame["local_second"] - local))
        if abs(native["local_second"] - local) > .0011:
            raise ValueError("Alert timestamp cannot be mapped to a native crop frame")
        alert["local_second"] = local
        alert["second"] = native["source_second"]
        alert["source_frame_id"] = native["source_frame_id"]
        if not interval["score_start"] <= alert["second"] < interval["score_end"]:
            raise ValueError("Alert escaped the labeled interval")
    return alerts


def metrics(rows):
    counts = dict(TP=0, TN=0, FP=0, FN=0, errors=0)
    for row in rows:
        if row.get("status") != "ok":
            counts["errors"] += 1
        else:
            counts[("T" if row["label"] == row["alert"] else "F") +
                   ("P" if row["alert"] else "N")] += 1
    counts["evaluated"] = sum(counts[key] for key in ("TP", "TN", "FP", "FN"))
    counts["total"] = len(rows)
    counts["recall"] = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else None
    counts["precision"] = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else None
    return counts


def preservation(rows):
    matched = [row for row in rows if row.get("baseline") is not None]
    previous = [row for row in matched if row["label"] and row["baseline"]["alert"]]
    lost = [row["id"] for row in previous if row["status"] == "ok" and not row["alert"]]
    unassessed = [row["id"] for row in previous if row["status"] != "ok"]
    return {"matched": len(matched), "unmatched_ids": [r["id"] for r in rows if r.get("baseline") is None],
            "previous_true_positives": len(previous), "lost_true_positive_ids": lost,
            "unassessed_previous_true_positive_ids": unassessed,
            "all_previous_true_positives_preserved": not lost and not unassessed,
            "alert_changed_ids": [r["id"] for r in matched if r["status"] == "ok"
                                  and r["alert"] != r["baseline"]["alert"]],
            "interval_comparability": "Strict crop resets all state at label start. Historical full-clip context is not equivalent."}


def run(args):
    import main2 as app
    manifest_path, output = resolve(args.manifest), resolve(args.output)
    manifest = read_json(manifest_path)
    cases = manifest_records(manifest)
    # Historical config fixes the original visual baseline independently of
    # whichever model/settings the GUI currently has selected.
    config_source = resolve(args.config) if args.config else DEFAULT_MANIFEST
    if args.config:
        cfg = app.load_config(str(config_source))
    else:
        cfg = app.merge_config(app.DEFAULT_CONFIG, read_json(config_source)["runtime"]["config"])
    cfg["model"]["video_context_enabled"] = args.enable_context
    cfg["model"]["video_context_kind"] = getattr(args, 'context_kind', 'r3d')
    cfg["model"]["video_context_head"] = str(resolve(args.checkpoint))
    cfg["model"]["video_context_backbone"] = str(resolve(args.backbone))
    cfg["notifications"]["enabled"] = False
    cfg["display"]["sound"] = False
    cfg["evidence"]["enabled"] = cfg["evidence"]["continuous"] = False
    cfg["feedback"]["enabled"] = False
    cfg = app.validate_config(cfg)
    baseline_paths = [resolve(path) for path in args.baseline] if args.baseline else list(DEFAULT_BASELINES)
    baselines = [row for path in baseline_paths for row in baseline_rows(path)]
    references = {case["id"]: find_baseline(case, baselines) for case in cases}
    backend = cfg["model"]["backend"]
    if backend == "auto":
        backend = "resnet" if resolve(cfg["model"]["path"]).is_file() else "keras"
    model_paths = [resolve(cfg["model"]["keras_path"] if backend == "keras" else cfg["model"]["path"])]
    if cfg["detection"]["require_interaction"]:
        if not cfg["person"]["enabled"]:
            raise ValueError("Interaction benchmark requires enabled people model")
        model_paths.append(resolve(cfg["person"]["path"]))
    if args.enable_context:
        model_paths.extend([resolve(args.checkpoint), resolve(args.backbone)])
    if cfg["model"].get("feature_weights"):
        model_paths.append(resolve(cfg["model"]["feature_weights"]))
    model_hashes = {str(path): digest(path) for path in model_paths}
    code_paths = [ROOT / name for name in ("verify_recall_runtime.py", "main2.py", "school_policy.py",
                  "temporal_visual.py", "action_context.py", "action_head.py", "action_runtime.py", "action_sampling.py",
                  "vit_runtime.py", "vit_causal_context.py", "explore_vit_causal.py", "train_vit_causal.py",
                  "probe_vit_recall.py")]
    code_hashes = {str(path): digest(path) for path in code_paths}
    # Refuse every existing directory, including an incomplete run. Retrying
    # uses a new destination; no historical result is resumed or overwritten.
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "intervals").mkdir()
    safe_config = redact_config(cfg)
    config_bytes = json.dumps(safe_config, sort_keys=True, allow_nan=False).encode("utf-8")
    report = {"schema": 1, "state": "running", "started": utc_now(), "scope": SCOPE,
              "human_review_pending": True, "manifest": str(manifest_path),
              "manifest_sha256": digest(manifest_path), "planned": len(cases),
              "context_enabled": args.enable_context, "runtime_config_redacted": safe_config,
              "runtime_config_redacted_sha256": hashlib.sha256(config_bytes).hexdigest(),
              "config_source": str(config_source), "config_source_sha256": digest(config_source),
              "models": model_hashes, "implementation": code_hashes,
              "baselines": {str(path): digest(path) for path in baseline_paths},
              "external_messages_sent": 0, "audio_enabled": False,
              "label_use": "Detector receives only input path, runtime config, and engines. Labels assessed after raw detector output is saved.",
              "interval_protocol": "Only actual native frames inside [score_start,score_end); FFV1 pixel verification; fresh per-call detector state. Unsupported VFR is an error.",
              "results": []}
    write_json(output / "results.json", report)
    try:
        app.runtime_imports()
        engine = app.VisualEngine(cfg["model"])
        people = app.PeopleEngine(cfg["person"]) if cfg["detection"]["require_interaction"] else None
        if people is not None and people.model is None:
            raise RuntimeError("People model unavailable: " + people.reason)
        if args.enable_context and getattr(engine, "video_context", None) is None:
            raise RuntimeError("Explicit context request did not load action engines")
        report["model_identity"] = engine.identity
        if args.enable_context:
            report["action_sampling"] = engine.video_context[1].sampling_profile.as_dict()
        for index, case in enumerate(cases):
            started, interval = time.monotonic(), None
            row = {key: case[key] for key in ("id", "path", "sha256", "label")}
            for key in ("category", "source_group", "origin", "score_start", "score_end"):
                if key in case:
                    row[key] = case[key]
            row["baseline"] = references[case["id"]]
            try:
                source = Path(case["path"])
                if digest(source) != case["sha256"]:
                    raise ValueError("Source SHA256 differs from manifest")
                detector_input = source
                if "score_start" in case:
                    detector_input = output / "intervals" / f"{index:03d}.avi"
                    interval = strict_interval_crop(app.cv2, source, detector_input,
                                                    case["score_start"], case["score_end"])
                    write_json(output / "intervals" / f"{index:03d}.json", interval)
                raw = app.analyze_clip(detector_input, cfg, engine=engine,
                                       people_engine=people,
                                       emit=lambda message: print(message, flush=True))
                raw_path = output / "raw" / f"{index:03d}.json"
                write_json(raw_path, raw)
                row["raw_path"], row["raw_sha256"] = str(raw_path), digest(raw_path)
                if raw.get("cancelled"):
                    raise RuntimeError("Detector cancelled before clip completion")
                if digest(source) != case["sha256"]:
                    raise ValueError("Source changed during runtime inference")
                alerts = assess_alerts(raw, interval)
                row.update(status="ok", alert=bool(alerts), alerts=alerts,
                           action_context_enabled=raw.get("action_context_enabled", False))
                if args.enable_context and not row["action_context_enabled"]:
                    raise RuntimeError("Detector did not enable action monitor")
            except Exception as exc:
                row.update(status="error", error=app.redact(exc))
                row.pop("alert", None)
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            row["interval_metadata"] = str(output / "intervals" / f"{index:03d}.json") if interval else None
            report["results"].append(row)
            report["metrics"] = metrics(report["results"])
            report["preservation"] = preservation(report["results"])
            write_json(output / "results.json", report)
            print(f"{index + 1}/{len(cases)} {case['id']}: {row['status']}", flush=True)
        if any(digest(Path(path)) != expected for path, expected in model_hashes.items()):
            raise RuntimeError("Model file changed during benchmark")
        if any(digest(Path(path)) != expected for path, expected in code_hashes.items()):
            raise RuntimeError("Runtime implementation changed during benchmark")
        if digest(manifest_path) != report["manifest_sha256"]:
            raise RuntimeError("Manifest changed during benchmark")
        report["state"] = "complete" if not report["metrics"]["errors"] else "complete_with_errors"
    except BaseException as exc:
        report.update(state="failed", fatal_error=app.redact(exc))
        raise
    finally:
        report["ended"] = utc_now()
        write_json(output / "results.json", report)
    print(json.dumps({"output": str(output), "metrics": report["metrics"],
                      "preservation": report["preservation"]}, ensure_ascii=False), flush=True)
    return 0 if report["state"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", required=True, help="New output directory; existing paths refused")
    parser.add_argument("--checkpoint", default="models/action_recall_public_20261016.pt", help="Local frozen action head")
    parser.add_argument("--backbone", default="models/r3d_18_kinetics400.pt", help="Local frozen R3D weights")
    parser.add_argument("--enable-context", action="store_true", help="Explicitly enable causal video branch")
    parser.add_argument("--context-kind", choices=('r3d', 'vit_cls_research'), default='r3d',
                        help="Explicit research-only CLS adapter or original R3D")
    parser.add_argument("--config", help="Optional runtime config; default is historical latest25 original config")
    parser.add_argument("--baseline", action="append", help="Reference results JSON; repeat to combine nonconflicting references")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
