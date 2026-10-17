"""Offline precision diagnosis with frozen inputs and current pair action evidence.

Replay may change interaction logic only. Detector/tracker candidates require fresh
inference. No NotificationQueue is constructed, and no network API is used.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import time
import types

ROOT = Path(__file__).resolve().parent
SECTIONS = ("model", "person", "detection", "audio")
MARGIN = .21


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_source(text):
    return re.sub(r"https://discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_-]+",
                  "DISCORD_WEBHOOK_REDACTED", text)


def load_app(path):
    app = types.ModuleType("precision_app")
    app.__file__ = str(ROOT/"main1.py")
    app.__precision_source__ = str(Path(path).resolve())
    sys.modules[app.__name__] = app
    exec(compile(Path(path).read_text(encoding="utf-8-sig"), str(path), "exec"), app.__dict__)
    app.runtime_imports()
    return app


def instrument(app):
    original = app.InteractionVerifier
    signature = inspect.signature(original.update)

    class DiagnosticVerifier(original):
        def update(self, t, boxes, frame_shape, visual, *args, **kwargs):
            # Bind against the loaded runtime: main1 has an audio argument;
            # current main2 is visual-only. Preserve each API's calling rules.
            bound = signature.bind(self, t, boxes, frame_shape, visual, *args, **kwargs)
            result = super().update(t, boxes, frame_shape, visual, *args, **kwargs)
            cfg = bound.arguments.get("detection") or app.DEFAULT_CONFIG["detection"]
            gap_limit = max(.45, 2.1/cfg["analysis_fps"])
            supported = {tuple(p["tracks"]) for p in result["pairs"]}
            diagnostics = []
            for key, history in self.pairs.items():
                if not history or history[-1]["t"] != t:
                    continue
                events = [r for r in history if r["near"] and r["action"]]
                strong = sum(r["strike"] >= 1.8 for r in events) >= 3
                events = [r for r in events if not r["walk"] or strong]
                for i in range(len(events)-1, 0, -1):
                    if events[i]["t"]-events[i-1]["t"] > gap_limit:
                        events = events[i:]
                        break
                diagnostics.append({"tracks": list(key), **history[-1],
                                    "supported": key in supported,
                                    "action_times": [r["t"] for r in events]})
            result["diagnostics"] = diagnostics
            return result

    app.InteractionVerifier = DiagnosticVerifier


class SilentSource:
    pipeline = None
    reason = "Offline visual/person verification; audio sources excluded"


class Replay:
    def __init__(self, app, analysis, people):
        self.app, self.analysis, self.people = app, analysis, people
        self.visual_index = self.people_index = 0
        self.identity = analysis["model"]

    def score(self, frame, state):
        row = self.analysis["timeline"][self.visual_index]
        self.visual_index += 1
        state.smooth = row["smooth"]
        return row["visual"], row["smooth"]

    def detect(self, frame, tracker):
        row = self.people[self.people_index]
        self.people_index += 1
        if list(frame.shape) != row["shape"]:
            raise ValueError("Replay frame shape mismatch")
        return [self.app.PersonBox(**{**box, "keypoints": self.app.np.asarray(box["keypoints"])
                                     if box["keypoints"] is not None else None}) for box in row["boxes"]]

    def check(self, row):
        if self.visual_index != self.people_index or row["second"] != self.people[self.people_index-1]["second"]:
            raise ValueError("Replay timestamp/count mismatch")


def score_intervals(annotations, analyses):
    rows = []
    for annotation in annotations:
        a, b = annotation["start"], annotation["end"]
        analysis = analyses[annotation["video"]]
        samples = [r for r in analysis["timeline"] if a-MARGIN <= r["second"] <= b+MARGIN]
        confirmations, actions = {}, {}
        alerts = [r["second"] for r in samples if a <= r["second"] <= b and r["automatic_alert"]]
        for sample in samples:
            for pair in sample["interaction"].get("diagnostics", []):
                key = tuple(pair["tracks"])
                times = [t for t in pair["action_times"] if a-MARGIN <= t <= b+MARGIN]
                actions.setdefault(key, set()).update(times)
                if pair["supported"] and len(times) >= 2:
                    confirmations.setdefault(key, sample["second"])
        pairs = [{"tracks": list(key), "first_action": min(actions[key]),
                  "confirmation": second, "latency_from_first_action": round(second-min(actions[key]), 4),
                  "action_samples": len(actions[key])} for key, second in confirmations.items()]
        rows.append({**annotation, "confirmed_pairs": pairs, "fresh_interaction_confirmed": bool(pairs),
                     "notification_fire_seconds": alerts,
                     "lingering_supported_samples": sum(r["interaction"]["supported"] for r in samples),
                     "scored": annotation["review_label"] != "ambiguous"})
    positive = [r for r in rows if r["review_label"] == "physical_interaction"]
    negative = [r for r in rows if r["review_label"] == "negative"]
    return {"rules": {"sampling_margin_seconds": MARGIN, "minimum_current_interval_action_samples": 2,
                       "ambiguous_excluded": True, "lingering_gate_is_not_detection": True},
            "physical_interaction_intervals": len(positive),
            "confirmed_interaction_intervals": sum(r["fresh_interaction_confirmed"] for r in positive),
            "negative_intervals": len(negative),
            "negative_intervals_with_supported_evidence": sum(r["fresh_interaction_confirmed"] for r in negative),
            "negative_intervals_with_notifications": sum(bool(r["notification_fire_seconds"]) for r in negative),
            "ambiguous_intervals": sum(r["review_label"] == "ambiguous" for r in rows), "intervals": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "fresh"), default="replay")
    parser.add_argument("--module", type=Path, default=ROOT/"main1.py")
    parser.add_argument("--input", type=Path, default=ROOT/"reports/precision_20261013/baseline_videos")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("unchanged", "stable_pose", "stable_anchor", "recovery", "anchor_tracker"), default="unchanged")
    parser.add_argument("--only", action="append")
    parser.add_argument("--annotations", type=Path, default=ROOT/"reports/precision_20261013/scenarios_locked.json")
    args = parser.parse_args()
    out = args.output.resolve()
    source = args.input.resolve()
    if out == source:
        raise ValueError("Keep producer evidence separate")
    out.mkdir(parents=True, exist_ok=True)
    app = load_app(args.module)
    cfg = app.load_config()
    original_config = json.loads((source/"configuration.json").read_text(encoding="utf-8"))
    cfg.update(original_config["config"])
    cfg["notifications"]["enabled"] = False
    cfg["audio"]["enabled"] = cfg["audio"]["asr_enabled"] = False
    app.validate_config(cfg)
    model_hashes = {k: sha256(app.resolve_path(cfg[section]["path"]))
                    for k, section in (("model_sha256", "model"), ("person_model_sha256", "person"))}
    producer_hashes = {"model_sha256": original_config["model_sha256"],
                       "person_model_sha256": original_config.get("person_model_sha256", original_config.get("person_sha256"))}
    if any(producer_hashes[k] != v for k, v in model_hashes.items()):
        raise ValueError("Producer weights changed")
    # Tracked boxes are not raw detections: only interaction changes are replayable.
    if args.mode == "replay":
        if args.variant in ('recovery', 'anchor_tracker'):
            raise ValueError('Detector/tracker variants require fresh inference')
        import ast
        producer_path = source/"main1_before_redacted.py"
        def inference_tree(path):
            tree = ast.parse(safe_source(Path(path).read_text(encoding="utf-8-sig")))
            tree.body = [n for n in tree.body if not isinstance(n, ast.ClassDef) or n.name != "InteractionVerifier"]
            return ast.dump(tree, include_attributes=False)
        if inference_tree(args.module) != inference_tree(producer_path):
            raise ValueError("Detector/tracker/capture changed; fresh inference required")
    if args.variant in ("stable_pose", "stable_anchor", "anchor_tracker"):
        from precision_candidates import stable_pose_verifier
        app.InteractionVerifier = stable_pose_verifier(app, common_scale=args.variant == 'stable_pose')
    if args.variant == 'recovery':
        from precision_candidates import recovery_people_engine
        app.PeopleEngine = recovery_people_engine(app)
    if args.variant == 'anchor_tracker':
        from precision_candidates import bounded_tracker
        app.SimpleTracker = bounded_tracker(app)
    instrument(app)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    snapshot = safe_source(args.module.read_text(encoding="utf-8-sig"))
    (out/"main1_redacted.py").write_text(snapshot, encoding="utf-8")
    (out/"verifier_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out/"candidate_snapshot.py").write_bytes((ROOT/"precision_candidates.py").read_bytes())
    configuration = {"code_sha256": sha256(args.module), "snapshot_sha256": sha256(out/"main1_redacted.py"),
                     "variant": args.variant, "mode": args.mode, **model_hashes,
                     "config": {s: cfg[s] for s in SECTIONS}, "external_alerts_sent": False,
                     "annotation_sha256": sha256(args.annotations), "verifier_sha256": sha256(__file__),
                     "candidate_sha256": sha256(ROOT/"precision_candidates.py")}
    app.atomic_json(out/"configuration.json", configuration)
    paths = sorted((ROOT/"AI26A0020/test_videos").glob("*.mp4"), key=lambda p: int(re.search(r"\d+", p.stem)[0]))
    if args.only:
        paths = [p for p in paths if p.name in args.only]
    visual = people = None
    if args.mode == "fresh":
        visual, people = app.VisualEngine(cfg["model"]), app.PeopleEngine(cfg["person"])
        if people.model is None:
            raise RuntimeError(people.reason)
    analyses, results = {}, []
    import av
    for path in paths:
        with av.open(str(path)) as container:
            if container.streams.audio:
                raise ValueError("This evaluator requires silent sources")
        baseline = json.loads((source/(path.stem+"_analysis.json")).read_text(encoding="utf-8"))
        if baseline["cancelled"] or baseline["verification"]["input_sha256"] != sha256(path):
            raise ValueError("Producer input changed or is incomplete")
        replay = None
        if args.mode == "replay":
            rows = json.loads((source/(path.stem+"_people.json")).read_text(encoding="utf-8"))
            if len(rows) != len(baseline["timeline"]):
                raise ValueError("Incomplete producer poses")
            replay = Replay(app, baseline, rows)
            visual = people = replay
        print("START", path.name, args.mode, args.variant, flush=True)
        observations = []
        def observe(frame, row, boxes):
            if replay:
                replay.check(row)
            observations.append({"second": row["second"], "shape": list(frame.shape),
                                 "boxes": [dataclasses.asdict(b) for b in boxes]})
        started = time.perf_counter()
        result = app.analyze_clip(path, cfg, engine=visual, people_engine=people, audio_engine=SilentSource(),
                                  observer=observe, emit=lambda text: print(text, flush=True))
        result["wall_seconds"] = time.perf_counter()-started
        if replay and replay.visual_index != len(baseline["timeline"]):
            raise ValueError("Producer observations not fully consumed")
        stripped = json.loads(json.dumps(result["timeline"]))
        for row in stripped:
            row["interaction"].pop("diagnostics", None)
        result["same_decisions_as_producer"] = stripped == baseline["timeline"]
        if args.variant == "unchanged" and args.mode == "replay" and not result["same_decisions_as_producer"]:
            raise AssertionError("Baseline replay does not reproduce producer decisions")
        result["verification"] = {"input_sha256": sha256(path), "configuration_sha256": sha256(out/"configuration.json"),
                                  "producer_analysis_sha256": sha256(source/(path.stem+"_analysis.json")),
                                  "producer_people_sha256": sha256(source/(path.stem+"_people.json"))}
        app.atomic_json(out/(path.stem+"_analysis.json"), result)
        (out/(path.stem+"_people.json")).write_text(json.dumps(observations, default=lambda x: x.tolist(),
                                                            ensure_ascii=False), encoding="utf-8")
        analyses[path.name] = result
        results.append({"video": path.name, "samples": len(result["timeline"]),
                        "wall_seconds": result["wall_seconds"], "alerts": len(result["automatic_alerts"]),
                        "same_decisions_as_producer": result["same_decisions_as_producer"]})
        selected = [a for a in annotations if a["video"] in analyses]
        app.atomic_json(out/"interval_report.json", score_intervals(selected, analyses))
        app.atomic_json(out/"summary.json", {"results": results, "external_alerts_sent": False})
        print("DONE", results[-1], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
