"""Compare actual person detection on original and manually head-masked probes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import main1 as app


def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(1,union)


def measure(targets, boxes):
    detections=[b for b in boxes if b.observed]
    candidates=sorted(((overlap(t["xyxy"],[b.x1,b.y1,b.x2,b.y2]),i,j)
                       for i,t in enumerate(targets) for j,b in enumerate(detections)),reverse=True)
    used_targets,used_boxes,pairs=set(),set(),[]
    for iou,i,j in candidates:
        if iou<.3:
            break
        if i not in used_targets and j not in used_boxes:
            used_targets.add(i);used_boxes.add(j)
            pairs.append({"target":targets[i]["id"],"track":detections[j].track_id,"iou":round(iou,4)})
    return {"targets":len(targets),"matched_at_iou_0_3":len(pairs),"matched":pairs,
            "missing_target_ids":[t["id"] for i,t in enumerate(targets) if i not in used_targets],
            "observed_detections":len(detections),
            "unmatched_detections":len(detections)-len(used_boxes)}


def main():
    global app
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations",type=Path,default=app.ROOT/"reports/test_videos_20261012/person_probes/annotations.json")
    parser.add_argument("--output",type=Path,default=app.ROOT/"reports"/f"person_probe_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--model")
    parser.add_argument("--confidence",type=float)
    parser.add_argument("--imgsz",type=int)
    parser.add_argument("--enhancement",choices=("none","clahe"),default="none")
    parser.add_argument("--candidate", choices=("unchanged", "recovery"), default="unchanged",
                        help="Offline experiment; does not change application settings")
    parser.add_argument("--module", type=Path, help="Read a prepared/baseline source without activating it")
    args=parser.parse_args()
    if args.module:
        from verify_main1_precision import load_app
        app = load_app(args.module)
    else:
        app.runtime_imports()
    cfg=app.load_config()
    if args.model:cfg["person"]["path"]=args.model
    if args.confidence is not None:cfg["person"]["confidence"]=args.confidence
    if args.imgsz is not None:cfg["person"]["imgsz"]=args.imgsz
    app.validate_config(cfg)
    import torch
    torch.set_num_threads(cfg["model"]["threads"])
    if args.candidate == 'recovery':
        from precision_candidates import recovery_people_engine
        engine=recovery_people_engine(app)(cfg['person'])
    else:
        engine=app.PeopleEngine(cfg["person"])
    if engine.model is None:raise RuntimeError(engine.reason)
    if args.enhancement=="clahe":
        original_model=engine.model
        clahe=app.cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        def enhanced_model(frame,**kwargs):
            lab=app.cv2.cvtColor(frame,app.cv2.COLOR_BGR2LAB)
            lab[:,:,0]=clahe.apply(lab[:,:,0])
            return original_model(app.cv2.cvtColor(lab,app.cv2.COLOR_LAB2BGR),**kwargs)
        engine.model=enhanced_model
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    annotation=json.loads(args.annotations.read_text(encoding="utf-8"))
    results=[]
    for case in annotation["cases"]:
        frame=app.cv2.imread(case["frame"])
        if frame is None:raise FileNotFoundError(case["frame"])
        panels=[];row={"video":case["video"],"second":case["second"],"notes":case.get("notes","")}
        for mode in ("original","head_masked"):
            image=frame.copy()
            if mode=="head_masked":
                for target in case["targets"]:
                    x1,y1,x2,y2=target["head_xyxy"]
                    image[y1:y2,x1:x2]=(24,24,24)
            started=time.perf_counter()
            boxes=engine.detect(image,app.SimpleTracker())
            seconds=time.perf_counter()-started
            rx1,ry1,rx2,ry2=case["roi"]
            primary=[b for b in boxes if rx1<=b.center[0]<=rx2 and ry1<=b.center[1]<=ry2]
            result=measure(case["targets"],primary)
            result.update(latency_ms=round(seconds*1000,2),
                boxes=[{"xyxy":[round(b.x1,2),round(b.y1,2),round(b.x2,2),round(b.y2,2)],"confidence":round(b.confidence,4)} for b in primary])
            row[mode]=result
            for target in case["targets"]:
                x1,y1,x2,y2=target["xyxy"]
                app.cv2.rectangle(image,(x1,y1),(x2,y2),(0,230,0),1)
            for b in primary:
                x1,y1,x2,y2=b.int_box
                app.cv2.rectangle(image,(x1,y1),(x2,y2),(0,70,255),2)
            app.cv2.rectangle(image,(0,0),(image.shape[1],27),(0,0,0),-1)
            title=f"{case['video']} {case['second']}s {mode}: {result['matched_at_iou_0_3']}/{result['targets']} targets (green=reference red=detector)"
            app.cv2.putText(image,title,(8,19),app.cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,app.cv2.LINE_AA)
            panels.append(image)
        app.cv2.imwrite(str(output/(Path(case["video"]).stem+"_comparison.jpg")),app.np.concatenate(panels,axis=1))
        results.append(row)
        print(case["video"],"original",row["original"]["matched_at_iou_0_3"],"masked",row["head_masked"]["matched_at_iou_0_3"],"targets",row["original"]["targets"],flush=True)
    report={"created":app.now_iso(),"scope":annotation["scope"],"iou_threshold":.3,
            "config":cfg["person"],"experimental_enhancement":args.enhancement,"experimental_candidate":args.candidate,"main1_sha256":hashlib.sha256((args.module or app.ROOT/"main1.py").read_bytes()).hexdigest(),
            "annotation_sha256":hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            "model_sha256":hashlib.sha256(app.resolve_path(cfg["person"]["path"]).read_bytes()).hexdigest(),
            "targets":sum(r["original"]["targets"] for r in results),
            "original_matches":sum(r["original"]["matched_at_iou_0_3"] for r in results),
            "head_masked_matches":sum(r["head_masked"]["matched_at_iou_0_3"] for r in results),"results":results,
            "note":"Independent fresh tracking for every image. Masked matches require actual detector observations; retained old tracks do not count. Unmatched detections are not automatically false positives because annotation coverage is partial."}
    app.atomic_json(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("targets","original_matches","head_masked_matches")}),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
