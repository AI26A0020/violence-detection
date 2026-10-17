"""Explicit offline experiment; --extract is the only pretrained-model loading mode."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from vit_causal_context import PROFILE,CausalCLSWindow,prepare_frame_rgb224
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'schoolguard_data/retraining/vit_causal_20261016_v3'
REPORT=ROOT/'reports/recall_20261016/vit_causal_v3'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)==data:return
        raise FileExistsError('Refuse to overwrite experiment artifact: '+str(path))
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def validate_splits(rows):
    if not rows or any(r.get('training_split') not in ('train','validation') or type(r.get('label')) not in (int,bool) or r['label'] not in (0,1) for r in rows):raise ValueError('Invalid split/labels')
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate source bytes')
    for key in ('source_group','sha256'):
        if {r[key] for r in rows if r['training_split']=='train'}&{r[key] for r in rows if r['training_split']=='validation'}:raise ValueError('Split leakage')
    for split in ('train','validation'):
        if {int(r['label']) for r in rows if r['training_split']==split}!={0,1}:raise ValueError('Both source classes required per split')

def validate_cache(path,source_sha,backbone_sha,manifest_sha):
    required={'features','seconds','start_seconds','frame_ids','frame_seconds','segments','real_frames','source_sha256','backbone_sha256','schema','profile_json','manifest_sha256'}
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=required:raise ValueError('Cache schema keys differ')
        for key,value in [('source_sha256',source_sha),('backbone_sha256',backbone_sha),('schema',PROFILE['schema']),('manifest_sha256',manifest_sha)]:
            if f[key].shape!=() or str(f[key])!=value:raise ValueError('Cache provenance mismatch: '+key)
        if f['profile_json'].shape!=() or json.loads(str(f['profile_json']))!=PROFILE:raise ValueError('Cache profile differs')
        arrays={k:f[k].copy() for k in required if k not in {'source_sha256','backbone_sha256','schema','profile_json','manifest_sha256'}}
    x=arrays['features'];n=len(x)
    if x.dtype!=np.float32 or x.shape!=(n,2304) or not np.isfinite(x).all():raise ValueError('Invalid features')
    for key in ('seconds','start_seconds','segments','real_frames'):
        if arrays[key].shape!=(n,):raise ValueError('Cache alignment differs')
    for key in ('frame_ids','frame_seconds'):
        if arrays[key].shape!=(n,4):raise ValueError('Four real observations required')
    t=arrays['frame_seconds'];ids=arrays['frame_ids'];segments=arrays['segments']
    if not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(segments.dtype,np.integer):raise ValueError('Integer identities required')
    if not np.isfinite(t).all() or np.any(t<0) or np.any(ids<0) or np.any(segments<0):raise ValueError('Invalid observation identities')
    if np.any(np.diff(t,axis=1)<=0) or np.any(np.diff(t,axis=1)>PROFILE['max_gap']+1e-6) or np.any(np.diff(ids,axis=1)<=0):raise ValueError('Repeated/reordered/gapped window')
    if np.any(t[:,-1]-t[:,0]+1e-6<PROFILE['min_span']) or np.any(arrays['real_frames']!=4):raise ValueError('Fabricated warmup')
    if not np.array_equal(arrays['seconds'],t[:,-1]) or not np.array_equal(arrays['start_seconds'],t[:,0]):raise ValueError('Timestamps unaligned')
    if n>1 and (np.any(np.diff(arrays['seconds'])<=0) or np.any(np.diff(ids[:,-1])<=0) or np.any(np.diff(segments)<0)):raise ValueError('Reordered windows')
    for segment in set(segments.tolist()):
        stamps=arrays['seconds'][segments==segment]
        if np.any(np.diff(stamps)+1e-6<PROFILE['score_interval']):raise ValueError('Score cadence violated')
    return arrays

def prepare():
    from types import SimpleNamespace
    from retrain_database import verify_final_holdout_separation
    source=ROOT/'schoolguard_data/retraining/action_film_recall_20261016/manifest.json'
    rows=read(source)['records']
    validate_splits(rows)
    verify_final_holdout_separation(SimpleNamespace(holdout=None),rows)
    groups={s:{r['source_group'] for r in rows if r['training_split']==s} for s in ('train','validation')}
    hashes={s:{r['sha256'] for r in rows if r['training_split']==s} for s in ('train','validation')}
    if groups['train']&groups['validation'] or hashes['train']&hashes['validation']:raise ValueError('Split leakage')
    if any(r['training_split'] not in groups or type(r['label']) not in (int,bool) or r['label'] not in (0,1) for r in rows):raise ValueError('Invalid labels/split')
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate source bytes')
    for row in rows:
        if digest(row['path'])!=row['sha256']:raise ValueError('Changed source')
    save(RUN/'manifest.json',dict(records=rows,source_manifest=str(source),source_manifest_sha256=digest(source),
        profile=PROFILE,scope='Source-separated provisional development labels; latest25 excluded from training and checkpoint selection',
        extraction_segments='Sources <=4sec read full; longer sources first2sec and last2sec separately; no cross-seek context',
        human_review_pending=True))
    print('Prepared',len(rows),'sources; no model loaded',flush=True)


def audit_timestamps(stamps,fps,reported_count):
    """Accept CFR timestamps with constant offset and a missing/repeated tail only."""
    if not np.isfinite(fps) or not 1<=fps<=120:raise ValueError('Invalid native FPS')
    if len(stamps)<2:raise ValueError('Insufficient decoded observations for CFR audit')
    offset=None;valid=0;anomalies=0;tail=False;previous=None
    tolerance=max(0.002,0.05/fps)
    for i,stamp in enumerate(stamps):
        good=np.isfinite(stamp) and stamp>=0 and (previous is None or stamp>previous)
        if not good:
            tail=True;anomalies+=1;continue
        if tail:raise ValueError('Interior timestamp anomaly or inconsistent clock')
        if offset is None:offset=float(stamp)-i/fps
        if abs(float(stamp)-(offset+i/fps))>tolerance:raise ValueError('Non-CFR timestamp progression')
        previous=float(stamp);valid+=1
    if valid<2:raise ValueError('CFR clock cannot be verified')
    return dict(clock_policy=PROFILE['clock'],fps=float(fps),actual_decoded_count=len(stamps),
        container_reported_count=float(reported_count),timestamp_anomaly_count=anomalies,
        valid_timestamp_count=valid,decoder_timestamp_offset=offset,tolerance_seconds=tolerance)

def audit_source(path):
    import cv2
    cap=cv2.VideoCapture(str(path));stamps=[]
    if not cap.isOpened():raise ValueError('Cannot open source')
    fps=cap.get(cv2.CAP_PROP_FPS);reported=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame is None or not frame.size:raise ValueError('Empty successful decoded frame')
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    finally:cap.release()
    return audit_timestamps(stamps,fps,reported)

def extract(which):
    import cv2
    import main2 as app
    from vit_causal_context import FrozenViTCLS
    app.runtime_imports();cv2.setNumThreads(1)
    training=read(RUN/'manifest.json');rows=training['records']
    if which=='latest25':rows=read(ROOT/'reports/requested_revision_20261015/holdout_new/results.json')['results']
    folder=RUN/'features' if which=='training' else REPORT/'latest25_features';folder.mkdir(parents=True,exist_ok=True)
    audits={}
    for index,row in enumerate(rows):
        if digest(row['path'])!=row['sha256']:raise ValueError('Changed source before audit')
        audit=audit_source(row['path'])
        if digest(row['path'])!=row['sha256']:raise ValueError('Changed source during audit')
        audits[row['sha256']]=audit
        save(folder/'decode_audits'/(row['sha256']+'.json'),dict(source_sha256=row['sha256'],**audit))
        print('CFR audit',index+1,'/',len(rows),audit,flush=True)
    model=FrozenViTCLS();observations=0;started=time.monotonic()
    manifest_sha=digest(RUN/'manifest.json')
    save(folder/'provenance.json',dict(profile=PROFILE,backbone_sha256=model.weights_sha256,
        manifest_sha256=digest(RUN/'manifest.json'),which=which,
        preprocessing_status='Assumed standard ViT RGB bilinear224 mean/std.5; cached metadata does not supply preprocessing. CLS avoids unknown original class mapping.',
        architecture='Strict timm-to-torchvision all-key mapping; timm LayerNorm epsilon1e-6 assumption',
        scene_detection_initialized=True,scene_cut_cadence='every genuine native decoded frame',
        timestamp_policy=PROFILE['clock'],latest25_used_for_fit=False))
    for index,row in enumerate(rows):
        sha=row['sha256'];dest=folder/(sha+'.npz')
        if digest(row['path'])!=sha:raise ValueError('Changed source')
        if dest.exists():
            validate_cache(dest,sha,model.weights_sha256,manifest_sha)
            trace=read(folder/(sha+'.json'))
            if trace['cache_sha256']!=digest(dest):raise ValueError('Trace/cache binding mismatch')
            print('Cached',index+1,len(rows),flush=True);continue
        audit=audits[sha];fps=audit['fps'];n=audit['actual_decoded_count'];duration=n/fps
        segments=[(0,duration)] if which=='latest25' or duration<=4 else [(0,2),(max(0,duration-2),duration)]
        outputs=[];sample_count=0;cuts=0;window=CausalCLSWindow()
        cap=cv2.VideoCapture(row['path']);frame_id=0;active=None;next_sample=0.;scene=None
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame is None or not frame.size:raise ValueError('Empty decoded frame')
                if frame_id>=n:raise ValueError('Decode count changed since audit')
                stamp=frame_id/fps
                segment=next((j for j,(start,end) in enumerate(segments) if start-1e-6<=stamp<end-1e-6),None)
                if segment is not None:
                    if segment!=active:
                        window.reset();scene=app.SceneCutDetector();active=segment
                        next_sample=np.ceil((segments[segment][0]-1e-6)*4)/4
                    cut=bool(scene.check_cut(frame))
                    if cut:window.reset();cuts+=1
                    if stamp+1e-6>=next_sample:
                        while next_sample<=stamp+1e-6:next_sample+=.25
                        feature=model.encode_frame(prepare_frame_rgb224(frame));sample_count+=1;observations+=1
                        value=window.observe(feature,stamp,frame_id)
                        if value is not None:outputs.append(value)
                frame_id+=1
            if frame_id!=n:raise ValueError('Decode count changed since audit')
        finally:cap.release()
        if digest(row['path'])!=sha:raise ValueError('Source changed during decode')
        features=np.stack([x['feature'] for x in outputs]) if outputs else np.empty((0,2304),np.float32)
        np.savez_compressed(dest,features=features,seconds=np.array([x['second'] for x in outputs]),
            start_seconds=np.array([x['start_second'] for x in outputs]),
            frame_ids=np.array([x['frame_ids'] for x in outputs],dtype=np.int64).reshape(-1,4),
            frame_seconds=np.array([x['frame_seconds'] for x in outputs],dtype=np.float64).reshape(-1,4),
            segments=np.array([x['segment'] for x in outputs],dtype=np.int64),
            real_frames=np.array([x['real_frames'] for x in outputs]),
            source_sha256=np.asarray(sha),backbone_sha256=np.asarray(model.weights_sha256),schema=np.asarray(PROFILE['schema']),
            profile_json=np.asarray(json.dumps(PROFILE,sort_keys=True)),manifest_sha256=np.asarray(manifest_sha))
        validate_cache(dest,sha,model.weights_sha256,manifest_sha)
        save(folder/(sha+'.json'),dict(path=row['path'],sampled_native_frames=sample_count,scene_cuts=cuts,
            cache_sha256=digest(dest),source_sha256=sha,
            decode_audit=audit,selected_segments=segments,windows=[{k:v for k,v in x.items() if k!='feature'} for x in outputs]))
        print('ViT CLS',index+1,'/',len(rows),'frames',sample_count,'windows',len(outputs),'elapsed',round(time.monotonic()-started,1),flush=True)
    if not (folder/'completed.json').exists():save(folder/'completed.json',dict(sources=len(rows),new_frame_forwards=observations,elapsed=time.monotonic()-started))

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--extract',choices=['training','latest25']);args=p.parse_args()
    if args.prepare:prepare()
    if args.extract:extract(args.extract)
if __name__=='__main__':main()
