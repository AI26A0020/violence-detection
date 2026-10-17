"""Small source-calibrated MIL head on cached CLS context; no pretrained model load."""
import argparse,copy
from pathlib import Path
import numpy as np
from vit_causal_context import PROFILE
from explore_vit_causal import RUN,REPORT,ROOT,read,save,digest,validate_cache,validate_splits

def head(torch,hidden):
    nn=torch.nn
    return nn.Sequential(nn.Linear(2304,hidden),nn.ReLU(),nn.Dropout(.4),nn.Linear(hidden,1)) if hidden else nn.Linear(2304,1)

def validate_canonical_splits(rows):
    validate_splits(rows)
    groups={split:set() for split in ('train','validation')}
    for row in rows:
        for key in ('source_group','original_source_group','canonical_movie_group'):
            value=row.get(key)
            if value:groups[row['training_split']].add(str(value).strip().casefold())
    if groups['train']&groups['validation']:raise ValueError('Canonical source alias split leakage')

def validate_decode_audit(audit,arrays):
    count=audit['actual_decoded_count'];reported=audit['container_reported_count'];fps=audit['fps']
    if (type(count) is not int or count<2 or not np.isfinite(reported) or reported!=int(reported)
        or abs(reported-count)>1 or audit['timestamp_anomaly_count'] not in (0,1)
        or audit['valid_timestamp_count']+audit['timestamp_anomaly_count']!=count
        or audit['valid_timestamp_count']<2 or audit['clock_policy']!=PROFILE['clock']
        or not np.isfinite(fps) or not 1<=fps<=120):raise ValueError('Unverified CFR decode audit')
    ids=arrays['frame_ids'];seconds=arrays['frame_seconds']
    if np.any(ids>=count) or not np.allclose(seconds,ids/fps,atol=1e-9,rtol=0):raise ValueError('Native frame clock mismatch')
    return audit

def bound_decode_audit(path,source_sha,arrays):
    trace=read(path.with_suffix('.json'))
    if trace['cache_sha256']!=digest(path) or trace['source_sha256']!=source_sha:raise ValueError('Trace/cache mismatch')
    audit=read(path.parent/'decode_audits'/(source_sha+'.json'))
    if audit.pop('source_sha256',None)!=source_sha:raise ValueError('Decode audit/source mismatch')
    validate_decode_audit(audit,arrays)
    if trace['decode_audit']!=audit:raise ValueError('Decode audit/trace mismatch')
    return audit

def fit():
    import torch
    from types import SimpleNamespace
    from retrain_database import verify_final_holdout_separation
    torch.set_num_threads(2);m=read(RUN/'manifest.json');rows=m['records']
    if (RUN/'candidate.pt').exists() or (REPORT/'fit.json').exists():raise FileExistsError('Preserve completed candidate/fit outputs')
    validate_canonical_splits(rows)
    verify_final_holdout_separation(SimpleNamespace(holdout=None),rows)
    if digest(m['source_manifest'])!=m['source_manifest_sha256']:raise ValueError('Manifest changed')
    provenance=read(RUN/'features/provenance.json');backbone=provenance['backbone_sha256']
    if provenance['profile']!=PROFILE or provenance['manifest_sha256']!=digest(RUN/'manifest.json'):raise ValueError('Feature provenance changed')
    packs=[]
    for row in rows:
        if digest(row['path'])!=row['sha256']:raise ValueError('Source changed')
        path=RUN/'features'/(row['sha256']+'.npz')
        arrays=validate_cache(path,row['sha256'],backbone,digest(RUN/'manifest.json'))
        bound_decode_audit(path,row['sha256'],arrays)
        packs.append(torch.from_numpy(arrays['features']))
    train=[i for i,r in enumerate(rows) if r['training_split']=='train' and len(packs[i])]
    valid=[i for i,r in enumerate(rows) if r['training_split']=='validation']
    for key in ('source_group','sha256'):
        if {r[key] for r in rows if r['training_split']=='train'}&{rows[i][key] for i in valid}:raise ValueError('Split leakage')
    mean=torch.stack([packs[i].mean(0) for i in train]).mean(0)
    scale=torch.stack([packs[i].mean(0) for i in train]).std(0).clamp_min(.1)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():raise ValueError('Invalid normalization')
    packs=[(x-mean)/scale for x in packs];candidates=[]
    for hidden in (0,64):
      for decay in (.01,.1):
        torch.manual_seed(261016);model=head(torch,hidden);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=decay);best=None
        for epoch in range(1,101):
          model.train();order=torch.randperm(len(train)).tolist()
          for at in range(0,len(order),24):
            ids=[train[j] for j in order[at:at+24]];terms=[]
            for i in ids:
                logits=model(packs[i]).flatten()
                terms.append(torch.nn.functional.softplus(-logits.max()) if rows[i]['label'] else torch.nn.functional.softplus(logits).mean()+.5*torch.nn.functional.softplus(logits.max()))
            loss=torch.stack(terms).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);optimizer.step()
          if epoch%5:continue
          model.eval()
          with torch.inference_mode():vals=[float(model(packs[i]).max()) if len(packs[i]) else None for i in valid]
          negatives=[v for i,v in zip(valid,vals) if not rows[i]['label'] and v is not None]
          if not negatives:raise ValueError('No nonempty validation negatives')
          offset=max(negatives)+.2
          tp=sum(bool(rows[i]['label']) and v is not None and v>=offset for i,v in zip(valid,vals))
          if best is None or tp>best['tp']:best=dict(tp=tp,epoch=epoch,offset=offset,state=copy.deepcopy(model.state_dict()),scores=vals)
        candidates.append(dict(hidden=hidden,decay=decay,**best));print('Fit CLS',hidden,decay,'TP',best['tp'],'offset',best['offset'],flush=True)
    chosen=max(candidates,key=lambda c:c['tp']);target=RUN/'candidate.pt'
    torch.save(dict(architecture='vit_cls_context_mil_v1',input_dim=2304,profile=PROFILE,backbone_sha256=backbone,
        hidden=chosen['hidden'],state_dict=chosen['state'],mean=mean,scale=scale,offset=chosen['offset'],
        manifest_sha256=digest(RUN/'manifest.json'),latest25_used_for_fit=False,latest25_used_for_selection=False),target)
    positives=sum(bool(rows[i]['label']) for i in valid)
    save(REPORT/'fit.json',dict(checkpoint=str(target),selected={k:v for k,v in chosen.items() if k!='state'},
      candidates=[{k:v for k,v in c.items() if k!='state'} for c in candidates],validation_positive=positives,
      validation_false_negative=positives-chosen['tp'],empty_validation_positive=sum(bool(rows[i]['label']) and not len(packs[i]) for i in valid),
      calibration='source validation maximum nonempty negative logit plus.2; empty positive clips count FN',
      provisional_labels=True,latest25_used_for_fit=False,latest25_used_for_selection=False))

def evaluate():
    import torch
    torch.set_num_threads(2);ckpt=torch.load(RUN/'candidate.pt',weights_only=True,map_location='cpu')
    if (REPORT/'latest25_results.json').exists():raise FileExistsError('Preserve completed evaluation output')
    provenance=read(REPORT/'latest25_features/provenance.json')
    if (ckpt['architecture']!='vit_cls_context_mil_v1' or ckpt['input_dim']!=2304 or ckpt['profile']!=PROFILE
      or ckpt['hidden'] not in (0,64) or ckpt['backbone_sha256']!=provenance['backbone_sha256']
      or ckpt['manifest_sha256']!=digest(RUN/'manifest.json') or provenance['profile']!=PROFILE
      or provenance['manifest_sha256']!=digest(RUN/'manifest.json')
      or ckpt['mean'].shape!=(2304,) or ckpt['scale'].shape!=(2304,)
      or not torch.isfinite(ckpt['mean']).all() or not torch.isfinite(ckpt['scale']).all()
      or torch.any(ckpt['scale']<=0) or not np.isfinite(ckpt['offset'])):raise ValueError('Incompatible feature profile')
    model=head(torch,ckpt['hidden']);model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    rows=read(ROOT/'reports/requested_revision_20261015/holdout_new/results.json')['results'];results=[]
    if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation bytes')
    fit_rows=read(RUN/'manifest.json')['records']
    if {r['sha256'] for r in rows}&{r['sha256'] for r in fit_rows} or {r['source_group'] for r in rows}&{r['source_group'] for r in fit_rows}:raise ValueError('Evaluation/training leakage')
    for row in rows:
      if digest(row['path'])!=row['sha256']:raise ValueError('Changed evaluation source')
      path=REPORT/'latest25_features'/(row['sha256']+'.npz')
      arrays=validate_cache(path,row['sha256'],ckpt['backbone_sha256'],ckpt['manifest_sha256'])
      bound_decode_audit(path,row['sha256'],arrays)
      x=torch.from_numpy(arrays['features']);stamps=arrays['seconds']
      with torch.inference_mode():scores=torch.sigmoid(model((x-ckpt['mean'])/ckpt['scale']).flatten()-ckpt['offset']).tolist() if len(x) else []
      if not np.isfinite(scores).all():raise ValueError('Nonfinite head predictions')
      results.append(dict(path=row['path'],label=row['label'],alert=any(s>=.5 for s in scores),windows=[dict(second=float(t),score=s) for t,s in zip(stamps,scores)]))
    metrics={k:sum(bool(r['label'])==actual and r['alert']==pred for r in results) for k,actual,pred in [('TP',True,True),('TN',False,False),('FP',False,True),('FN',True,False)]}
    save(REPORT/'latest25_results.json',dict(scope='Consumed development regression; no independent accuracy claim',metrics=metrics,rows=results,checkpoint_sha256=digest(RUN/'candidate.pt')))
    print(metrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.fit:fit()
    if a.evaluate:evaluate()
