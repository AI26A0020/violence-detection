"""Development-only cached ViT probe. No runtime configuration is changed."""
import json, time, hashlib, argparse
from pathlib import Path
from functools import partial
import torch
from torch import nn
import cv2
import numpy as np
from torchvision.models import vit_b_16
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/recall_20261016/vit'

def load_model():
    path=ROOT/'models/vit-base-violence-detection/pytorch_model.bin'
    weights=torch.load(path,map_location='cpu',weights_only=True)
    model=vit_b_16(num_classes=2,norm_layer=partial(nn.LayerNorm,eps=1e-6))
    mapped={}
    fixed={'cls_token':'class_token','pos_embed':'encoder.pos_embedding',
           'patch_embed.proj':'conv_proj','norm':'encoder.ln','head':'heads.head'}
    block={'norm1':'ln_1','norm2':'ln_2','attn.qkv':'self_attention.in_proj',
           'attn.proj':'self_attention.out_proj','mlp.fc1':'mlp.0','mlp.fc2':'mlp.3'}
    for key,value in weights.items():
        if key.startswith('blocks.'):
            _,index,tail=key.split('.',2)
            name,param=tail.rsplit('.',1)
            dest=f'encoder.layers.encoder_layer_{index}.{block[name]}'
            dest+=('_'+param if name=='attn.qkv' else '.'+param)
        else:
            name,sep,param=key.rpartition('.')
            dest=fixed[name]+'.'+param if sep else fixed[key]
        mapped[dest]=value
    model.load_state_dict(mapped,strict=True)
    return model.eval(),hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--latest-only',action='store_true')
    parser.add_argument('--limit',type=int,default=0,help='Bounded startup benchmark; zero means all selected clips')
    args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    model,digest=load_model()
    latest=json.loads((ROOT/'reports/requested_revision_20261015/holdout_new/results.json').read_text())['results']
    old=json.loads((ROOT/'reports/precision_20261015/temporal_development_assessment.json').read_text())['baseline_rows']
    cases=[dict(c,origin='latest25_development') for c in latest]
    if not args.latest_only:cases += [dict(c,origin='old61_negative_development') for c in old if not c['label']]
    if args.limit:cases=cases[:args.limit]
    provenance={'checkpoint_sha256':digest,'weights_format':'timm ViT base patch16; every key mapped strictly to torchvision',
      'class_mapping':'UNKNOWN: supplied config and README do not declare class order; both class probabilities retained',
      'preprocessing':'ASSUMPTION: PIL bilinear RGB resize224; normalize mean/std .5 (standard HF ViT)',
      'architecture_assumption':'timm default LayerNorm epsilon1e-6; supplied HF config epsilon1e-12 conflicts with actual timm-format weights',
      'scope':'development only; latest25 previously exposed; labels provisional; no production conclusions',
      'sampling':'native decoder observations nearest next .25s target; no padded/repeated frames; no frame cap',
      'fps':4,'threads':2}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    results=[]
    for number,case in enumerate(cases):
        path=Path(case['path']);key=hashlib.sha256(str(path).encode()).hexdigest()[:16]
        dest=OUT/(key+'.json')
        if dest.exists():
            row=json.loads(dest.read_text());results.append(row);print(number,path.name,'cached',flush=True);continue
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);fps=fps if 1<=fps<=120 else 25
        index=0;next_time=0.;previous=-1.;observations=[];started=time.monotonic()
        with torch.inference_mode():
            while True:
                ok,frame=cap.read()
                if not ok:break
                stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if not np.isfinite(stamp) or stamp<=previous:stamp=max(index/fps,previous+1/fps)
                previous=stamp;index+=1
                if stamp+1e-6<next_time:continue
                while next_time<=stamp+1e-6:next_time+=.25
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                array=np.asarray(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)).copy()
                tensor=torch.from_numpy(array).permute(2,0,1).float().div_(255).sub_(.5).div_(.5)
                scores=model(tensor.unsqueeze(0)).softmax(1)[0].tolist()
                observations.append({'second':round(stamp,6),'class0':scores[0],'class1':scores[1]})
        cap.release()
        summary={}
        for cls in ('class0','class1'):
            values=np.array([r[cls] for r in observations])
            summary[cls]={str(q):float(np.quantile(values,q)) for q in (0,.25,.5,.75,.9,.95,1)} if len(values) else {}
        row={'path':str(path),'label':case['label'],'origin':case['origin'],'observations':observations,'summary':summary,'elapsed':time.monotonic()-started}
        dest.write_text(json.dumps(row,indent=2),encoding='utf8');results.append(row)
        (OUT/'results.json').write_text(json.dumps({'provenance':provenance,'cases':results},indent=2),encoding='utf8')
        print(number+1,len(cases),path.name,len(observations),summary,flush=True)

if __name__=='__main__':run()
