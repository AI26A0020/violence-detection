"""Bounded causal CLS context. Class semantics from the original head are unused."""
from collections import deque
import math
import numpy as np

PROFILE={'schema':'vit-cls-last-mean-absdelta-4real4fps-v3','sample_fps':4,
         'min_frames':4,'min_span':.75,'window':4,'score_interval':.5,'max_gap':.60,
         'frame_dim':768,'input_dim':2304,'preprocess':'RGB-PIL-bilinear224-mean.5-std.5',
         'norm_eps':1e-6,'clock':'audited-native-CFR-successful-index-over-fps-v1'}

class CausalCLSWindow:
    def __init__(self):
        self.frames=deque(maxlen=4);self.last_score=None;self.last_id=None;self.last_stamp=None;self.segment=0
    def reset(self):
        self.frames.clear();self.last_score=None;self.segment+=1
    def observe(self,feature,stamp,frame_id,scene_cut=False):
        x=np.asarray(feature,dtype=np.float32)
        if x.shape!=(768,) or not np.isfinite(x).all() or not math.isfinite(stamp) or stamp<0:
            raise ValueError('Expected finite CLS768 and source timestamp')
        if type(frame_id) is not int or frame_id<0 or type(scene_cut) is not bool:raise ValueError('Invalid observation identity')
        if self.last_stamp is not None and (stamp<=self.last_stamp or frame_id<=self.last_id):
            raise ValueError('Repeated or reordered source observation')
        if scene_cut:self.reset()
        if self.frames and stamp-self.frames[-1][0]>PROFILE['max_gap']:self.reset()
        self.frames.append((float(stamp),x.copy(),frame_id));self.last_id=frame_id;self.last_stamp=float(stamp)
        if len(self.frames)<4 or stamp-self.frames[0][0]+1e-6<.75:return None
        if self.last_score is not None and stamp-self.last_score+1e-6<.5:return None
        self.last_score=stamp
        seq=np.stack([r[1] for r in self.frames])
        feature=np.concatenate((seq[-1],seq.mean(0),np.abs(np.diff(seq,axis=0)).mean(0)))
        return {'feature':feature,'second':float(stamp),'start_second':self.frames[0][0],
                'frame_ids':[r[2] for r in self.frames],'frame_seconds':[r[0] for r in self.frames],
                'real_frames':4,'segment':self.segment}

def prepare_frame_rgb224(bgr):
    from PIL import Image
    if not isinstance(bgr,np.ndarray) or bgr.dtype!=np.uint8 or bgr.ndim!=3 or bgr.shape[2]!=3:
        raise ValueError('Expected uint8 BGR image')
    rgb=np.asarray(Image.fromarray(bgr[:,:,::-1]).resize((224,224),Image.Resampling.BILINEAR)).copy()
    return (rgb.transpose(2,0,1).astype(np.float32)/255.-.5)/.5

class FrozenViTCLS:
    def __init__(self):
        import torch
        from probe_vit_recall import load_model
        torch.set_num_threads(2)
        self.torch=torch;self.model,self.weights_sha256=load_model()
        self.model.heads=torch.nn.Identity();self.model.eval()
    def encode_frame(self,prepared):
        x=np.asarray(prepared,dtype=np.float32)
        if x.shape!=(3,224,224) or not np.isfinite(x).all():raise ValueError('Invalid prepared RGB224')
        with self.torch.inference_mode():
            result=self.model(self.torch.from_numpy(x).unsqueeze(0))[0].cpu().numpy()
        if result.shape!=(768,) or not np.isfinite(result).all():raise ValueError('Invalid CLS output')
        return result
