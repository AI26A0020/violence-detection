"""School schedules and bounded temporal safety signals. No student identification."""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone, timedelta
import copy
import math

TZ = timezone(timedelta(hours=7))
POLICY_DEFAULTS = {
    'mode': 'standard', 'default_mode': 'strict',
    'periods': [
        {'start':'06:45','end':'09:00','mode':'strict','patrol':False},
        {'start':'09:00','end':'09:20','mode':'anti_false','patrol':True},
        {'start':'09:20','end':'11:30','mode':'strict','patrol':False},
        {'start':'13:00','end':'15:00','mode':'strict','patrol':False},
        {'start':'15:00','end':'15:20','mode':'anti_false','patrol':True},
        {'start':'15:20','end':'23:59','mode':'strict','patrol':False}],
    'days':[0,1,2,3,4,5], 'anti_false_seconds':3.0,
    'gather_seconds':15.0, 'fall_seconds':5.0, 'occlusion_seconds':8.0,
    'patrol_times':['17:00'], 'patrol_enabled':False,
    'camera_health':True, 'health_seconds':5.0
}
MODE_LABELS = {'standard':'Theo cấu hình AI', 'strict':'Nghiêm ngặt', 'anti_false':'Chống báo nhầm', 'auto':'Theo lịch trường'}

def minute(value):
    if not isinstance(value,str) or len(value)!=5 or value[2]!=':':
        raise ValueError('Giờ cần có dạng HH:MM.')
    h,m = map(int,value.split(':'))
    if not 0<=h<=23 or not 0<=m<=59: raise ValueError('Giờ ngoài phạm vi.')
    return h*60+m

def validate_policy(cfg, sources=()):
    if cfg.get('mode') not in MODE_LABELS or cfg.get('default_mode') not in ('standard','strict','anti_false'):
        raise ValueError('Chọn chế độ giám sát hợp lệ.')
    if not isinstance(cfg.get('days'),list) or not all(type(x) is int and 0<=x<=6 for x in cfg['days']):
        raise ValueError('Ngày trong tuần cần nằm trong 0–6 (thứ Hai–Chủ nhật).')
    occupied=set()
    for p in cfg['periods']:
        start,end=minute(p['start']),minute(p['end'])
        if start==end or p['mode'] not in ('strict','anti_false','standard') or type(p.get('patrol',False)) is not bool:
            raise ValueError('Khoảng giờ/chế độ không hợp lệ.')
        span=set(range(start,end)) if start<end else set(range(start,1440))|set(range(end))
        if span & occupied: raise ValueError('Các khoảng giờ không được chồng nhau.')
        occupied |= span
    for value in cfg['patrol_times']: minute(value)
    for key,lo,hi in [('anti_false_seconds',1,15),('gather_seconds',5,120),('fall_seconds',3,30),('occlusion_seconds',1,15),('health_seconds',3,30)]:
        v=cfg.get(key)
        if type(v) not in (float,int) or not math.isfinite(v) or not lo<=v<=hi: raise ValueError('Tham số lịch không hợp lệ: '+key)
    for key in ('patrol_enabled','camera_health'):
        if type(cfg.get(key)) is not bool:raise ValueError(key+' cần true/false.')
    for spec in sources:
        for z in spec.get('hidden_zones',[]):
            if not isinstance(z,list) or len(z)!=4 or not all(type(x) in (int,float) and math.isfinite(x) for x in z) or not(0<=z[0]<z[2]<=1 and 0<=z[1]<z[3]<=1):
                raise ValueError('Vùng góc khuất cần [x1,y1,x2,y2] chuẩn hóa 0–1.')

def mode_at(cfg, when=None):
    when=when or datetime.now(TZ)
    if when.tzinfo:when=when.astimezone(TZ)
    if cfg['mode']!='auto':return cfg['mode']
    if when.weekday() not in cfg['days']: return 'standard'
    n=when.hour*60+when.minute
    for p in cfg['periods']:
        a,b=minute(p['start']),minute(p['end'])
        if (a<=n<b if a<b else n>=a or n<b):return p['mode']
    return cfg['default_mode']

def detection_at(base, cfg, when=None):
    result=copy.deepcopy(base);mode=mode_at(cfg,when)
    if mode=='anti_false':result['sustain_seconds']=max(result['sustain_seconds'],cfg['anti_false_seconds'])
    if mode=='strict':result['sustain_seconds']=min(result['sustain_seconds'],.6)
    return result,mode

def patrol_slot(cfg, now):
    now=now.astimezone(TZ) if now.tzinfo else now
    if not cfg['patrol_enabled'] or now.weekday() not in cfg['days']:return None
    times=set(cfg['patrol_times'])|{p['start'] for p in cfg['periods'] if p.get('patrol')}
    hhmm=now.strftime('%H:%M')
    return now.strftime('%Y-%m-%d')+' '+hhmm if hhmm in times else None


def body_posture(box):
    """Use visible torso AND legs; a wide, desk-cropped box is not a fallen body.

    COCO pose can extrapolate hidden joints beyond the detection. Such points,
    low-confidence points and an incomplete leg chain are unknown evidence.
    """
    bw,bh=box.x2-box.x1,box.y2-box.y1
    if bw<=0 or bh<=0 or box.keypoints is None:return None
    def joint(i):
        try:x,y,c=map(float,box.keypoints[i][:3])
        except (IndexError,TypeError,ValueError):return None
        if not all(math.isfinite(v) for v in (x,y,c)) or c<.5:return None
        if not (box.x1-.15*bw<=x<=box.x2+.15*bw and box.y1-.15*bh<=y<=box.y2+.15*bh):return None
        return (x,y)
    chains=[]
    for indices in ((5,11,13,15),(6,12,14,16)):
        points=[joint(i) for i in indices]
        if all(p is not None for p in points):chains.append(points)
    if not chains:return None
    # Average complete sides, avoiding a torso assembled from unrelated sides.
    shoulder,hip,knee,ankle=[tuple(sum(c[i][axis] for c in chains)/len(chains) for axis in (0,1)) for i in range(4)]
    torso=math.dist(shoulder,hip);leg=math.dist(hip,ankle)
    span=max(bw,bh)
    if torso<.10*span or leg<.15*span:return None
    dx,dy=hip[0]-shoulder[0],hip[1]-shoulder[1]
    xs=[p[0] for p in (shoulder,hip,knee,ankle)];ys=[p[1] for p in (shoulder,hip,knee,ankle)]
    upright=(dy>1.25*abs(dx) and ankle[1]-hip[1]>.65*leg and knee[1]>hip[1] and ankle[1]>knee[1])
    prone=(abs(dx)>1.25*abs(dy) and max(xs)-min(xs)>1.35*(max(ys)-min(ys))
           and abs(ankle[1]-hip[1])<.60*leg and abs(ankle[0]-shoulder[0])>torso)
    return {'kind':'upright' if upright else 'prone' if prone else 'other',
            'shoulder':shoulder,'hip':hip,'ankle':ankle,'size':math.dist(shoulder,ankle)}


class ScenarioMonitor:
    def __init__(self):self.reset()
    def reset(self):
        self.last=None;self.groups={};self.poses={};self.falls={};self.last_fire={}
        self.conflict=None

    def _update_falls(self,t,seen,cfg,conflict,signal):
        active=set()
        observed_ids={b.track_id for b in seen}
        for ident,fall in self.falls.items():
            if ident not in observed_ids:fall['consecutive']=False
        for b in seen:
            ident=b.track_id;pose=body_posture(b);fall=self.falls.get(ident)
            prior=self.poses.get(ident)
            if pose and pose['kind']=='upright':
                # At least three consistent, fresh standing observations establish
                # a floor reference. One bad box/ID assignment cannot establish it.
                continuous=(prior is not None and 0<t-prior['time']<=1.05
                            and math.dist(pose['hip'],prior['hip'])<.6*prior['size']
                            and .6*prior['size']<pose['size']<1.6*prior['size'])
                pose.update(time=t,since=prior['since'] if continuous else t,
                            count=prior['count']+1 if continuous else 1)
                self.poses[ident]=pose;self.falls.pop(ident,None)
                continue
            if pose is None or pose['kind']!='prone':
                if fall:fall['consecutive']=False
                continue
            if fall is None:
                if not prior or prior['count']<3 or prior['time']-prior['since']<.4-1e-6:continue
                size=prior['size']
                transition=(0<t-prior['time']<=2.5 and size>0
                            and pose['shoulder'][1]-prior['shoulder'][1]>.20*size
                            and pose['shoulder'][1]>=prior['ankle'][1]-.35*size
                            and math.dist(pose['hip'],prior['hip'])<1.25*size
                            and .4*size<pose['size']<1.7*size)
                if not transition:continue
                fall={'last':t,'anchor':pose['hip'],'size':size,'seconds':0.,'count':1,'consecutive':True}
                self.falls[ident]=fall
            else:
                dt=t-fall['last']
                if not 0<dt<=1.05 or math.dist(pose['hip'],fall['anchor'])>.35*fall['size']:
                    self.falls.pop(ident,None);self.poses.pop(ident,None)
                    continue
                # Missing/uncertain pose time never contributes to the dwell time.
                if fall['consecutive']:fall['seconds']+=dt
                fall.update(last=t,count=fall['count']+1,consecutive=True)
            active.add(ident)
            if fall['seconds']+1e-6<cfg['fall_seconds'] or fall['count']<6:continue
            neighbors=[p for p in seen if p.track_id!=ident and math.dist(p.center,b.center)<2*fall['size']]
            verified_conflict=(conflict and (b.is_fighting or any(p.is_fighting and p.fight_partner_id==ident for p in neighbors)))
            if self.conflict:
                stamp,ids,_,_=self.conflict
                verified_conflict=verified_conflict or (ident in ids and 0<=t-stamp<=cfg['occlusion_seconds'])
            signal('Người nằm sàn kéo dài · cần hỗ trợ khẩn' if verified_conflict else 'Người ngã chưa đứng dậy · cần kiểm tra',
                   f'Có chuyển từ tư thế đứng sang nằm, được xác nhận bằng thân và chân qua nhiều quan sát; chưa hồi phục sau {cfg["fall_seconds"]:g} giây. Cần người lớn kiểm tra nguyên nhân.',
                   [ident]+[p.track_id for p in neighbors])
        # A brief pose miss is tolerated, without adding unseen time. Recovery,
        # long occlusion, a backward timestamp or a changed ID cannot keep a timer.
        self.poses={i:v for i,v in self.poses.items() if 0<=t-v['time']<=2.5}
        self.falls={i:v for i,v in self.falls.items() if i in active or 0<=t-v['last']<=.6}

    def update(self,t,boxes,shape,cfg,mode,zones=(),conflict=False):
        if self.last is not None and (t<self.last or t-self.last>1.5):self.reset()
        self.last=t
        seen=[b for b in boxes if b.observed]
        byid={b.track_id:b for b in seen};h,w=shape[:2]
        signals=[]
        def signal(kind,reason,ids):
            if t-self.last_fire.get(kind,-math.inf)>=30:
                self.last_fire[kind]=t;signals.append({'kind':kind,'reason':reason,'ids':list(ids)})
        # Group timers require the same actual IDs and stable geometry in a configured hidden zone.
        active=set()
        for zi,z in enumerate(zones):
            members=[b for b in seen if z[0]*w<=b.center[0]<=z[2]*w and z[1]*h<=b.center[1]<=z[3]*h]
            if len(members)<(2 if mode=='strict' else 3):continue
            span=max(max(b.x2-b.x1,b.y2-b.y1) for b in members)
            if any(math.dist(a.center,b.center)>2.0*span for a in members for b in members):continue
            key=(zi,tuple(sorted(b.track_id for b in members)));active.add(key)
            centers={b.track_id:b.center for b in members}
            old=self.groups.get(key)
            if old is None or any(math.dist(centers[i],old[1][i])>.25*span for i in centers):
                self.groups[key]=(t,centers);continue
            near_wall=any(min(b.center[0]-z[0]*w,z[2]*w-b.center[0],b.center[1]-z[1]*h,z[3]*h-b.center[1])<.3*span for b in members)
            if t-old[0]>=cfg['gather_seconds']:
                if len(members)>=3 and near_wall:
                    signal('Tụ tập kéo dài tại góc khuất · cần kiểm tra',f'Nhóm đứng sát ổn định trên {cfg["gather_seconds"]:g} giây tại vùng đã chọn, có người sát mép vùng. Chưa xác định nguyên nhân; cần người lớn kiểm tra.',centers)
                elif mode=='strict':
                    signal('Tụ tập kéo dài tại góc khuất','Chế độ nghiêm ngặt: từ hai người tụ tập lâu trong vùng trường đã chọn.',centers)
        self.groups={k:v for k,v in self.groups.items() if k in active}
        self._update_falls(t,seen,cfg,conflict,signal)
        # Overlapping detections and motion do not establish physical contact.
        # Every fight alert, including strict mode, must pass the shared temporal
        # InteractionVerifier and AlertGate. Do not bypass them with a box timer.
        if conflict:
            involved=[b for b in seen if b.is_fighting]
            if len(involved)>=2:self.conflict=(t,tuple(b.track_id for b in involved),(sum(b.center[0] for b in involved)/len(involved),sum(b.center[1] for b in involved)/len(involved)),max(b.y2-b.y1 for b in involved))
        hold=False
        if self.conflict:
            stamp,ids,center,size=self.conflict
            vanished=any(i not in byid for i in ids)
            onlookers=sum(math.dist(b.center,center)<2.5*size for b in seen if b.track_id not in ids)
            hold=vanished and onlookers>=4 and 0<t-stamp<=cfg['occlusion_seconds']
            if t-stamp>cfg['occlusion_seconds']:self.conflict=None
        return {'signals':signals,'occlusion_hold':hold,'mode':mode}

class CameraHealth:
    def __init__(self):
        self.since=None;self.reason=None;self.alerted=False;self.baseline=None;self.last=None;self.still_since=None
    def update(self,t,frame,persist=5):
        import cv2
        import numpy as np
        gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(96,64))
        mean,std=float(gray.mean()),float(gray.std())
        sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
        color=cv2.resize(frame,(96,64))//32
        _,counts=np.unique(color.reshape(-1,3),axis=0,return_counts=True)
        uniform=float(counts.max()/gray.size)
        frozen=self.last is not None and np.max(np.abs(gray.astype(float)-self.last.astype(float)))<.5
        if frozen:self.still_since=t if self.still_since is None else self.still_since
        else:self.still_since=None
        self.last=gray
        reason=('Hình quá tối' if mean<12 and std<12 else
                'Mảng màu đồng nhất che phần lớn hình (>70%)' if uniform>.70 and std<25 else
                'Hình mờ giảm mạnh so với nền trước đó' if self.baseline and sharp<max(3,self.baseline*.10) and std<35 else
                'Hình không thay đổi; cần kiểm tra camera có bị đứng' if self.still_since is not None and t-self.still_since>=10 else None)
        if reason is None:
            self.baseline=sharp if self.baseline is None else .98*self.baseline+.02*sharp
            self.since=None;self.reason=None;self.alerted=False
        elif reason!=self.reason:self.since=t;self.reason=reason
        fire=reason is not None and self.since is not None and t-self.since>=persist and not self.alerted
        if fire:self.alerted=True
        return {'fire':fire,'reason':reason,'mean':mean,'std':std,'sharpness':sharp,'uniform_fraction':uniform}

def patrol_grid(workers, now_mono):
    """Use the protected analyzed image; mosaic entire overview, mark stale/paused sources."""
    import cv2
    import numpy as np
    tiles=[];details=[]
    for worker in workers:
        view=worker.snapshot_view()
        fresh=(view.frame is not None and view.source_stamp>0 and 0<=now_mono-view.source_stamp<=3 and not worker.pause.is_set())
        tile=np.zeros((240,320,3),np.uint8)
        if fresh:
            small=cv2.resize(view.frame,(64,40))
            tile[30:230]=cv2.resize(small,(320,200),interpolation=cv2.INTER_NEAREST)
        label=f"{worker.spec['label']} | {'LIVE' if fresh else 'STALE / PAUSED'}"
        # ASCII fallback in the actual mosaic, full Vietnamese names included in caption.
        cv2.putText(tile,label.encode('ascii','replace').decode(),(5,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,220,30) if fresh else (0,50,255),1)
        tiles.append(tile);details.append({'camera':worker.spec['label'],'fresh':fresh})
    if not tiles:return None,details
    cols=2 if len(tiles)>1 else 1
    while len(tiles)%cols:tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]),details
