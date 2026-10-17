"""Capture-rate, bounded context buffer and background 3s H.264 clips."""
from collections import deque
from pathlib import Path
import logging
import threading
import queue
import time


class QuickClip:
    def __init__(self, folder, completed, privacy=lambda:False, enabled=True, min_free_mb=250):
        self.folder=Path(folder);self.completed=completed;self.privacy=privacy;self.enabled=enabled;self.min_free_mb=min_free_mb
        self.lock=threading.Lock();self.frames=deque(maxlen=60);self.pending={};self.last=-1e9
        self.protected=bool(self.privacy())
        self.jobs=queue.Queue(maxsize=4);self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True,name='quick-evidence')
        self.thread.start()

    def feed(self,t,frame):
        if not self.enabled or t-self.last<.095:return
        import cv2
        self.last=t
        h,w=frame.shape[:2];scale=min(1,640/w)
        img=cv2.resize(frame,(max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)))
        protected=bool(self.privacy())
        with self.lock:
            self._privacy_epoch(protected)
        if protected:
            # No stale AI boxes are used in capture thread; protect the whole overview.
            img=cv2.resize(cv2.resize(img,(32,18)),img.shape[1::-1],interpolation=cv2.INTER_NEAREST)
        ok,jpg=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,75])
        if not ok:return
        raw=jpg.tobytes()
        with self.lock:
            if protected!=bool(self.privacy()):return
            self.frames.append((t,raw))
            for eid,job in list(self.pending.items()):
                if t<=job['stamp']+1.5:job['frames'].append((t,raw))
                if t>=job['stamp']+1.5:
                    self._queue(eid,self.pending.pop(eid))

    def trigger(self,eid,stamp):
        if not self.enabled:return False
        with self.lock:
            self._privacy_epoch(bool(self.privacy()))
            if len(self.pending)>=4:return False
            frames=[(t,raw) for t,raw in self.frames if stamp-1.5<=t<=stamp+1.5]
            self.pending[eid]={'stamp':stamp,'frames':frames}
            if self.last>=stamp+1.5:self._queue(eid,self.pending.pop(eid))
        return True

    def _privacy_epoch(self,protected):
        # A newly protected event must never inherit pre-buffered unprotected pictures.
        if protected!=self.protected:
            self.frames.clear();self.pending.clear();self.protected=protected

    def _queue(self,eid,job):
        try:self.jobs.put_nowait((eid,job))
        except queue.Full:logging.getLogger('schoolguard').warning('Hàng đợi clip ngắn đầy: %s',eid)

    def reset(self):
        with self.lock:
            for eid,job in list(self.pending.items()):self._queue(eid,job)
            self.pending.clear();self.frames.clear();self.last=-1e9

    def run(self):
        while not self.stop.is_set() or not self.jobs.empty():
            try:eid,job=self.jobs.get(timeout=.1)
            except queue.Empty:continue
            try:
                frames=sorted(dict(job['frames']).items())
                if frames:
                    target,meta=self.encode(eid,job['stamp'],frames)
                    self.completed(eid,target,meta)
            except Exception:
                logging.getLogger('schoolguard').exception('Không lưu/gửi được clip ngắn %s',eid)
            finally:self.jobs.task_done()

    def encode(self,eid,stamp,frames):
        import cv2
        import numpy as np
        import av
        import shutil
        if shutil.disk_usage(self.folder).free<self.min_free_mb*1024**2:raise OSError('Không đủ dung lượng clip ngắn.')
        target=self.folder/'quick_clips'/(eid+'.mp4');target.parent.mkdir(exist_ok=True)
        first=cv2.imdecode(np.frombuffer(frames[0][1],np.uint8),cv2.IMREAD_COLOR)
        size=first.shape[1::-1]
        with av.open(str(target),'w') as output:
            stream=output.add_stream('libx264',rate=10)
            stream.width,stream.height=size;stream.pix_fmt='yuv420p';stream.options={'crf':'27','preset':'veryfast'}
            for i in range(30):
                when=stamp-1.5+i/10
                _,raw=min(frames,key=lambda row:abs(row[0]-when))
                img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                image=av.VideoFrame.from_ndarray(cv2.resize(img,size),format='bgr24')
                for packet in stream.encode(image):output.mux(packet)
            for packet in stream.encode():output.mux(packet)
        return target,{'seconds':3,'fps':10,'encoded_frames':30,'captured_frames':len(frames),
                       'pre_captured':sum(t<stamp for t,_ in frames),'post_captured':sum(t>=stamp for t,_ in frames),
                       'coverage_seconds':round(frames[-1][0]-frames[0][0],3),
                       'note':'Resampled capture timestamps; thiếu hình ở đầu/cuối nguồn được lặp, không phải bằng chứng mới.'}

    def close(self):
        self.reset();self.stop.set();self.thread.join()
