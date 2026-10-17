"""Bounded causal video inference shared by cameras; no model imports or downloads.

Construct ActionWorker(context, learned_head) once, then ActionMonitor(worker, id)
per camera. feed(BGR, source_seconds, source_frame_id, scene_cut=bool) is a capture
callback: it samples and applies official per-frame preparation outside the queue
lock; R3D inference stays on the worker. poll(source_seconds) returns new evidence
or None. In synchronous=True mode poll also runs the same inference function;
this mode is for offline replay, never a live capture callback.

Prepared storage is bounded by (32 * max_cameras + 16) * max_prepared_frame_bytes,
including pending/inflight snapshots, plus one prepared frame per active feed.
Each feed may temporarily convert one bounded input frame. Optional raw-buffer
mode has the same bound using max_frame_bytes instead. One model workspace exists.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import math
import threading
import time

import numpy as np


ALERT_TEXT = 'Nghi có xô xát - cần người đến kiểm tra'


def _source(stamp, frame_id=None):
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float, np.number)) or not math.isfinite(stamp) or stamp < 0:
        raise ValueError('Expected a finite nonnegative source timestamp')
    if frame_id is not None and (isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0):
        raise ValueError('Expected a nonnegative integer source frame ID')


class ActionWorker:
    """One serial R3D worker, latest pending snapshot per registered camera.

    extractor supplies prepare_frame(rgb) and infer_samples(prepared, segment).
    head supplies score(embeddings[N,512]), as FrozenActionHead does. Neither is
    called under the queue lock. Caller owns these models until close returns.
    """
    def __init__(self, extractor, head, *, max_cameras=4, worker_interval=.5,
                 max_frame_bytes=4096*2160*3, synchronous=False,
                 prepare_in_feed=True, max_prepared_frame_bytes=3*112*112*4):
        if not isinstance(max_cameras, int) or max_cameras < 1 or not math.isfinite(worker_interval) or worker_interval < .5:
            raise ValueError('Require bounded cameras and a worker interval >= .5 seconds')
        if not isinstance(max_frame_bytes, int) or max_frame_bytes < 3:
            raise ValueError('Require a positive frame byte limit')
        if not isinstance(max_prepared_frame_bytes, int) or max_prepared_frame_bytes < 1:
            raise ValueError('Require a positive prepared-frame byte limit')
        self.extractor, self.head = extractor, head
        self.max_cameras, self.worker_interval = max_cameras, float(worker_interval)
        self.max_frame_bytes, self.synchronous = max_frame_bytes, bool(synchronous)
        self.prepare_in_feed = bool(prepare_in_feed)
        self.max_prepared_frame_bytes = max_prepared_frame_bytes
        self.sample_fps = getattr(extractor, 'sample_fps', 8)
        self.min_frames = getattr(extractor, 'min_frames', 8)
        self.min_span = getattr(extractor, 'min_span', .875)
        self.feature_dim = getattr(extractor, 'feature_dim', 512)
        self.window = getattr(extractor, 'window', 16)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (self.sample_fps, self.min_frames, self.window, self.feature_dim)):
            raise ValueError('Require integer causal action profile and feature dimensions')
        self.cls_context = (self.sample_fps == 4 and self.min_frames == 4
                            and self.min_span == .75 and self.feature_dim == 2304
                            and self.window == 4)
        if not self.cls_context and (self.sample_fps not in (8, 16) or not isinstance(self.min_frames, int)
                or not 8 <= self.min_frames <= 16 or not math.isfinite(self.min_span)
                or self.min_span < .875
                or self.feature_dim != 512 or self.window != 16
                or (self.sample_fps == 16 and (self.min_frames != 16 or self.min_span < .9375))):
            raise ValueError('Unsupported causal action sampling profile')
        self.sample_interval = 1./self.sample_fps
        self.condition = threading.Condition()
        self.monitors, self.pending = {}, OrderedDict()
        self.closed, self.last_started = False, -math.inf
        self.inflight = None
        self.thread = None
        if not self.synchronous:
            self.thread = threading.Thread(target=self._run, name='causal-action', daemon=True)
            self.thread.start()

    def _register(self, monitor):
        with self.condition:
            if self.closed:
                raise RuntimeError('Action worker is closed')
            if monitor.camera_id in self.monitors:
                raise ValueError('Camera already registered')
            if len(self.monitors) >= self.max_cameras:
                raise ValueError('Action worker camera limit reached')
            self.monitors[monitor.camera_id] = monitor

    def _take(self):
        monitor, snapshot = self.pending.popitem(last=False)
        self.inflight = monitor
        self.last_started = time.monotonic()
        return monitor, snapshot

    def _infer(self, snapshot, monitor=None):
        segment, rows, submitted = snapshot
        prepared = (list(rows) if self.prepare_in_feed else
                    [(stamp, frame_id, self.extractor.prepare_frame(frame[:, :, ::-1]))
                     for stamp, frame_id, frame in rows])
        raw = (self.extractor.infer_samples(prepared, segment, stream_key=monitor)
               if self.cls_context else self.extractor.infer_samples(prepared, segment))
        embedding = np.asarray(raw['embedding'])
        if embedding.shape != (self.feature_dim,) or not np.isfinite(embedding).all():
            raise ValueError('Malformed action embedding')
        scores = np.asarray(self.head.score(embedding[None]), dtype=float)
        if scores.shape != (1,) or not np.isfinite(scores).all() or not 0 <= scores[0] <= 1:
            raise ValueError('Malformed learned action probability')
        return {'score': float(scores[0]), 'segment': segment,
                'second': rows[-1][0], 'start_second': rows[0][0],
                'frame_ids': tuple(row[1] for row in rows),
                'frame_seconds': tuple(row[0] for row in rows),
                'real_frames': len(rows), 'span': rows[-1][0] - rows[0][0],
                'sample_fps': self.sample_fps, 'min_real_frames': self.min_frames,
                'min_span': self.min_span,
                '_submitted': submitted}

    def _execute(self, monitor, snapshot):
        try:
            result, error = self._infer(snapshot, monitor), None
        except Exception as exc:
            result, error = None, type(exc).__name__  # bounded status; no source paths/frames
        with self.condition:
            self.inflight = None
            if not self.closed and not monitor.closed and monitor.segment == snapshot[0]:
                monitor.error = error
                if result is not None:
                    monitor.result = result
            self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                while not self.closed:
                    delay = self.worker_interval - (time.monotonic() - self.last_started)
                    if self.pending and delay <= 0:
                        break
                    self.condition.wait(delay if self.pending else None)
                if self.closed:
                    return
                monitor, snapshot = self._take()
            self._execute(monitor, snapshot)

    def run_one(self):
        """Offline deterministic advancement. No artificial wall-time sleeps."""
        if not self.synchronous:
            raise RuntimeError('run_one is only available for synchronous replay')
        with self.condition:
            if self.closed or not self.pending:
                return False
            if self.inflight is not None:
                raise RuntimeError('Concurrent synchronous replay is forbidden')
            monitor, snapshot = self._take()
        self._execute(monitor, snapshot)
        return True

    def close(self, timeout=5.):
        """Stop accepting work, invalidate results and join the current inference."""
        with self.condition:
            self.closed = True
            self.pending.clear()
            for monitor in list(self.monitors.values()):
                monitor._reset_locked(clear_identity=True)
                monitor.closed = True
            self.monitors.clear()
            self.condition.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout)
            finished = not self.thread.is_alive()
        else:
            finished = self.inflight is None
        if finished and hasattr(self.extractor, 'clear_cache'):
            self.extractor.clear_cache()
        return finished


class ActionMonitor:
    """Per-camera real-frame history, stale-result checks and alert cooldown."""
    def __init__(self, worker, camera_id, *, cooldown=30., threshold=.5,
                 max_gap=None, max_result_age=1.5):
        if not isinstance(camera_id, str) or not camera_id or len(camera_id) > 256:
            raise ValueError('Expected a bounded camera ID')
        if max_gap is None:
            max_gap = getattr(worker.extractor, 'max_gap', .60 if worker.cls_context else .45)
        if (not all(math.isfinite(x) for x in (cooldown, threshold, max_gap, max_result_age))
                or cooldown < 0 or threshold != .5 or max_gap < .125 or max_result_age <= 0):
            raise ValueError('Invalid causal action decision settings; learned threshold is .5')
        self.worker, self.camera_id = worker, camera_id
        self.cooldown, self.threshold = float(cooldown), float(threshold)
        self.max_gap, self.max_result_age = float(max_gap), float(max_result_age)
        self.frames = deque(maxlen=worker.window)
        self.feed_lock = threading.Lock()  # serialize only this camera's frame preparation
        self.segment, self.closed = 0, False
        self.last_seen = self.last_id = self.next_sample = self.last_queued = None
        self.last_alert, self.result, self.error = -math.inf, None, None
        worker._register(self)

    def _reset_locked(self, clear_identity=False):
        self.segment += 1
        self.frames.clear()
        self.worker.pending.pop(self, None)
        self.result = self.error = self.next_sample = self.last_queued = None
        if clear_identity:
            self.last_seen = self.last_id = None

    def reset(self):
        """Explicit source restart; allows timestamps/IDs to restart at zero."""
        with self.worker.condition:
            self._reset_locked(clear_identity=True)
            self.last_alert = -math.inf

    def feed(self, frame_bgr, source_stamp, frame_id, scene_cut):
        """Return whether sampled. A real scene-cut decision is required per frame."""
        _source(source_stamp, frame_id)
        if not isinstance(scene_cut, (bool, np.bool_)):
            raise ValueError('An explicit scene-cut boolean is required')
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or min(frame_bgr.shape[:2]) < 1
                or frame_bgr.nbytes > self.worker.max_frame_bytes):
            raise ValueError('Expected a bounded actual BGR uint8 frame')
        stamp, frame_id = float(source_stamp), int(frame_id)
        with self.feed_lock:
            return self._feed(frame_bgr, stamp, frame_id, bool(scene_cut))

    def _feed(self, frame_bgr, stamp, frame_id, scene_cut):
        with self.worker.condition:
            if self.closed or self.worker.closed:
                raise RuntimeError('Action monitor is closed')
            if self.last_seen is not None:
                if stamp <= self.last_seen or frame_id <= self.last_id:
                    raise ValueError('Repeated or reordered source frames are forbidden')
                if scene_cut or stamp - self.last_seen > self.max_gap:
                    # A shot boundary clears evidence, not the source's sample
                    # grid. Re-anchoring to a rounded native-frame time would
                    # resample an already sampled capture stream a second time.
                    next_sample = self.next_sample
                    self._reset_locked()
                    self.next_sample = next_sample
            elif scene_cut:
                self._reset_locked()
            self.last_seen, self.last_id = stamp, frame_id
            if self.next_sample is not None and stamp + 1e-8 < self.next_sample:
                return False
            generation = self.segment
            if self.next_sample is None:
                self.next_sample = stamp + self.worker.sample_interval
            else:
                interval = self.worker.sample_interval
                self.next_sample += interval * (math.floor(max(0., stamp-self.next_sample)/interval + 1e-8) + 1)
        # Official per-frame transform is independent of temporal inference. It
        # owns its result; neither raw 4K frames nor their backing arrays persist.
        # This may run concurrently with reset/close, so recheck generation below.
        if self.worker.prepare_in_feed:
            owned = self.worker.extractor.prepare_frame(frame_bgr[:, :, ::-1])
            if isinstance(owned, np.ndarray):
                # Do not retain a small NumPy view backed by a full input frame.
                owned = np.array(owned, copy=True, order='C')
                owned.flags.writeable = False
            nbytes = getattr(owned, 'nbytes', None)
            if hasattr(owned, 'untyped_storage'):
                nbytes = owned.untyped_storage().nbytes()
            if not isinstance(nbytes, (int, np.integer)) or not 0 < nbytes <= self.worker.max_prepared_frame_bytes:
                raise ValueError('Prepared action frame exceeds its fixed memory bound')
        else:
            owned = np.array(frame_bgr, copy=True, order='C')
            owned.flags.writeable = False
        with self.worker.condition:
            if self.closed or self.worker.closed or self.segment != generation:
                return False
            self.frames.append((stamp, frame_id, owned))
            if (len(self.frames) < self.worker.min_frames
                    or stamp-self.frames[0][0] + 1e-8 < self.worker.min_span
                    or (self.last_queued is not None and stamp-self.last_queued + 1e-8 < .5)):
                return True
            self.last_queued = stamp
            # Replacing a camera's snapshot cannot grow the queue. OrderedDict
            # retains its position so a fast camera cannot starve another camera.
            self.worker.pending[self] = (self.segment, tuple(self.frames), time.monotonic())
            self.worker.condition.notify()
            return True

    def poll(self, current_stamp):
        """Consume new fresh evidence. fire is a human-review cue, never intent."""
        _source(current_stamp)
        if self.worker.synchronous:
            self.worker.run_one()
        with self.worker.condition:
            result, self.result = self.result, None
            if result is None or self.closed or self.worker.closed or result['segment'] != self.segment:
                return None
            age = float(current_stamp)-result['second']
            if age < -1e-8 or age > self.max_result_age:
                return None
            if not self.worker.synchronous and time.monotonic()-result['_submitted'] > self.max_result_age:
                return None
            result.pop('_submitted')
            qualifies = result['score'] >= self.threshold
            fire = qualifies and result['second']-self.last_alert >= self.cooldown
            if fire:
                self.last_alert = result['second']
            result.update(camera_id=self.camera_id, fire=fire, threshold=self.threshold,
                          reason='review' if fire else ('cooldown' if qualifies else 'below_threshold'),
                          message=ALERT_TEXT if qualifies else '')
            return result

    def close(self):
        with self.worker.condition:
            self._reset_locked(clear_identity=True)
            self.closed = True
            if self.worker.monitors.get(self.camera_id) is self:
                del self.worker.monitors[self.camera_id]
            self.worker.condition.notify_all()
