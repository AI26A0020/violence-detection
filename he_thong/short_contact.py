"""Causal raised-kick evidence from observed pose, without identities or labels."""
import numpy as np


class RaisedKickMonitor:
    """A directed raised foot must recoil on a later, distinct observation.

    This supplements repeated-contact verification for brief kicks. It never
    skips the application's score/sustain gate. Evidence expires after 650 ms;
    missing people, display-only tracks, scene resets or gaps cannot confirm it.
    """
    def __init__(self, joint):
        self.joint = joint
        self.previous, self.pending, self.holds = {}, {}, {}
        self.last_time = None

    def update(self, t, boxes, visual, threshold):
        if self.last_time is not None and (t <= self.last_time or t-self.last_time > .65):
            self.previous.clear(); self.pending.clear(); self.holds.clear()
        self.last_time = t
        fresh = {b.track_id: b for b in boxes
                 if b.observed and not b.display_tracked and b.height >= 25}
        current = {}
        for ident, a in fresh.items():
            pose = {j: p for j in (11, 12, 13, 14, 15, 16)
                    if (p := self.joint(a, j)) is not None}
            current[ident] = (t, pose, a.height)
            previous = self.previous.get(ident)
            if previous is None or not .04 <= t-previous[0] <= .45:
                continue
            dt, old = t-previous[0], previous[1]
            if not all(j in pose and j in old for j in (11, 12)):
                continue
            hip = np.mean([pose[j] for j in (11, 12)], axis=0)
            old_hip = np.mean([old[j] for j in (11, 12)], axis=0)
            scale = max(30, (a.height+previous[2])/2)
            for ankle, other, knee, other_knee in ((15, 16, 13, 14), (16, 15, 14, 13)):
                if not all(j in pose for j in (ankle, other, knee, other_knee)) or ankle not in old:
                    continue
                point = pose[ankle]
                relative = (point-hip)/scale
                delta = ((point-hip)-(old[ankle]-old_hip))/scale
                speed = float(np.linalg.norm(delta))/dt
                for target_id, b in fresh.items():
                    if target_id == ident:
                        continue
                    key = (ident, target_id, ankle)
                    pending = self.pending.get(key)
                    if pending and .04 <= t-pending['t'] <= .45:
                        recoil = float(np.dot(delta, pending['delta'])) < -.008
                        apart = np.linalg.norm(relative-pending['relative']) >= .12
                        if (recoil and apart and speed >= .85
                                and visual is not None and visual >= threshold
                                and b.local_motion >= .08
                                and np.linalg.norm(np.asarray(b.center)-pending['target_center']) <= .4*max(30, b.height)):
                            self.holds[(ident, target_id)] = {
                                'until': t+.65, 'start': pending['t'], 'times': [pending['t'], t]}
                    raised = (point[1] < pose[other][1]-.16*scale
                              and pose[knee][1] < pose[other_knee][1]-.08*scale)
                    directed = (point[0]-hip[0])*(b.center[0]-hip[0]) > .025*scale**2
                    near = (b.x1-.10*b.height <= point[0] <= b.x2+.10*b.height
                            and b.y1 <= point[1] <= b.y1+.78*b.height)
                    if (raised and directed and near and speed >= 1.1
                            and a.local_motion >= .08
                            and visual is not None and visual >= threshold):
                        self.pending[key] = {'t': t, 'delta': delta, 'relative': relative,
                                             'target_center': np.asarray(b.center)}
        self.previous = current
        self.pending = {k: v for k, v in self.pending.items()
                        if t-v['t'] <= .45 and k[0] in fresh and k[1] in fresh}
        self.holds = {k: v for k, v in self.holds.items()
                      if t <= v['until'] and k[0] in fresh and k[1] in fresh}
        return [dict(tracks=list(key), kind='chân nâng hướng vào người rồi thu về',
                     duration=hold['times'][-1]-hold['times'][0],
                     action_times=hold['times'], action_samples=2,
                     confirmed_since=hold['start']) for key, hold in self.holds.items()]


class RaisedPunchMonitor:
    """Directed raised wrist followed by an observed recoil of the same wrist.

    Shoulder-relative motion rejects camera/body translation. A slower recoil
    needs uninterrupted near-contact observations. Perspective scale, target
    continuity, real timestamps, the image score and AlertGate still apply.
    """
    def __init__(self, joint):
        self.joint = joint
        self.previous, self.pending, self.holds = {}, {}, {}
        self.last_time = None

    def update(self, t, boxes, visual, threshold):
        if self.last_time is not None and (t <= self.last_time or t-self.last_time > .65):
            self.previous.clear(); self.pending.clear(); self.holds.clear()
        self.last_time = t
        fresh = {b.track_id: b for b in boxes
                 if b.observed and not b.display_tracked and b.height >= 25}
        current = {}
        for ident, a in fresh.items():
            pose = {j: p for j in (5, 6, 9, 10, 11, 12)
                    if (p := self.joint(a, j)) is not None}
            current[ident] = (t, pose, a.height)
            previous = self.previous.get(ident)
            if (previous is None or not .04 <= t-previous[0] <= .45
                    or not all(j in pose and j in previous[1] for j in (5, 6))):
                continue
            dt, old = t-previous[0], previous[1]
            anchor = np.mean([pose[5], pose[6]], axis=0)
            prior = np.mean([old[5], old[6]], axis=0)
            scale = max(30, (a.height+previous[2])/2)
            for wrist in (9, 10):
                if wrist not in pose or wrist not in old:
                    continue
                point = pose[wrist]
                delta = ((point-anchor)-(old[wrist]-prior))/scale
                speed = float(np.linalg.norm(delta))/dt
                for target_id, b in fresh.items():
                    if ident == target_id or min(a.height, b.height)/max(a.height, b.height) < .35:
                        continue
                    key = (ident, target_id, wrist)
                    pending = self.pending.get(key)
                    if (pending and .04 <= t-pending['t'] <= .9
                            and (t-pending['t'] <= .45 or t-pending['last_near'] <= .25)):
                        recoil = float(np.dot(delta, pending['delta'])) < -.004
                        stable = np.linalg.norm(np.asarray(b.center)-pending['target_center']) <= .4*max(30, b.height)
                        if (recoil and stable and speed >= .7 and b.local_motion >= .06
                                and visual is not None and visual >= threshold):
                            self.holds[(ident, target_id)] = {
                                'start': pending['t'], 'until': t+.65, 'times': [pending['t'], t]}
                    hips = [pose[j] for j in (11, 12) if j in pose]
                    hip_y = float(np.mean(hips, axis=0)[1]) if hips else a.y1+.65*a.height
                    toward = float(np.dot(delta, (np.asarray(b.center)-anchor)/scale)) > .012
                    near = (b.x1-.06*b.height <= point[0] <= b.x2+.06*b.height
                            and b.y1 <= point[1] <= b.y1+.65*b.height)
                    if pending and near:
                        pending['last_near'] = t
                    if (near and toward and point[1] < hip_y-.02*scale and speed >= .85
                            and a.local_motion >= .06 and visual is not None and visual >= threshold):
                        self.pending[key] = dict(t=t, delta=delta, last_near=t,
                                                 target_center=np.asarray(b.center))
        self.previous = current
        self.pending = {k: v for k, v in self.pending.items()
                        if t-v['t'] <= .9 and k[0] in fresh and k[1] in fresh}
        self.holds = {k: v for k, v in self.holds.items()
                      if t <= v['until'] and k[0] in fresh and k[1] in fresh}
        return [dict(tracks=list(key), kind='tay nâng hướng vào người rồi thu về',
                     duration=hold['times'][-1]-hold['times'][0], action_times=hold['times'],
                     action_samples=2, confirmed_since=hold['start'])
                for key, hold in self.holds.items()]
