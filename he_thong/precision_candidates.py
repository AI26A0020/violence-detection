"""Bounded, offline experiments; importing this file never activates a candidate."""
from collections import deque
import ast
import math
from pathlib import Path


def stable_pose_verifier(app, common_scale=True):
    class StablePoseVerifier(app.InteractionVerifier):
        @staticmethod
        def joint(box, index):
            points = box.keypoints
            if points is None or len(points) <= index or len(points[index]) < 3:
                return None
            point = app.np.asarray(points[index][:3], dtype=float)
            if not app.np.isfinite(point).all() or point[2] < .35:
                return None
            return point[:2]

        def observe(self, box, t):
            history = self.history.setdefault(box.track_id, deque(maxlen=16))
            history.append((t, app.np.asarray(box.center), max(30.0, box.height)))
            while history and t-history[0][0] > 2.4:
                history.popleft()
            walking = False
            if len(history) >= 4 and history[-1][0]-history[0][0] >= .6:
                positions = app.np.asarray([r[1] for r in history])
                lengths = app.np.linalg.norm(app.np.diff(positions, axis=0), axis=1)
                path = float(lengths.sum())
                net = float(app.np.linalg.norm(positions[-1]-positions[0]))
                scale = float(app.np.mean([r[2] for r in history]))
                duration = history[-1][0]-history[0][0]
                walking = path > .12*scale and net/max(1e-6, path) > .90 and net/scale/duration > .10
            current = {j: p for j in (5, 6, 9, 10, 11, 12, 15, 16)
                       if (p := self.joint(box, j)) is not None}
            previous = self.pose_history.get(box.track_id)
            speeds = {}
            if previous and .04 <= t-previous[0] <= .65:
                shared = [j for j in (11, 12) if j in current and j in previous[1]]
                if not shared:
                    shared = [j for j in (5, 6) if j in current and j in previous[1]]
                if shared:
                    anchor = app.np.mean([current[j] for j in shared], axis=0)
                    old_anchor = app.np.mean([previous[1][j] for j in shared], axis=0)
                    scale = max(30.0, (box.height+previous[2])/2)
                    current_scale = scale if common_scale else max(30., box.height)
                    old_scale = scale if common_scale else max(30., previous[2])
                    speeds = {j: min(12.0, float(app.np.linalg.norm((p-anchor)/current_scale
                                                -(previous[1][j]-old_anchor)/old_scale)/(t-previous[0])))
                              for j, p in current.items() if j in (9, 10, 15, 16) and j in previous[1]}
            self.pose_history[box.track_id] = (t, current, box.height)
            return walking, speeds
    return StablePoseVerifier


def bounded_tracker(app):
    """Experimental factory; the selected production change is a literal source edit."""
    path = Path(getattr(app, '__precision_source__', app.__file__))
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SimpleTracker')
    source = ast.get_source_segment(text, definition)
    old = 'elif dist <= max(260.0, diag * 2.2):'
    new = 'elif dist <= max(60.0, diag * .9):'
    if new in source:
        return app.SimpleTracker
    if source.count(old) != 1:
        raise ValueError('Unsupported tracker version for this offline experiment')
    source = source.replace('class SimpleTracker:', 'class BoundedTracker:', 1).replace(old, new, 1)
    namespace = dict(app.__dict__)
    exec(compile(source, '<bounded-tracker-experiment>', 'exec'), namespace)
    return namespace['BoundedTracker']


def recovery_people_engine(app):
    class RecoveryPeopleEngine(app.PeopleEngine):
        """One real crop at most every third observation, before one tracker update."""
        def parse_boxes(self, results, offset=(0, 0)):
            confidence = self.cfg['confidence']
            boxes = []
            for result in results:
                points = result.keypoints.data.cpu().numpy() if getattr(result, 'keypoints', None) is not None else None
                for i, box in enumerate(result.boxes):
                    xy = list(map(float, box.xyxy[0].tolist()))
                    score = float(box.conf[0])
                    pose = points[i].copy() if points is not None and i < len(points) else None
                    if pose is not None:
                        pose[:, :2] += offset
                    xy[0] += offset[0]; xy[2] += offset[0]
                    xy[1] += offset[1]; xy[3] += offset[1]
                    if score < confidence:
                        visible = [p[:2] for p in pose[5:17] if p[2] >= .35 and app.np.isfinite(p).all()] if pose is not None else []
                        if len(visible) < 3:
                            continue
                        if float(app.np.ptp(app.np.asarray(visible), axis=0).max()) < max(8, .12*(xy[3]-xy[1])):
                            continue
                    boxes.append(app.PersonBox(*xy, confidence=score, keypoints=pose))
            return boxes

        @staticmethod
        def crop_for(boxes, shape):
            proposals = []
            for a in boxes:
                weak_head = a.keypoints is not None and sum(p[2] >= .35 for p in a.keypoints[:5]) < 2
                if a.confidence >= .5 and not weak_head:
                    continue
                group = [a]
                for b in boxes:
                    if b is a:
                        continue
                    scale = max(30, (a.height+b.height)/2)
                    if math.dist(a.center, b.center) < 1.25*scale:
                        group.append(b)
                height = max(b.y2 for b in group)-min(b.y1 for b in group)
                x1 = max(0, int(min(b.x1 for b in group)-.35*height))
                x2 = min(shape[1], int(max(b.x2 for b in group)+.35*height)+1)
                y1 = max(0, int(min(b.y1 for b in group)-.12*height))
                y2 = min(shape[0], int(max(b.y2 for b in group)+.12*height)+1)
                if min(x2-x1, y2-y1) >= 48 and (x2-x1)*(y2-y1) < .65*shape[0]*shape[1]:
                    proposals.append((a.confidence, (x1, y1, x2, y2)))
            return min(proposals, default=(0, None), key=lambda p: p[0])[1]

        def detect(self, frame, tracker):
            if self.model is None:
                return []
            tick = getattr(tracker, '_recovery_tick', 0)
            tracker._recovery_tick = tick+1
            confidence = self.cfg['confidence']
            minimum = min(confidence, self.cfg.get('min_body_confidence', .10)) if self.cfg.get('body_occlusion', True) else confidence
            with self.lock:
                warmed_threads = getattr(self, '_warm_start_threads', None)
                if warmed_threads is not None:
                    import torch
                    torch.set_num_threads(warmed_threads)
                    self._warm_start_threads = None
                results = self.model(frame, classes=[0], conf=minimum, imgsz=int(self.cfg['imgsz']), device='cpu', verbose=False)
                boxes = self.parse_boxes(results)
                crop = self.crop_for(boxes, frame.shape) if tick % 3 == 0 else None
                if crop:
                    x1, y1, x2, y2 = crop
                    extra = self.model(frame[y1:y2, x1:x2], classes=[0], conf=minimum, imgsz=384, device='cpu', verbose=False)
                    boxes.extend(self.parse_boxes(extra, (x1, y1)))
            return tracker.update(app._nms_boxes(boxes))
    return RecoveryPeopleEngine
