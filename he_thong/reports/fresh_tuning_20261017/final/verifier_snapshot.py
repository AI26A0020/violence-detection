"""Fresh, offline main2 inference and full-length H.264 videos with alert boxes.

Predictions are not ground-truth labels. Never starts notifications or a camera.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time

import main2 as app


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def natural(path):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r'(\d+)', str(path))]


def say(message):
    print(f'[{app.now_iso()}] {message}', flush=True)


class Observations:
    def __init__(self):
        self.rows = []

    def __call__(self, frame, row, boxes):
        self.rows.append(dict(frame_index=row['frame_index'], second=row['second'],
            shape=list(frame.shape), boxes=[dataclasses.asdict(box) for box in boxes]))


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def participant_ids(row):
    # Cooldown suppresses repeated notifications, never the visible active alert.
    return tuple(row['alert_track_ids']) if row['alert_active'] else ()


def restore_boxes(observation, width, height):
    sy, sx = height/observation['shape'][0], width/observation['shape'][1]
    boxes = []
    for original in observation['boxes']:
        data = dict(original, keypoints=None)
        for key in ('x1', 'x2'):
            data[key] *= sx
        for key in ('y1', 'y2'):
            data[key] *= sy
        boxes.append(app.PersonBox(**data))
    return boxes


def render(source, target, result, observations, width=960):
    """Decode every source frame; only earlier AI observations may color it.

    The application's optical-flow tracker moves display copies between AI
    samples. Its copies never enter inference. Red annotations use the same
    evidence-age boundary as the live preview and expire on a scene cut.
    """
    import av
    cap = app.cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError('Không mở được video để xuất: '+str(source))
    fps = result['source_fps']
    analysis_fps = result.get('signature', {}).get('config', {}).get('detection', {}).get('analysis_fps', 5)
    annotation_ttl = min(.4, max(.08, 1.1/analysis_fps))
    native_w, native_h = int(cap.get(app.cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(app.cv2.CAP_PROP_FRAME_HEIGHT))
    dw = max(2, min(width, native_w)//2*2)
    dh = max(2, round(native_h*dw/native_w)//2*2)
    rate = Fraction(str(fps)).limit_denominator(100000)
    temporary = target.with_name(target.stem+'.partial.mp4')
    container = None
    tracker, analyzed = app.PreviewBoxTracker(), None
    timeline = result['timeline']
    if len(timeline) != len(observations):
        cap.release()
        raise ValueError('Số quan sát người và AI không khớp')
    pointer, index, red_frames, red_samples, previous_second = -1, 0, 0, 0, -1.
    poster_saved = False
    sample_snapshots = []
    try:
        container = av.open(str(temporary), 'w', options={'movflags': '+faststart'})
        stream = container.add_stream('libx264', rate=rate)
        stream.width, stream.height, stream.pix_fmt = dw, dh, 'yuv420p'
        stream.options = {'crf': '22', 'preset': 'veryfast', 'threads': '2'}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            second = cap.get(app.cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isfinite(second) or second <= previous_second:
                second = max(index/fps, previous_second+1/fps)
            previous_second = second
            display = app.cv2.resize(frame, (dw, dh))
            fresh = False
            while pointer+1 < len(timeline) and timeline[pointer+1]['frame_index'] <= index:
                pointer += 1
                row = timeline[pointer]
                if row['frame_index'] != observations[pointer]['frame_index']:
                    raise ValueError('Sai vị trí quan sát')
                if row['frame_index'] != index:
                    raise ValueError('Thiếu khung nguồn chứa quan sát AI')
                analyzed = app.CameraView(frame=display.copy(),
                    boxes=restore_boxes(observations[pointer], dw, dh),
                    source_seq=index+1, analysis_seq=index+1, source_stamp=second+1.,
                    alert_active=row['alert_active'], alert_track_ids=participant_ids(row))
                fresh = True
            active, ids, boxes = False, (), []
            if analyzed is not None:
                row = timeline[pointer]
                boxes = tracker.project(display, index+1, second+1., analyzed,
                                        row['scene_epoch'], second+1.)
                age = second+1.-analyzed.source_stamp
                valid = age <= annotation_ttl+1e-8 and tracker.cut_stamp <= analyzed.source_stamp
                active = bool(row['alert_active'] and valid)
                ids = participant_ids(row) if active else ()
                if fresh:
                    boxes = analyzed.boxes
                app.draw_tactical_boxes(display, boxes, dw, dh, active,
                    row['smooth']*100, alert_track_ids=ids, previous_frame=not fresh)
            red = bool(active and any(b.track_id in ids for b in boxes))
            red_frames += int(red)
            red_samples += int(red and fresh)
            color = (0, 25, 240) if active else (55, 180, 80)
            if active:
                app.cv2.rectangle(display, (2, 2), (dw-3, dh-3), color, 4)
            app.cv2.rectangle(display, (6, 6), (min(dw-6, 710), 58), (18, 22, 30), -1)
            status = 'NGHI XO XAT - CAN KIEM TRA' if active else 'DANG THEO DOI'
            score = f"AI={timeline[pointer]['smooth']:.3f}" if pointer >= 0 else 'AI=WARMUP'
            app.cv2.putText(display, f'{source.name} | {second:.2f}s | {score}',
                (14, 26), app.cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, app.cv2.LINE_AA)
            app.cv2.putText(display, status, (14, 49), app.cv2.FONT_HERSHEY_SIMPLEX,
                .52, color, 1, app.cv2.LINE_AA)
            if not poster_saved and red:
                app.cv2.imwrite(str(target.with_suffix('.jpg')), display)
                poster_saved = True
            if fresh and red and len(sample_snapshots) < 12 and (not sample_snapshots or second-sample_snapshots[-1] >= 2):
                app.cv2.imwrite(str(target.parent/f'{target.stem}_{second:07.2f}s.jpg'), display)
                sample_snapshots.append(second)
            for packet in stream.encode(av.VideoFrame.from_ndarray(display, format='bgr24')):
                container.mux(packet)
            if index == 0:
                fallback_poster = display.copy()
            index += 1
        for packet in stream.encode():
            container.mux(packet)
        if index != result['decoded_frames'] or pointer+1 != len(timeline):
            raise ValueError('Số khung xuất khác số khung phân tích')
        if not poster_saved and index:
            app.cv2.imwrite(str(target.with_suffix('.jpg')), fallback_poster)
    finally:
        cap.release()
        if container is not None:
            container.close()
    with av.open(str(temporary)) as check:
        decoded = sum(1 for _ in check.decode(video=0))
    if decoded != index:
        raise ValueError('Video xuất không giải mã đủ khung')
    temporary.replace(target)
    return dict(frames=index, decoded_output_frames=decoded, fps=float(rate),
        red_person_frames=red_frames, red_person_samples=red_samples,
        width=dw, height=dh, audio='No audio track; visual review only',
        sha256=sha256(target), poster=target.with_suffix('.jpg').name)


def gallery(output, summary):
    cards = []
    for item in summary['results']:
        name = html.escape(item['file'])
        if item.get('error'):
            cards.append(f'<article><h2>{name}</h2><p>{html.escape(item["error"])}</p></article>')
            continue
        video = html.escape(item['annotated_video'])
        buttons = ' '.join(f'<button data-time="{a["second"]}" onclick="jump(this)">{a["second"]:.1f}s</button>' for a in item['automatic_alerts'])
        cards.append(f'''<article><h2>{name}</h2><video controls preload="metadata" poster="{item['render']['poster']}"><source src="{video}" type="video/mp4"></video>
<p>{item['duration_seconds']:.1f}s · {item['decoded_frames']} khung giải mã · {len(item['automatic_alerts'])} lần cảnh báo · {item['render']['red_person_frames']} khung có viền người đỏ</p>
<p>{html.escape(item['decision']['label'])}</p><div>{buttons or 'Chưa phát cảnh báo tự động.'}</div>
<p><a href="{html.escape(item['analysis'])}">Dữ liệu phân tích</a></p></article>''')
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kết quả TOTAL_VIDEO video SchoolGuard</title>
<style>body{font:16px system-ui;background:#101720;color:#e5edf5;margin:0;padding:24px;max-width:1200px;margin:auto}h1{font-size:28px}p{line-height:1.6}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}article{background:#1a2634;border-radius:12px;padding:18px}video{width:100%;background:#000}button{background:#ac2636;color:white;border:0;border-radius:5px;padding:8px;margin:3px;cursor:pointer}a{color:#8dd3ff}@media(max-width:500px){main{display:block}article{margin-bottom:20px}}</style>
<h1>SchoolGuard · Kiểm tra toàn bộ video</h1><p>Chạy mới bằng main2.py. Khung đỏ đánh dấu người có bằng chứng tương tác khi cảnh báo đang hoạt động. Khung xanh là người đang theo dõi. Bấm mốc giây để xem trước cảnh báo 2 giây.</p>
<p>Đây là kết quả dự đoán AI, chưa phải nhãn đúng đã được người phụ trách xác nhận. Có thể báo nhầm hoặc bỏ sót; chưa có căn cứ cam kết chính xác 100%. Bản xem giữ đầy đủ khung hình, không kèm âm thanh.</p>
<p><a href="REPORT.md">Báo cáo</a> · <a href="summary.json">Kết quả tổng hợp</a></p><main>'''+''.join(cards)+'''</main>
<script>function jump(b){const v=b.closest('article').querySelector('video');v.currentTime=Math.max(0,Number(b.dataset.time)-2);v.play()}</script></html>'''
    page = page.replace('TOTAL_VIDEO', str(summary['total']))
    (output/'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=app.ROOT/'AI26A0020/test_videos')
    parser.add_argument('--output', type=Path, default=app.ROOT/'reports'/('test_videos_'+time.strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--only', action='append')
    parser.add_argument('--render-only', action='store_true', help='Render previously saved fresh inference after checking all hashes')
    parser.add_argument('--config', help='Explicit candidate configuration; defaults to the application configuration')
    parser.add_argument('--analysis-only', action='store_true', help='Save actual inference; render separately after selection')
    args = parser.parse_args()
    folder, output = args.folder.resolve(), args.output.resolve()
    if not folder.is_dir() or output == folder or output.is_relative_to(folder):
        parser.error('Thư mục đầu ra phải tách khỏi thư mục video nguồn')
    paths = sorted((p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in app.VIDEO_SUFFIXES), key=natural)
    if args.only:
        requested = set(args.only)
        paths = [p for p in paths if p.name in requested]
        if {p.name for p in paths} != requested:
            parser.error('Thiếu video được yêu cầu')
    if not paths:
        parser.error('Không có video nguồn')
    if len({p.stem for p in paths}) != len(paths):
        parser.error('Tên video trùng nhau; cần tên riêng cho mỗi video')
    output.mkdir(parents=True, exist_ok=True)
    app.runtime_imports()
    cfg = app.load_config(args.config)
    cfg['notifications']['enabled'] = cfg['feedback']['enabled'] = cfg['quick_clip']['enabled'] = False
    cfg['display']['sound'] = False
    cfg['evidence'].update(enabled=False, continuous=False)
    if cfg['model']['video_context_enabled']:
        raise ValueError('Bộ xuất này yêu cầu cấu hình hình ảnh hiện hành, không có video context nghiên cứu')
    from verify_recall_runtime import redact_config
    config = redact_config({k: cfg[k] for k in ('model', 'person', 'detection', 'school_policy')})
    code_files = ['main2.py', 'short_contact.py', 'confidence_head.py', 'school_policy.py', 'temporal_visual.py', 'action_runtime.py']
    signature = dict(code={n: sha256(app.ROOT/n) for n in code_files},
        models={cfg[s]['path']: sha256(app.resolve_path(cfg[s]['path'])) for s in ('model', 'person')}, config=config)
    metadata = output/'configuration.json'
    if args.render_only:
        if read(metadata)['signature'] != signature:
            raise ValueError('Mã, mô hình hoặc cấu hình thay đổi; phải chạy suy luận mới')
    else:
        if metadata.exists():
            raise FileExistsError('Dùng thư mục kết quả mới để giữ bằng chứng lần chạy trước')
        app.atomic_json(metadata, dict(created=app.now_iso(), signature=signature,
            verifier_sha256=sha256(__file__), offline=True))
        (output/'main2_verified.py').write_bytes((app.ROOT/'main2.py').read_bytes())
        (output/'verifier_snapshot.py').write_bytes(Path(__file__).read_bytes())
    visual = people = None
    if not args.render_only:
        visual, people = app.VisualEngine(cfg['model']), app.PeopleEngine(cfg['person'])
        if people.model is None:
            raise RuntimeError(people.reason)
    summary = dict(created=app.now_iso(), folder=str(folder), total=len(paths), completed=0, results=[],
        external_notifications_sent=False, inference='Fresh main2.analyze_clip; no cached predictions',
        accuracy='Not established: no complete, independently human-confirmed interval labels')
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        say(f'START {index}/{len(paths)} {path.name}')
        input_hash = sha256(path)
        analysis, observation_file = output/(path.stem+'_analysis.json'), output/(path.stem+'_people.json')
        try:
            if args.render_only:
                result, observed = read(analysis), read(observation_file)
                if result['source_sha256'] != input_hash or result['signature'] != signature or result['cancelled']:
                    raise ValueError('Bằng chứng suy luận không khớp video hiện tại')
                if result.get('observations_sha256') != sha256(observation_file):
                    raise ValueError('Bằng chứng hộp người đã thay đổi; phải chạy suy luận mới')
            else:
                observer = Observations()
                result = app.analyze_clip(path, cfg, engine=visual, people_engine=people, observer=observer, emit=say)
                observed = observer.rows
                if result['cancelled'] or not result['decoded_frames'] or not result['timeline']:
                    raise ValueError('Phân tích chưa hoàn tất hoặc không có kết quả AI')
                if result['reported_frames'] > 0 and result['decoded_frames'] != result['reported_frames']:
                    raise ValueError('Không giải mã đủ số khung nguồn được khai báo')
                result.update(source_sha256=input_hash, signature=signature, inference_wall_seconds=round(time.perf_counter()-started, 2))
                observation_file.write_text(json.dumps(observed, ensure_ascii=False, default=json_value), encoding='utf-8')
                result['observations_sha256'] = sha256(observation_file)
                app.atomic_json(analysis, result)
            target = output/(path.stem+'_annotated.mp4')
            say(f'RENDER {path.name}: {len(result["automatic_alerts"])} cảnh báo, {result["decoded_frames"]} khung')
            rendered = {} if args.analysis_only else render(path, target, result, observed)
            if sha256(path) != input_hash or any(sha256(app.ROOT/n) != d for n, d in signature['code'].items()):
                raise ValueError('Video hoặc mã thay đổi trong khi chạy')
            if any(sha256(app.resolve_path(n)) != d for n, d in signature['models'].items()):
                raise ValueError('Mô hình thay đổi trong khi chạy')
            item = dict(file=path.name, source_sha256=input_hash, duration_seconds=result['duration_seconds'],
                decoded_frames=result['decoded_frames'], samples=len(result['timeline']), automatic_alerts=result['automatic_alerts'],
                decision=result['decision'], render=rendered, analysis=analysis.name, annotated_video=target.name,
                inference_wall_seconds=result['inference_wall_seconds'], wall_seconds=round(time.perf_counter()-started, 2))
            say(f'DONE {path.name}: '+(f'{rendered["red_person_frames"]} khung có viền người đỏ' if rendered else 'saved inference'))
        except Exception as exc:
            item = dict(file=path.name, source_sha256=input_hash, error=app.redact(exc))
            say(f'ERROR {path.name}: {item["error"]}')
        summary['results'].append(item)
        summary['completed'] = index
        app.atomic_json(output/'summary.json', summary)
        if not args.analysis_only:
            gallery(output, summary)
    summary['errors'] = sum(bool(r.get('error')) for r in summary['results'])
    summary['all_sources_preserved'] = all(sha256(p) == r['source_sha256'] for p, r in zip(paths, summary['results']))
    app.atomic_json(output/'summary.json', summary)
    say('Kết quả: '+str(output/'index.html'))
    return 1 if summary['errors'] or not summary['all_sources_preserved'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
