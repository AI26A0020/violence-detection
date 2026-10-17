"""Collect adult false-alarm feedback into a reviewable, hash-bound learning inbox."""
from pathlib import Path
import hashlib,json

CATEGORIES=('play','sports','crowd','fall','walking','social_contact')

def sha256(path):
    path=Path(path);before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Clip thay đổi trong lúc kiểm tra; cần xem và duyệt lại.')
    return digest.hexdigest()

def candidate_media(store,row):
    quick=store.folder/'quick_clips'/(row['id']+'.mp4')
    source=quick if quick.is_file() and quick.with_suffix('.json').is_file() else Path(row.get('clip',''))
    if source.is_file() and source.resolve().is_relative_to(store.folder) and (source==quick or not row.get('clip_state','').startswith('Đang ghi')):
        return source.resolve()
    return None

def init_learning(store):
    with store.lock,store.conn:
        store.conn.execute('''CREATE TABLE IF NOT EXISTS learning_reviews (
            event_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, reviewer TEXT NOT NULL,
            category TEXT NOT NULL, approved_at TEXT NOT NULL)''')

def approve(store,eid,reviewer,category):
    from datetime import datetime
    from evaluation_workflow import provisional_review
    row=store.get(eid)
    if not row or row['status']!='Báo nhầm':raise ValueError('Chỉ duyệt học từ sự kiện đang được đánh giá Báo nhầm.')
    if not reviewer.strip() or category not in CATEGORIES:raise ValueError('Cần tên người duyệt và loại cảnh.')
    if provisional_review({'reviewer':reviewer}):raise ValueError('Phê duyệt học cần người phụ trách; tên/ghi chú duyệt AI hoặc sơ bộ không được chấp nhận.')
    media=candidate_media(store,row)
    if media is None:raise ValueError('Clip chưa hoàn tất. Xem toàn bộ clip trước khi duyệt.')
    digest=sha256(media)
    with store.lock,store.conn:
        if store.conn.execute('SELECT status FROM events WHERE id=?',(eid,)).fetchone()['status']!='Báo nhầm':
            raise ValueError('Đánh giá vừa đổi; hãy kiểm tra lại sự kiện.')
        store.conn.execute('INSERT OR REPLACE INTO learning_reviews VALUES(?,?,?,?,?)',
                           (eid,digest,reviewer.strip(),category,datetime.now().astimezone().isoformat()))
        store.conn.execute('INSERT INTO audit(created,event_id,action,detail) VALUES(?,?,?,?)',
                           (datetime.now().astimezone().isoformat(),eid,'approve_learning',digest))
    return sync_inbox(store)

def sync_inbox(store):
    from main2 import atomic_json
    samples=[]
    for row in store.rows(status='Báo nhầm',limit=None):
        media=candidate_media(store,row)
        if media is None:continue
        digest=sha256(media)
        with store.lock:proof=store.conn.execute('SELECT * FROM learning_reviews WHERE event_id=? AND sha256=?',(row['id'],digest)).fetchone()
        samples.append({'event_id':row['id'],'path':str(media),'sha256':digest,'label':False,
            'source_group':'camera:'+row['camera']+':'+row['created'][:10],
            'category':proof['category'] if proof else '', 'reviewer':proof['reviewer'] if proof else '',
            'reviewed':bool(proof),'approved_for_training':bool(proof),
            'human_verified':bool(proof), 'review_type':'human' if proof else 'pending',
            'note':'Nhãn Báo nhầm tự vào hộp chờ; chỉ dùng học khi người phụ trách xem toàn clip và duyệt.'})
    value={'schema':1,'purpose':'hard_negative_review','samples':samples,'automatic_deployment':False}
    target=store.folder/'hard_negatives'/'inbox'/'manifest.json'
    if not target.exists() or json.loads(target.read_text(encoding='utf-8'))!=value:atomic_json(target,value)
    return {'manifest':str(target),'candidates':len(samples),'approved':sum(s['approved_for_training'] for s in samples)}
