"""Local launcher for the collected SchoolGuard application and its test videos."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
PYTHON = ROOT / '.venv/Scripts/python.exe'


def run(action, extra):
    if action == 'chay':
        return subprocess.call([str(PYTHON), '-B', '-X', 'utf8', str(ROOT/'main2.py'), *extra], cwd=ROOT)
    if action in ('moi', 'cu'):
        folder = PACKAGE / ('fresh_eval_videos_20261017' if action == 'moi' else 'test_videos')
        return subprocess.call([str(PYTHON), '-B', '-X', 'utf8', str(ROOT/'verify_schoolguard_videos.py'),
                                '--folder', str(folder), *extra], cwd=ROOT)
    if action == 'ket-qua':
        pages = list((ROOT/'reports').rglob('index.html'))
        if not pages:
            print('Chua co ket qua. Hay chay kiem tra video truoc.')
            return 1
        os.startfile(str(max(pages, key=lambda p: p.stat().st_mtime)))
        return 0
    if action == 'cai-dat':
        return subprocess.call(['cmd', '/d', '/c', str(ROOT/'CAI_THU_VIEN.bat')], cwd=ROOT)
    raise ValueError('Lenh khong hop le: '+action)


def main():
    if len(sys.argv) > 1:
        return run(sys.argv[1], sys.argv[2:])
    choices = {'1': 'chay', '2': 'moi', '3': 'cu', '4': 'ket-qua', '5': 'cai-dat'}
    while True:
        print('\nAI26A0020 - SchoolGuard\n'
              '1. Mo ung dung\n'
              '2. Kiem tra 25 video moi\n'
              '3. Kiem tra 10 video trong test_videos\n'
              '4. Xem ket qua gan nhat\n'
              '5. Cai lai thu vien khi chuyen may\n'
              '0. Thoat')
        choice = input('Chon: ').strip()
        if choice == '0':
            return 0
        if choice not in choices:
            print('Hay chon tu 0 den 5.')
            continue
        try:
            result = run(choices[choice], [])
            if result:
                print('Tac vu chua hoan tat. Xem thong bao loi o tren.')
        except (OSError, ValueError) as exc:
            print(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
