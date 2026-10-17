@echo off
setlocal
cd /d "%~dp0"
if exist "he_thong\.venv\Scripts\python.exe" (
    "he_thong\.venv\Scripts\python.exe" -B -X utf8 "he_thong\menu.py" %*
) else (
    py -3.11 -B -X utf8 "he_thong\menu.py" %*
)
if errorlevel 1 pause
