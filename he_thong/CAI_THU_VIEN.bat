@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Cai dat hoan tat. Mo BAT_DAU.bat o thu muc ben ngoai de su dung.
pause
exit /b 0
:failed
echo Cai dat chua thanh cong. Can Python 3.11 va ket noi mang.
pause
exit /b 1
