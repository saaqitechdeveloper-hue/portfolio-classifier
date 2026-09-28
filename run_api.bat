@echo off
cd /d "%~dp0"
echo Starting Portfolio AI Image Classification API on http://127.0.0.1:8001 ...
call .venv\Scripts\activate.bat
python -m uvicorn api.app:app --host 127.0.0.1 --port 8001
pause
