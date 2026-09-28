@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting ngrok tunnel on port 8001 (Portfolio AI API)...
echo ========================================================
"C:\Users\Rrs computers\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe" http 8001
pause
