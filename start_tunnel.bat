@echo off
cd /d "%~dp0"
echo Starting Cloudflare Public Tunnel for Portfolio AI API (port 8001)...
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8001
pause
