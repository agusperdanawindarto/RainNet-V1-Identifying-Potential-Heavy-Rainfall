@echo off
setlocal
cd /d "%~dp0"
echo Restarting RainNet V1 with latest code and new yellow theme...
py -3.8 scripts\restart_dashboard.py
if errorlevel 1 (
 echo RainNet failed to restart. Check logs in PROGRAM_RAINNET\logs.
 pause
 exit /b 1
)
start "" "http://127.0.0.1:8501/?rainnet_build=yellow_38provinces"
