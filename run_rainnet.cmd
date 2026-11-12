@echo off
setlocal
cd /d "%~dp0"
echo Starting RainNet V1 - multi-agent JMA research dashboard...
py -3.8 -m streamlit run app\dashboard.py --server.port 8501
if errorlevel 1 (
 echo Python 3.8 / Streamlit not found. Select a Python 3.8 environment with requirements installed in VS Code.
 pause
)
