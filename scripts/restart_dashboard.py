"""Safe restart of the RainNet Streamlit server when it serves stale UI.

Only terminates a process if it owns port 8501 AND the process command line
references this project's exact app/dashboard.py. Other processes are untouched.
Run from VS Code: py -3.8 scripts/restart_dashboard.py
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import requests

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app" / "dashboard.py").resolve()
PORT = 8501
HEALTH = f"http://127.0.0.1:{PORT}/_stcore/health"


def process_using_port():
    for con in psutil.net_connections(kind="inet"):
        if (con.status == psutil.CONN_LISTEN and con.laddr
                and con.laddr.port == PORT and con.pid):
            return psutil.Process(con.pid)
    return None


def _matches_rainnet(proc):
    try:
        command = [str(arg).replace("/", "\\").casefold()
                   for arg in proc.cmdline()]
        target = str(APP).replace("/", "\\").casefold()
        return target in command and "streamlit" in " ".join(command)
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def main():
    old = process_using_port()
    if old is not None:
        if not _matches_rainnet(old):
            raise RuntimeError(f"Port {PORT} is used by an unrelated process "
                               f"(PID {old.pid}); refusing to stop it.")
        print("STOP_OLD_RAINNET", old.pid, flush=True)
        old.terminate()
        try:
            old.wait(timeout=14)
        except psutil.TimeoutExpired:
            raise RuntimeError("Old RainNet did not stop; no new process started.")
    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)
    stdout = (logdir / "streamlit_restart_stdout.log").open("a", encoding="utf-8")
    stderr = (logdir / "streamlit_restart_stderr.log").open("a", encoding="utf-8")
    command = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(PORT),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "auto",
    ]
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
             if os.name == "nt" else 0)
    try:
        child = subprocess.Popen(command, cwd=str(ROOT),
                                 stdout=stdout, stderr=stderr,
                                 creationflags=flags)
    finally:
        stdout.close()
        stderr.close()
    for attempt in range(35):
        if child.poll() is not None:
            raise RuntimeError("RainNet stopped during startup; inspect logs.")
        try:
            r = requests.get(HEALTH, timeout=2)
            if r.ok and "ok" in r.text.lower():
                print("RAINNET_RESTART_OK", child.pid, HEALTH, flush=True)
                print("OPEN", f"http://127.0.0.1:{PORT}/?rainnet_build=yellow_38provinces",
                      flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("Streamlit did not become healthy; inspect logs.")


if __name__ == "__main__":
    main()
