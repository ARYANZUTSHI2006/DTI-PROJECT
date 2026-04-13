from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    root = Path(__file__).resolve().parent
    python_exe = root / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        print("Missing interpreter:", python_exe)
        return 1

    cmd = [
        str(python_exe),
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        "8501",
        "--server.address",
        "127.0.0.1",
    ]

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    for _ in range(30):
        if _is_port_open("127.0.0.1", 8501):
            print(f"Streamlit started at http://127.0.0.1:8501 (PID {process.pid})")
            return 0
        time.sleep(0.5)

    print(f"Started process (PID {process.pid}), but port 8501 not ready yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
