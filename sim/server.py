# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "websocket-client"]
# ///
"""Reach a remote JupyterLab from the command line: copy files and run shell commands.

Used to run the simulation suite on a server whose only door is JupyterLab.

    uv run server.py put <local file> <remote path>
    uv run server.py get <remote path> <local file>
    uv run server.py sh '<command>'

Remote paths are relative to JupyterLab's root directory. The address comes from
APRIL_JUPYTER_URL (default http://localhost:8033, i.e. behind an SSH tunnel) and the
password from the file named by APRIL_JUPYTER_SECRET
(default ~/.config/april_engine/jupyter_token).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import uuid
from pathlib import Path

import requests
import websocket

BASE = os.environ.get("APRIL_JUPYTER_URL", "http://localhost:8033").rstrip("/")
SECRET = Path(
    os.environ.get("APRIL_JUPYTER_SECRET", "~/.config/april_engine/jupyter_token")
).expanduser()
TIMEOUT = 60


def login() -> requests.Session:
    session = requests.Session()
    session.get(f"{BASE}/login", timeout=TIMEOUT)
    session.post(
        f"{BASE}/login",
        data={"_xsrf": session.cookies.get("_xsrf"), "password": SECRET.read_text()},
        timeout=TIMEOUT,
        allow_redirects=False,
    )
    if session.get(f"{BASE}/api/status", timeout=TIMEOUT).status_code != 200:
        sys.exit(f"JupyterLab at {BASE} refused the password in {SECRET}")
    session.headers["X-XSRFToken"] = session.cookies.get("_xsrf") or ""
    return session


def put(session: requests.Session, local: str, remote: str) -> None:
    content = base64.b64encode(Path(local).read_bytes()).decode()
    response = session.put(
        f"{BASE}/api/contents/{remote}",
        json={"type": "file", "format": "base64", "content": content},
        timeout=10 * TIMEOUT,
    )
    response.raise_for_status()


def get(session: requests.Session, remote: str, local: str) -> None:
    response = session.get(
        f"{BASE}/api/contents/{remote}",
        params={"type": "file", "format": "base64"},
        timeout=10 * TIMEOUT,
    )
    response.raise_for_status()
    Path(local).write_bytes(base64.b64decode(response.json()["content"]))


def sh(session: requests.Session, command: str) -> int:
    """Runs `command` under bash on the server through a throwaway kernel; returns its status."""
    kernel = session.post(f"{BASE}/api/kernels", json={"name": "python3"}, timeout=TIMEOUT).json()
    try:
        socket = websocket.create_connection(
            f"{BASE.replace('http', 'ws', 1)}/api/kernels/{kernel['id']}/channels",
            cookie="; ".join(f"{name}={value}" for name, value in session.cookies.items()),
            origin=BASE,
            timeout=None,
        )
        request = uuid.uuid4().hex
        code = (
            "import subprocess\n"
            f"process = subprocess.Popen({command!r}, shell=True, executable='/bin/bash',"
            " stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
            "for line in process.stdout:\n"
            "    print(line, end='')\n"
            "status = process.wait()\n"
        )
        socket.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": request,
                        "username": "",
                        "session": uuid.uuid4().hex,
                        "msg_type": "execute_request",
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "channel": "shell",
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {"status": "status"},
                        "allow_stdin": False,
                    },
                }
            )
        )
        status = 1
        while True:
            message = json.loads(socket.recv())
            if message.get("parent_header", {}).get("msg_id") != request:
                continue
            kind, content = message["msg_type"], message["content"]
            if kind == "stream":
                print(content["text"], end="", flush=True)
            elif kind == "error":
                print("\n".join(content["traceback"]), file=sys.stderr)
            elif kind == "execute_reply" and content["status"] == "ok":
                status = int(content["user_expressions"]["status"]["data"]["text/plain"])
            elif kind == "status" and content["execution_state"] == "idle":
                socket.close()
                return status
    finally:
        session.delete(f"{BASE}/api/kernels/{kernel['id']}", timeout=TIMEOUT)


def main() -> int:
    commands = {"put": (put, 2), "get": (get, 2), "sh": (sh, 1)}
    if len(sys.argv) < 2 or sys.argv[1] not in commands or len(sys.argv) - 2 != commands[sys.argv[1]][1]:
        sys.exit(__doc__)
    action, _ = commands[sys.argv[1]]
    return action(login(), *sys.argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main())
