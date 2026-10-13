#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="$task_root/.venv/bin/python"
task_models="$task_root/pretrained_models"
task_run="$task_root/output/web_demo"
task_pid_file="$task_run/web_demo.pid"
task_log="$task_run/web_demo.log"
mkdir -p "$task_run"

running_pid() {
    [[ -f "$task_pid_file" ]] || return 1
    local pid command_line
    read -r pid < "$task_pid_file"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [[ "$command_line" == "$task_python $task_root/app.py "* ]] || return 1
    printf '%s\n' "$pid"
}

case "${1:-status}" in
    start)
        exec 9>"$task_run/control.lock"
        flock -n 9 || { echo "Another start/stop operation is running." >&2; exit 1; }
        if task_pid="$(running_pid)"; then
            echo "Web Demo is already running (PID $task_pid), port 8088."
            exit 0
        fi
        [[ -x "$task_python" ]] || { echo "Create .venv with uv sync --frozen --extra dev first." >&2; exit 1; }
        unset PYTHONPATH PYTHONHOME
        export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
        export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 GRADIO_ANALYTICS_ENABLED=False GRADIO_SHARE=False
        export GRADIO_TEMP_DIR="$task_run/gradio"
        export PYTHONUNBUFFERED=1
        # Explicit loopback bypass keeps Gradio's local health check out of any proxy.
        export NO_PROXY="127.0.0.1,localhost,0.0.0.0${NO_PROXY:+,$NO_PROXY}"
        "$task_python" - "$task_models" <<'PY'
import importlib.util
import socket
import sys
from pathlib import Path

root = Path(sys.argv[1])
# Reuse startup validation so incomplete downloads cannot silently fall back online.
spec = importlib.util.spec_from_file_location("web_demo_startup", root.parent / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
app.validate_local_models(str(root / "VoxCPM2"), str(root / "SenseVoiceSmall"), str(root / "ZipEnhancer"))
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("0.0.0.0", 8088))
PY
        cd "$task_root"
        nohup setsid "$task_python" "$task_root/app.py" \
            --model-id "$task_models/VoxCPM2" \
            --asr-model-id "$task_models/SenseVoiceSmall" \
            --zipenhancer-model-id "$task_models/ZipEnhancer" \
            --local-files-only --no-optimize --host 0.0.0.0 --port 8088 --device cuda \
            >> "$task_log" 2>&1 < /dev/null 9>&- &
        task_pid=$!
        printf '%s\n' "$task_pid" > "$task_pid_file"
        if ! "$task_python" - "$task_pid" <<'PY'
import os
import sys
import time
import urllib.request

pid = int(sys.argv[1])
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for attempt in range(60):
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/stat") as process:
            if process.read().split(") ", 1)[1].startswith("Z"):
                break
        with opener.open("http://127.0.0.1:8088/config", timeout=1) as response:
            if response.status == 200:
                sys.exit(0)
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(0.5)
sys.exit(1)
PY
        then
            echo "Web Demo did not become ready; inspect $task_log. PID: $task_pid" >&2
            tail -n 30 "$task_log" >&2
            exit 1
        fi
        echo "Web Demo started (PID $task_pid): http://<server-ip>:8088"
        echo "Models load on first use. Log: $task_log"
        ;;
    stop)
        exec 9>"$task_run/control.lock"
        flock -n 9 || { echo "Another start/stop operation is running." >&2; exit 1; }
        if task_pid="$(running_pid)"; then
            kill -TERM "$task_pid"
            for attempt in {1..30}; do
                if ! running_pid > /dev/null; then
                    rm -f "$task_pid_file"
                    echo "Web Demo stopped."
                    exit 0
                fi
                sleep 1
            done
            echo "Process $task_pid is still shutting down; inspect $task_log." >&2
            exit 1
        fi
        rm -f "$task_pid_file"
        echo "Web Demo is not running."
        ;;
    status)
        if task_pid="$(running_pid)"; then
            echo "Web Demo is running (PID $task_pid): http://<server-ip>:8088"
        else
            echo "Web Demo is not running."
            exit 1
        fi
        ;;
    logs)
        [[ -f "$task_log" ]] && tail -n 60 "$task_log" || echo "No Web Demo log yet."
        ;;
    *)
        echo "Usage: bash scripts/web_demo.sh {start|stop|status|logs}" >&2
        exit 2
        ;;
esac
