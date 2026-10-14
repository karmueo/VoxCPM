#!/usr/bin/env python3
"""Build and run the pinned VoxCPM2 llama.cpp-omni CUDA CLI."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import math
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "llama_cpp_omni"
SOURCE_DIR = OUTPUT_ROOT / "src"
BUILD_DIR = OUTPUT_ROOT / "build"
LOG_DIR = OUTPUT_ROOT / "logs"
RUN_DIR = OUTPUT_ROOT / "runs"
MODEL_DIR = REPO_ROOT / "pretrained_models" / "VoxCPM2-GGUF"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

ENGINE_URL = "https://github.com/tc-mb/llama.cpp-omni.git"
ENGINE_COMMIT = "722eb4a6724598a8c83d313cd60e744deabb9c24"
MODEL_REPO = "DennisHuang648/VoxCPM2-GGUF"
MODEL_REVISION = "169f64d8b98bbaab1761e4ca3a83e6af653456cc"
MODEL_SOURCES = {
    "huggingface": {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "resolve_url_template": f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/FILENAME",
        "download_url_template": (
            f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{{filename}}?download=true"
        ),
    },
    "modelscope": {
        "repo": "DennisHuang/VoxCPM2-GGUF",
        "revision": "bb319272866df0419de6a2357c9e675421b2688e",
        "resolve_url_template": (
            "https://modelscope.cn/models/DennisHuang/VoxCPM2-GGUF/resolve/"
            "bb319272866df0419de6a2357c9e675421b2688e/FILENAME"
        ),
        "download_url_template": (
            "https://modelscope.cn/models/DennisHuang/VoxCPM2-GGUF/resolve/"
            "bb319272866df0419de6a2357c9e675421b2688e/{filename}"
        ),
    },
}
DEFAULT_MODEL_SOURCE = "huggingface"
MODEL_FILES = {
    "base": {
        "name": "VoxCPM2-BaseLM-F16.gguf",
        "size": 3247980544,
        "sha256": "8be62e899f8f32b3c2109c99950b46671135a8f40b27b1df720f6823d36cc59e",
    },
    "acoustic": {
        "name": "VoxCPM2-Acoustic-F16.gguf",
        "size": 1825096352,
        "sha256": "5bde898488ad635ff55d24da53543768fa33d5e5cdc538ce190e5ef831038e85",
    },
}
DEFAULT_GPU = 1
CUDA_ARCH_REQUESTED = "120"
BUILD_FLAGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH_REQUESTED}",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_OPENSSL=OFF",
]
TEST_TEXTS = {
    "english": "Hello, welcome to VoxCPM2.",
    "chinese": "你好，欢迎使用 VoxCPM2 语音合成。",
}
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


class CliError(RuntimeError):
    """An actionable error for the command-line interface."""


@dataclass(frozen=True)
class InferOptions:
    text: str
    output: Path
    gpu: int = DEFAULT_GPU
    reference_audio: Path | None = None
    reference_text: str | None = None
    base_model: Path = MODEL_DIR / MODEL_FILES["base"]["name"]
    acoustic_model: Path = MODEL_DIR / MODEL_FILES["acoustic"]["name"]
    seed: int = 42
    timesteps: int = 10
    cfg: float = 2.0
    steps: int = 200


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _run_capture(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except FileNotFoundError as exc:
        raise CliError(f"Required command not found: {command[0]}") from exc
    if result.returncode:
        detail = result.stdout.strip() or f"exit code {result.returncode}"
        raise CliError(f"Command failed: {shlex.join(command)}\n{detail}")
    return result.stdout


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shlex.join(command)}\nLog: {log_path}", flush=True)
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
            return_code = process.wait()
    except FileNotFoundError as exc:
        raise CliError(f"Required command not found: {command[0]}") from exc
    if return_code:
        raise CliError(f"Command failed with exit code {return_code}; see {log_path}")


def _git(source_dir: Path, *args: str) -> str:
    return _run_capture(["git", "-C", str(source_dir), *args])


def _prepare_engine() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DIR.exists():
        SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run_logged(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", ENGINE_URL, str(SOURCE_DIR)],
            cwd=REPO_ROOT,
            log_path=LOG_DIR / "clone.log",
        )
    if not (SOURCE_DIR / ".git").exists():
        raise CliError(f"Engine source path exists but is not a Git checkout: {SOURCE_DIR}")

    origin = _git(SOURCE_DIR, "remote", "get-url", "origin").strip().rstrip("/")
    if origin.removesuffix(".git") != ENGINE_URL.removesuffix(".git"):
        raise CliError(f"Unexpected engine origin at {SOURCE_DIR}: {origin}")
    dirty = _git(SOURCE_DIR, "status", "--porcelain").strip()
    if dirty:
        raise CliError(f"Refusing to change a modified engine checkout: {SOURCE_DIR}\n{dirty}")

    _run_logged(
        ["git", "-C", str(SOURCE_DIR), "fetch", "--depth", "1", "origin", ENGINE_COMMIT],
        cwd=REPO_ROOT,
        log_path=LOG_DIR / "fetch.log",
    )
    _run_capture(["git", "-C", str(SOURCE_DIR), "checkout", "--detach", ENGINE_COMMIT])
    actual = _git(SOURCE_DIR, "rev-parse", "HEAD").strip()
    if actual != ENGINE_COMMIT:
        raise CliError(f"Engine commit mismatch: expected {ENGINE_COMMIT}, got {actual}")


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache_key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            while chunk := file_obj.read(8 * 1024 * 1024):
                digest.update(chunk)
        _HASH_CACHE[cache_key] = digest.hexdigest()
    return _HASH_CACHE[cache_key]


def _model_download_url(info: dict[str, Any], source_name: str) -> str:
    try:
        source = MODEL_SOURCES[source_name]
    except KeyError as exc:
        raise CliError(f"Unknown model source: {source_name}") from exc
    return source["download_url_template"].format(filename=info["name"])


def _model_record(
    path: Path,
    expected_size: int,
    expected_sha: str,
    *,
    download_source: str,
    resumed: bool = False,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": expected_size,
        "sha256": expected_sha,
        "mtime_ns": path.stat().st_mtime_ns,
        "download_source": download_source,
        "download_resumed": resumed,
    }


def _download_model(info: dict[str, Any], source_name: str = DEFAULT_MODEL_SOURCE) -> dict[str, Any]:
    if source_name not in MODEL_SOURCES:
        raise CliError(f"Unknown model source: {source_name}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / info["name"]
    expected_sha = info["sha256"]
    expected_size = info["size"]
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_file() and target.stat().st_size == expected_size and _sha256_file(target) == expected_sha:
        partial.unlink(missing_ok=True)
        print(f"Verified existing model: {target}", flush=True)
        return _model_record(
            target,
            expected_size,
            expected_sha,
            download_source="pre-existing (origin unknown)",
        )

    if partial.is_file():
        partial_size = partial.stat().st_size
        if partial_size == expected_size and _sha256_file(partial) == expected_sha:
            partial.replace(target)
            return _model_record(
                target,
                expected_size,
                expected_sha,
                download_source="pre-existing partial (origin unknown)",
            )
        if partial_size >= expected_size:
            partial.unlink()

    url = _model_download_url(info, source_name)
    print(f"Downloading {info['name']} from {url}", flush=True)
    request_headers = {"User-Agent": "VoxCPM2-CUDA-deployment/1.0"}
    last_error: Exception | None = None
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset >= expected_size:
            partial.unlink(missing_ok=True)
            offset = 0
        headers = dict(request_headers)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)
        digest = hashlib.sha256()
        if offset:
            with partial.open("rb") as existing:
                while chunk := existing.read(8 * 1024 * 1024):
                    digest.update(chunk)
        written = offset
        print(f"  attempt {attempt}/5; resuming at byte {offset:,}", flush=True)
        try:
            with urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                append = bool(offset and status == 206)
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise CliError(f"Invalid Range response while downloading {target.name}: {content_range!r}")
                else:
                    offset = 0
                    written = 0
                    digest = hashlib.sha256()
                mode = "ab" if append else "wb"
                with partial.open(mode) as file_obj:
                    while chunk := response.read(8 * 1024 * 1024):
                        file_obj.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        if written and written % (256 * 1024 * 1024) < len(chunk):
                            print(f"  downloaded {written:,} bytes", flush=True)
            if written == expected_size:
                actual_sha = digest.hexdigest()
                if actual_sha != expected_sha:
                    partial.unlink(missing_ok=True)
                    raise CliError(
                        f"Model SHA-256 mismatch for {target.name}: expected {expected_sha}, got {actual_sha}"
                    )
                partial.replace(target)
                return _model_record(
                    target,
                    written,
                    expected_sha,
                    download_source=source_name,
                    resumed=bool(offset),
                )
            if written > expected_size:
                partial.unlink(missing_ok=True)
                raise CliError(f"Model size exceeded expected size for {target.name}: {written} bytes")
            last_error = CliError(f"Incomplete download for {target.name}: {written}/{expected_size} bytes")
        except CliError as exc:
            if "Invalid Range response" in str(exc):
                partial.unlink(missing_ok=True)
            if "SHA-256 mismatch" in str(exc) or "size exceeded" in str(exc):
                raise
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < 5:
            time.sleep(min(2 ** (attempt - 1), 16))
    raise CliError(f"Could not download {target.name} after 5 attempts: {last_error}") from last_error


def _cuda_architecture_from_cache() -> str | None:
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_CUDA_ARCHITECTURES:"):
            value = line.partition("=")[2]
            return value or None
    return None


def _cuda_architecture_compiled() -> str | None:
    compile_flags = BUILD_DIR / "ggml" / "src" / "ggml-cuda" / "CMakeFiles" / "ggml-cuda.dir" / "flags.make"
    configure_log = LOG_DIR / "configure.log"
    for path in (compile_flags, configure_log):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        architectures = list(dict.fromkeys(re.findall(r"compute_(\d+[a-z]?)", content)))
        if not architectures:
            architectures = list(dict.fromkeys(re.findall(r"Using CMAKE_CUDA_ARCHITECTURES=(\d+[a-z]?)", content)))
        if architectures:
            return ",".join(architectures)
    return None


def _toolchain_info() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    cache = BUILD_DIR / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("//") or line.startswith("#") or ":" not in line:
                continue
            name, _, value = line.partition(":")
            _, _, value = value.partition("=")
            if name in {
                "CMAKE_CUDA_COMPILER",
                "CMAKE_CUDA_COMPILER_VERSION",
                "CMAKE_CXX_COMPILER",
                "CMAKE_CXX_COMPILER_ID",
                "CMAKE_CXX_COMPILER_VERSION",
            }:
                values[name] = value or None
    try:
        values["cmake_version"] = _run_capture(["cmake", "--version"]).splitlines()[0]
    except (CliError, IndexError):
        values["cmake_version"] = None
    return values


def _build_manifest(model_results: dict[str, Any], model_source: str = DEFAULT_MODEL_SOURCE) -> dict[str, Any]:
    try:
        source = MODEL_SOURCES[model_source]
    except KeyError as exc:
        raise CliError(f"Unknown model source: {model_source}") from exc
    try:
        gpu_inventory = _gpu_inventory()
    except CliError as exc:
        gpu_inventory = {"available": False, "error": str(exc)}
    compiled_architecture = _cuda_architecture_compiled()
    return {
        "engine": {"url": ENGINE_URL, "commit": ENGINE_COMMIT},
        "model_source": {
            "name": model_source,
            "repo": source["repo"],
            "revision": source["revision"],
            "resolve_url_template": source["resolve_url_template"],
        },
        "models": model_results,
        "build": {
            "cmake_flags": BUILD_FLAGS,
            "cuda_architecture_requested": CUDA_ARCH_REQUESTED,
            "cuda_architecture_cache": _cuda_architecture_from_cache(),
            "cuda_architecture_configured": compiled_architecture,
            "cuda_architecture_compiled": compiled_architecture,
            "cuda_enabled": True,
            "toolchain": _toolchain_info(),
        },
        "device_inventory_at_build": gpu_inventory,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_command(args: argparse.Namespace) -> int:
    if args.jobs <= 0:
        raise CliError("--jobs must be a positive integer")
    for command in ("git", "cmake"):
        if shutil.which(command) is None:
            raise CliError(f"Required command not found: {command}")
    _prepare_engine()
    models = {key: _download_model(info, args.model_source) for key, info in MODEL_FILES.items()}

    configure = ["cmake", "-S", str(SOURCE_DIR), "-B", str(BUILD_DIR), *BUILD_FLAGS]
    _run_logged(configure, cwd=REPO_ROOT, log_path=LOG_DIR / "configure.log")
    build = ["cmake", "--build", str(BUILD_DIR), "--target", "voxcpm2-cli", "-j", str(args.jobs)]
    _run_logged(build, cwd=REPO_ROOT, log_path=LOG_DIR / "build.log")

    binary = BUILD_DIR / "bin" / "voxcpm2-cli"
    if not binary.is_file():
        raise CliError(f"Build completed but CLI executable is missing: {binary}")
    manifest = _build_manifest(models, args.model_source)
    manifest["build"]["cli_path"] = str(binary)
    _write_json(MANIFEST_PATH, manifest)
    print(f"Build manifest: {MANIFEST_PATH}")
    return 0


def _parse_gpu_inventory(output: str) -> list[dict[str, Any]]:
    devices = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) < 4:
            continue
        try:
            index = int(row[0].strip())
            memory_total = int(row[3].strip())
        except ValueError:
            continue
        devices.append(
            {"index": index, "uuid": row[1].strip(), "name": row[2].strip(), "memory_total_mib": memory_total}
        )
    return devices


def _gpu_inventory() -> list[dict[str, Any]]:
    output = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = _parse_gpu_inventory(output)
    if not devices:
        raise CliError("nvidia-smi returned no NVIDIA GPUs")
    return devices


def _cuda_driver_probe(uuid: str) -> dict[str, Any]:
    probe = r"""
import ctypes
import sys
try:
    driver = ctypes.CDLL("libcuda.so.1")
except OSError as exc:
    print(f"Cannot load CUDA driver library libcuda.so.1: {exc}", file=sys.stderr)
    raise SystemExit(2)

def check(status, operation):
    if status != 0:
        print(f"CUDA driver call {operation} failed with status {status}", file=sys.stderr)
        raise SystemExit(3)

check(driver.cuInit(0), "cuInit")
count = ctypes.c_int()
check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
if count.value != 1:
    print(f"CUDA driver expected one selected device, found {count.value}", file=sys.stderr)
    raise SystemExit(4)
device = ctypes.c_int()
check(driver.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
name = ctypes.create_string_buffer(256)
check(driver.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
print(f"visible_count={count.value}; device0={name.value.decode(errors='replace')}")
"""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = uuid
    try:
        output = _run_capture([sys.executable, "-c", probe], env=env).strip()
    except CliError as exc:
        raise CliError(f"CUDA driver initialization failed for {uuid}: {exc}") from exc
    return {"uuid": uuid, "probe": output}


def _select_gpu(index: int) -> dict[str, Any]:
    if index < 0:
        raise CliError("--gpu must be a non-negative physical GPU index")
    inventory = _gpu_inventory()
    selected = next((item for item in inventory if item["index"] == index), None)
    if selected is None:
        visible = ", ".join(f"{item['index']}:{item['name']}" for item in inventory)
        raise CliError(f"Physical GPU index {index} is unavailable. Detected: {visible}")
    driver = _cuda_driver_probe(selected["uuid"])
    return {**selected, "cuda_driver": driver}


def _gpu_snapshot(uuid: str) -> dict[str, Any] | None:
    try:
        output = _run_capture(
            [
                "nvidia-smi",
                "--id",
                uuid,
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    except CliError:
        return None
    row = next(csv.reader(output.splitlines(), skipinitialspace=True), [])
    if len(row) < 4:
        return None
    try:
        memory_used = int(row[0].strip())
        memory_total = int(row[1].strip())
    except ValueError:
        return None
    try:
        utilization = int(row[2].strip())
    except ValueError:
        utilization = None
    try:
        temperature = int(row[3].strip())
    except ValueError:
        temperature = None
    return {
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "utilization_percent": utilization,
        "temperature_c": temperature,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
    }


def _check_model(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CliError(f"{role} model is missing or empty: {path}")
    resolved = path.resolve()
    stat = resolved.stat()
    with resolved.open("rb") as model_file:
        if model_file.read(4) != b"GGUF":
            raise CliError(f"{role} model does not have a GGUF header: {resolved}")

    info = {"path": str(resolved), "size": stat.st_size, "sha256": None}
    role_key = "base" if role == "BaseLM" else "acoustic" if role == "Acoustic" else None
    expected = MODEL_FILES.get(role_key) if role_key else None
    if expected is not None and resolved.name == expected["name"]:
        if stat.st_size != expected["size"]:
            raise CliError(f"{role} model size is invalid: {resolved} ({stat.st_size} bytes)")
        if resolved.parent == MODEL_DIR.resolve() and MANIFEST_PATH.is_file():
            try:
                manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                record = manifest.get("models", {}).get(role_key, {})
            except (OSError, json.JSONDecodeError):
                record = {}
            if (
                record.get("sha256") == expected["sha256"]
                and record.get("size") == stat.st_size
                and record.get("mtime_ns") == stat.st_mtime_ns
                and Path(record.get("path", "")).resolve() == resolved
            ):
                info["sha256"] = expected["sha256"]
                return info
        actual_sha = _sha256_file(resolved)
        if actual_sha != expected["sha256"]:
            raise CliError(f"{role} model SHA-256 mismatch: expected {expected['sha256']}, got {actual_sha}")
        info["sha256"] = actual_sha
        return info

    info["sha256"] = _sha256_file(resolved)
    return info


def _is_standard_reference(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getcomptype() != "NONE"
                or audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getframerate() != 16000
            ):
                return False
            frames = audio.getnframes()
            if frames <= 0:
                return False
            data = audio.readframes(frames)
            return len(data) == frames * 2 and any(data)
    except (wave.Error, EOFError, OSError):
        return False


def _prepare_reference_audio(path: Path, artifact_dir: Path) -> tuple[Path, bool, Callable[[], None]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise CliError(f"Reference audio is missing or empty: {path}")
    if _is_standard_reference(path):
        return path.resolve(), False, lambda: None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise CliError("Reference audio must be mono 16 kHz PCM16 WAV, or ffmpeg must be installed for conversion")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix="voxcpm2-reference-", suffix=".wav", dir=artifact_dir)
    os.close(handle)
    converted = Path(temp_name)
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(converted),
    ]
    try:
        _run_capture(command)
        if not _is_standard_reference(converted):
            raise CliError(f"ffmpeg did not produce a valid mono PCM16 WAV: {converted}")
    except Exception:
        converted.unlink(missing_ok=True)
        raise
    return converted, True, lambda: converted.unlink(missing_ok=True)


def _build_engine_command(binary: Path, options: InferOptions, reference_path: Path | None) -> list[str]:
    command = [
        str(binary),
        "-t",
        options.text,
        "-o",
        str(options.output),
        "--n-gpu-layers",
        "999",
        "--timesteps",
        str(options.timesteps),
        "--cfg",
        str(options.cfg),
        "--steps",
        str(options.steps),
        "--seed",
        str(options.seed),
    ]
    if reference_path is not None:
        if options.reference_text is None:
            command.extend(["-r", str(reference_path)])
        else:
            command.extend(["--prompt-wav", str(reference_path), "--prompt-text", options.reference_text])
    command.extend([str(options.base_model), str(options.acoustic_model)])
    return command


def _parse_engine_log(log_text: str) -> dict[str, Any]:
    backend_match = re.search(r"custom component backend=([^\s]+)", log_text, re.IGNORECASE)
    offload_matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", log_text, re.IGNORECASE)
    elapsed_match = re.search(r"Elapsed:\s*([0-9.]+)s,\s*RTF=([0-9.]+)", log_text)
    audio_match = re.search(
        r"Audio:\s*([0-9.]+)s,\s*(\d+) samples,\s*(\d+) Hz,\s*RMS=([0-9.]+),\s*peak=([0-9.]+)",
        log_text,
    )
    offload = [{"offloaded": int(item[0]), "total": int(item[1])} for item in offload_matches]
    return {
        "backend": backend_match.group(1) if backend_match else None,
        "gpu_layers": offload,
        "elapsed_seconds": float(elapsed_match.group(1)) if elapsed_match else None,
        "rtf": float(elapsed_match.group(2)) if elapsed_match else None,
        "audio_log": (
            {
                "duration_seconds": float(audio_match.group(1)),
                "samples": int(audio_match.group(2)),
                "sample_rate": int(audio_match.group(3)),
                "rms": float(audio_match.group(4)),
                "peak": float(audio_match.group(5)),
            }
            if audio_match
            else None
        ),
    }


def _has_cpu_fallback(log_text: str) -> bool:
    return (
        re.search(
            r"falling back to CPU|(?:custom component )?backend\s*=\s*CPU(?:\d+)?\b",
            log_text,
            re.IGNORECASE,
        )
        is not None
    )


def _audit_cuda(log_text: str, parsed: dict[str, Any]) -> None:
    backend = parsed["backend"] or ""
    if not backend.upper().startswith("CUDA"):
        if _has_cpu_fallback(log_text):
            raise CliError("Engine fell back to CPU. CUDA inference is required; see the run log.")
        raise CliError("Could not confirm a CUDA custom component backend in the engine log.")
    layers = parsed["gpu_layers"]
    if not layers or any(item["total"] <= 0 or item["offloaded"] != item["total"] for item in layers):
        raise CliError("CUDA inference requires every BaseLM layer to be offloaded to a GPU; see the run log.")


def _inspect_wav(path: Path, *, required_sample_rate: int | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise CliError(f"Output WAV is missing or empty: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if audio.getcomptype() != "NONE":
                raise CliError(f"Output WAV is compressed: {path}")
            if channels != 1 or sample_width != 2 or frame_count <= 0:
                raise CliError(f"Output WAV must contain mono PCM16 samples: {path}")
            if required_sample_rate is not None and sample_rate != required_sample_rate:
                raise CliError(f"Output WAV sample rate must be {required_sample_rate} Hz, got {sample_rate} Hz")
            peak = 0
            sum_squares = 0
            sample_count = 0
            while data := audio.readframes(65536):
                usable = len(data) - (len(data) % 2)
                for (sample,) in struct.iter_unpack("<h", data[:usable]):
                    magnitude = abs(sample)
                    peak = max(peak, magnitude)
                    sum_squares += sample * sample
                    sample_count += 1
            if sample_count != frame_count:
                raise CliError(f"Output WAV frame count is inconsistent: expected {frame_count}, read {sample_count}")
            rms = (sum_squares / sample_count) ** 0.5 / 32768.0
            if rms <= 1e-5 or peak == 0:
                raise CliError(f"Output WAV appears silent: {path}")
            duration = frame_count / sample_rate
            return {
                "path": str(path.resolve()),
                "channels": channels,
                "sample_width_bits": sample_width * 8,
                "sample_rate": sample_rate,
                "frames": frame_count,
                "duration_seconds": duration,
                "rms": rms,
                "peak": peak / 32768.0,
                "finite": True,
                "non_silent": True,
            }
    except (wave.Error, EOFError, OSError) as exc:
        raise CliError(f"Output is not a decodable PCM WAV file: {path}: {exc}") from exc


def _run_engine(
    *,
    command: Sequence[str],
    options: InferOptions,
    gpu: dict[str, Any],
    log_path: Path,
    offline: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    build_bin = str(BUILD_DIR / "bin")
    existing_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = build_bin + (os.pathsep + existing_library_path if existing_library_path else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = []
    before = _gpu_snapshot(gpu["uuid"])
    if before:
        snapshots.append(before)
    preexec: Callable[[], None] | None = _install_offline_seccomp if offline else None
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=preexec,
            )
            while True:
                try:
                    return_code = process.wait(timeout=0.75)
                    break
                except subprocess.TimeoutExpired:
                    if _has_cpu_fallback(log_path.read_text(encoding="utf-8", errors="replace")):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise CliError(
                            f"Engine selected the CPU backend; inference was terminated to prevent CPU fallback. "
                            f"See {log_path}"
                        )
                    snapshot = _gpu_snapshot(gpu["uuid"])
                    if snapshot:
                        snapshots.append(snapshot)
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if offline:
            raise CliError(f"Could not start offline inference with seccomp isolation: {exc}") from exc
        raise CliError(f"Could not start inference engine: {exc}") from exc
    wall_seconds = time.perf_counter() - started
    after = _gpu_snapshot(gpu["uuid"])
    if after:
        snapshots.append(after)
    if return_code:
        failed_log = log_path.read_text(encoding="utf-8", errors="replace")
        if _has_cpu_fallback(failed_log):
            raise CliError(f"Engine fell back to CPU and exited with code {return_code}; see {log_path}")
        raise CliError(f"Engine exited with code {return_code}; see {log_path}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_engine_log(log_text)
    _audit_cuda(log_text, parsed)
    wav = _inspect_wav(options.output, required_sample_rate=48000)
    if parsed["elapsed_seconds"] is None or parsed["rtf"] is None:
        raise CliError(f"Could not parse engine elapsed time and RTF from {log_path}")
    if parsed["audio_log"] is None:
        raise CliError(f"Could not parse engine audio statistics from {log_path}")
    peak_memory = max((item["memory_used_mib"] for item in snapshots), default=None)
    return {
        "wall_seconds": wall_seconds,
        "engine_seconds": parsed["elapsed_seconds"],
        "rtf_engine": parsed["rtf"],
        "rtf_wall": wall_seconds / wav["duration_seconds"],
        "wav": wav,
        "engine_log": parsed,
        "gpu_observations": snapshots,
        "gpu_peak_memory_used_mib": peak_memory,
        "log_path": str(log_path.resolve()),
        "offline_enforced": offline,
    }


def _manifest_summary() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {
            "engine_commit": ENGINE_COMMIT,
            "model_source": MODEL_SOURCES[DEFAULT_MODEL_SOURCE],
            "model_sources": MODEL_REPO,
            "manifest_path": None,
        }
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CliError(f"Build manifest is invalid: {MANIFEST_PATH}: {exc}") from exc
    commit = manifest.get("engine", {}).get("commit")
    if commit != ENGINE_COMMIT:
        raise CliError(f"Build manifest engine commit mismatch: expected {ENGINE_COMMIT}, got {commit}")
    model_source = manifest.get("model_source", {})
    return {
        "engine_commit": commit,
        "model_source": model_source,
        "model_sources": model_source.get("repo", MODEL_REPO),
        "model_revision": model_source.get("revision", MODEL_REVISION),
        "models": manifest.get("models", {}),
        "build": manifest.get("build", {}),
        "manifest_path": str(MANIFEST_PATH),
    }


def _validate_infer_options(options: InferOptions) -> None:
    if not options.text.strip():
        raise CliError("--text must not be empty")
    if options.reference_text is not None and options.reference_audio is None:
        raise CliError("--reference-text requires --reference-audio")
    if options.reference_text is not None and not options.reference_text.strip():
        raise CliError("--reference-text must not be empty when provided")
    if options.timesteps <= 0 or options.steps <= 0:
        raise CliError("--timesteps and --steps must be positive integers")
    if not math.isfinite(options.cfg) or options.cfg <= 0:
        raise CliError("--cfg must be a finite positive number")
    output = options.output.expanduser().resolve()
    protected_paths = {options.base_model.expanduser().resolve(), options.acoustic_model.expanduser().resolve()}
    if options.reference_audio is not None:
        protected_paths.add(options.reference_audio.expanduser().resolve())
    if output in protected_paths:
        raise CliError("Output WAV path must not overwrite a model file or reference audio")


def _infer_once(
    options: InferOptions,
    *,
    gpu: dict[str, Any],
    artifact_dir: Path,
    offline: bool = False,
) -> dict[str, Any]:
    _validate_infer_options(options)
    build_summary = _manifest_summary()
    base_model = _check_model(options.base_model, "BaseLM")
    acoustic_model = _check_model(options.acoustic_model, "Acoustic")
    binary = BUILD_DIR / "bin" / "voxcpm2-cli"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise CliError(f"VoxCPM2 CLI is missing or not executable: {binary}; run the build subcommand first")

    output = options.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    options = InferOptions(
        text=options.text,
        output=output,
        gpu=options.gpu,
        reference_audio=options.reference_audio,
        reference_text=options.reference_text,
        base_model=options.base_model.resolve(),
        acoustic_model=options.acoustic_model.resolve(),
        seed=options.seed,
        timesteps=options.timesteps,
        cfg=options.cfg,
        steps=options.steps,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = output.stem[:48] or "audio"
    run_id = f"{_timestamp()}_{stem}"
    log_path = artifact_dir / f"{run_id}.log"
    metadata_path = artifact_dir / f"{run_id}.json"
    temp_handle, temp_name = tempfile.mkstemp(prefix=f".{run_id}_", suffix=".wav", dir=artifact_dir)
    os.close(temp_handle)
    pending_output = Path(temp_name)
    pending_output.unlink()
    engine_options = InferOptions(
        text=options.text,
        output=pending_output,
        gpu=options.gpu,
        reference_audio=options.reference_audio,
        reference_text=options.reference_text,
        base_model=options.base_model,
        acoustic_model=options.acoustic_model,
        seed=options.seed,
        timesteps=options.timesteps,
        cfg=options.cfg,
        steps=options.steps,
    )
    temp_cleanup: Callable[[], None] = lambda: None
    converted = False
    prepared_reference: Path | None = None
    try:
        if options.reference_audio is not None:
            prepared_reference, converted, temp_cleanup = _prepare_reference_audio(
                options.reference_audio.expanduser().resolve(), artifact_dir
            )
        command = _build_engine_command(binary, engine_options, prepared_reference)
        print(f"GPU {gpu['index']}: {gpu['name']} ({gpu['uuid']})", flush=True)
        print(f"Output: {output}\nLog: {log_path}", flush=True)
        result = _run_engine(command=command, options=engine_options, gpu=gpu, log_path=log_path, offline=offline)
        _publish_output(pending_output, output)
        result["wav"]["path"] = str(output)
        result.update(
            {
                "text": options.text,
                "gpu": gpu,
                "reference_audio": str(options.reference_audio.resolve()) if options.reference_audio else None,
                "reference_text": options.reference_text,
                "reference_audio_converted": converted,
                "models": {"base": base_model, "acoustic": acoustic_model},
                "output_path": str(output),
                "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "build": build_summary,
                "command": command,
                "metadata_path": str(metadata_path.resolve()),
            }
        )
        _write_json(metadata_path, result)
        return result
    finally:
        pending_output.unlink(missing_ok=True)
        temp_cleanup()


def _publish_output(pending_output: Path, output: Path) -> None:
    try:
        pending_output.replace(output)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    handle, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(handle)
    staging = Path(temp_name)
    try:
        shutil.copyfile(pending_output, staging)
        staging.replace(output)
    finally:
        staging.unlink(missing_ok=True)


def _options_from_args(args: argparse.Namespace) -> InferOptions:
    return InferOptions(
        text=args.text,
        output=Path(args.output).expanduser(),
        gpu=args.gpu,
        reference_audio=Path(args.reference_audio).expanduser() if args.reference_audio else None,
        reference_text=args.reference_text,
        base_model=Path(args.base_model).expanduser(),
        acoustic_model=Path(args.acoustic_model).expanduser(),
        seed=args.seed,
        timesteps=args.timesteps,
        cfg=args.cfg,
        steps=args.steps,
    )


def infer_command(args: argparse.Namespace) -> int:
    gpu = _select_gpu(args.gpu)
    result = _infer_once(_options_from_args(args), gpu=gpu, artifact_dir=RUN_DIR)
    print(
        f"Done: {result['wav']['duration_seconds']:.2f}s audio in {result['wall_seconds']:.2f}s wall time "
        f"(engine {result['engine_seconds']:.2f}s, RTF {result['rtf_engine']:.3f})."
    )
    print(f"Metadata: {result['metadata_path']}")
    return 0


def _seccomp_filter_instructions() -> list[tuple[int, int, int, int]]:
    """Return classic-BPF instructions denying non-UNIX sockets and io_uring."""
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise CliError("--offline seccomp enforcement is supported on x86_64 Linux only")
    audit_arch_x86_64 = 0xC000003E
    seccomp_ret_kill_process = 0x80000000
    seccomp_ret_allow = 0x7FFF0000
    seccomp_ret_errno = 0x00050000 | 1  # EPERM
    bpf_ld_w_abs = 0x20
    bpf_jmp_jeq_k = 0x15
    bpf_jmp_jset_k = 0x45
    bpf_ret_k = 0x06

    def bpf_stmt(code: int, value: int) -> tuple[int, int, int, int]:
        return (code, 0, 0, value)

    def bpf_jump(value: int, jt: int, jf: int) -> tuple[int, int, int, int]:
        return (bpf_jmp_jeq_k, jt, jf, value)

    def bpf_jump_set(value: int, jt: int, jf: int) -> tuple[int, int, int, int]:
        return (bpf_jmp_jset_k, jt, jf, value)

    return [
        bpf_stmt(bpf_ld_w_abs, 4),  # seccomp_data.arch
        bpf_jump(audit_arch_x86_64, 1, 0),
        bpf_stmt(bpf_ret_k, seccomp_ret_kill_process),
        bpf_stmt(bpf_ld_w_abs, 0),  # seccomp_data.nr
        bpf_jump_set(0x40000000, 0, 1),  # reject x32 syscall ABI
        bpf_stmt(bpf_ret_k, seccomp_ret_kill_process),
        bpf_jump(41, 0, 3),  # socket -> inspect family; other syscall -> socketpair check
        bpf_stmt(bpf_ld_w_abs, 16),  # args[0], socket domain
        bpf_jump(1, 7, 0),  # AF_UNIX -> skip denial; all other families -> EPERM
        bpf_stmt(bpf_ret_k, seccomp_ret_errno),
        bpf_jump(53, 0, 3),  # socketpair -> inspect family; otherwise -> io_uring check
        bpf_stmt(bpf_ld_w_abs, 16),  # args[0], socketpair domain
        bpf_jump(1, 1, 0),  # AF_UNIX -> continue; all other families -> EPERM
        bpf_stmt(bpf_ret_k, seccomp_ret_errno),
        bpf_jump(425, 0, 1),  # io_uring_setup can create sockets outside socket(2)
        bpf_stmt(bpf_ret_k, seccomp_ret_errno),
        bpf_stmt(bpf_ret_k, seccomp_ret_allow),
    ]


def _probe_offline_seccomp() -> None:
    probe = r"""
import errno
import socket

for family in (socket.AF_INET, socket.AF_INET6):
    try:
        socket.socket(family, socket.SOCK_STREAM)
    except OSError as exc:
        if exc.errno != errno.EPERM:
            raise SystemExit(f"network socket was blocked with unexpected errno {exc.errno}")
    else:
        raise SystemExit(f"network socket family {family} was not blocked")

local_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
local_socket.close()
left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
left.close()
right.close()
print("seccomp blocks AF_INET/AF_INET6 and preserves AF_UNIX sockets")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=_install_offline_seccomp,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CliError(f"Could not install the Linux seccomp network filter: {exc}") from exc
    if result.returncode:
        raise CliError(f"Seccomp network filter probe failed: {result.stdout.strip()}")
    print(result.stdout.strip(), flush=True)


def _install_offline_seccomp() -> None:
    instructions = _seccomp_filter_instructions()

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    filters = (SockFilter * len(instructions))(*(SockFilter(*item) for item in instructions))
    program = SockFprog(len(instructions), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(38, 1, None, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(22, 2, ctypes.cast(ctypes.pointer(program), ctypes.c_void_p), 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _make_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VoxCPM2 CUDA Acceptance Report",
        "",
        f"Created: {report['created_utc']}",
        f"Engine commit: `{report['build'].get('engine_commit', ENGINE_COMMIT)}`",
        f"GPU: {report['gpu']['index']} — {report['gpu']['name']} ({report['gpu']['uuid']})",
        f"Offline network filter: {'enabled' if report['offline_enforced'] else 'not requested'}",
        "",
        "| Case | Run | Wall (s) | Engine (s) | Audio (s) | RTF | Peak GPU memory (MiB) | WAV |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["runs"]:
        if item.get("status") != "passed":
            lines.append(f"| {item['case']} | {item['run']} | failed | — | — | — | — | " f"{item.get('error', '')} |")
            continue
        lines.append(
            f"| {item['case']} | {item['run']} | {item['wall_seconds']:.2f} | {item['engine_seconds']:.2f} | "
            f"{item['audio_seconds']:.2f} | {item['rtf']:.3f} | {item['gpu_peak_memory_used_mib']} | "
            f"`{item['wav_path']}` |"
        )
    lines.extend(["", f"Result: **{report['status'].upper()}**", ""])
    return "\n".join(lines)


def test_command(args: argparse.Namespace) -> int:
    if args.runs <= 0:
        raise CliError("--runs must be a positive integer")
    if args.offline:
        _seccomp_filter_instructions()
        _probe_offline_seccomp()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(REPO_ROOT / "output")
    except ValueError as exc:
        raise CliError("--output-dir must be under the repository's ignored output/ directory") from exc
    reference = REPO_ROOT / "examples" / "reference_speaker.wav"
    if not reference.is_file() or reference.stat().st_size <= 0:
        raise CliError(f"Acceptance reference audio is missing or empty: {reference}")
    gpu = _select_gpu(args.gpu)
    _check_model(MODEL_DIR / MODEL_FILES["base"]["name"], "BaseLM")
    _check_model(MODEL_DIR / MODEL_FILES["acoustic"]["name"], "Acoustic")
    binary = BUILD_DIR / "bin" / "voxcpm2-cli"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise CliError(f"VoxCPM2 CLI is missing or not executable: {binary}; run the build subcommand first")

    cases = [
        ("english_tts", TEST_TEXTS["english"], None),
        ("chinese_tts", TEST_TEXTS["chinese"], None),
        ("english_clone", TEST_TEXTS["english"], reference),
        ("chinese_clone", TEST_TEXTS["chinese"], reference),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for case_name, text, reference_audio in cases:
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        for run_number in range(1, args.runs + 1):
            output = case_dir / f"run-{run_number:02d}.wav"
            options = InferOptions(
                text=text,
                output=output,
                gpu=args.gpu,
                reference_audio=reference_audio,
            )
            try:
                result = _infer_once(options, gpu=gpu, artifact_dir=case_dir, offline=args.offline)
                runs.append(
                    {
                        "case": case_name,
                        "run": run_number,
                        "status": "passed",
                        "wall_seconds": result["wall_seconds"],
                        "engine_seconds": result["engine_seconds"],
                        "audio_seconds": result["wav"]["duration_seconds"],
                        "rtf": result["rtf_engine"],
                        "rtf_wall": result["rtf_wall"],
                        "gpu_peak_memory_used_mib": result["gpu_peak_memory_used_mib"],
                        "gpu_observations": result["gpu_observations"],
                        "wav_path": result["wav"]["path"],
                        "metadata_path": result["metadata_path"],
                        "log_path": result["log_path"],
                        "first_run_in_case": run_number == 1,
                        "offline_enforced": result["offline_enforced"],
                    }
                )
                print(
                    f"{case_name} run {run_number}/{args.runs}: {result['wall_seconds']:.2f}s wall, "
                    f"{result['engine_seconds']:.2f}s engine, {result['wav']['duration_seconds']:.2f}s audio, "
                    f"RTF {result['rtf_engine']:.3f}",
                    flush=True,
                )
            except (CliError, OSError, wave.Error) as exc:
                runs.append({"case": case_name, "run": run_number, "status": "failed", "error": str(exc)})
                report = _create_acceptance_report(args, output_dir, gpu, runs, status="failed")
                _write_json(output_dir / "report.json", report)
                (output_dir / "report.md").write_text(_make_report_markdown(report), encoding="utf-8")
                raise CliError(f"Acceptance failed at {case_name} run {run_number}: {exc}") from exc

    report = _create_acceptance_report(args, output_dir, gpu, runs, status="passed")
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_make_report_markdown(report), encoding="utf-8")
    print(f"Acceptance report: {output_dir / 'report.md'}")
    return 0


def _create_acceptance_report(
    args: argparse.Namespace,
    output_dir: Path,
    gpu: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    try:
        build_summary = _manifest_summary()
    except CliError as exc:
        build_summary = {"error": str(exc), "manifest_path": str(MANIFEST_PATH)}
    return {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "runs_requested_per_case": args.runs,
        "offline_enforced": bool(args.offline),
        "offline_mechanism": "Linux x86_64 seccomp: AF_UNIX only; blocks io_uring_setup" if args.offline else None,
        "texts": TEST_TEXTS,
        "reference_audio": str((REPO_ROOT / "examples" / "reference_speaker.wav").resolve()),
        "build": build_summary,
        "output_dir": str(output_dir),
        "gpu_memory_scope": "whole selected physical GPU; includes other processes",
        "runs": runs,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _add_infer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", required=True, help="text to synthesize")
    parser.add_argument("--output", required=True, help="output WAV path")
    parser.add_argument("--gpu", type=_non_negative_int, default=DEFAULT_GPU, help="physical nvidia-smi GPU index")
    parser.add_argument("--reference-audio", help="optional reference audio for voice cloning")
    parser.add_argument("--reference-text", help="exact transcript for continuation-style voice cloning")
    parser.add_argument("--base-model", default=str(MODEL_DIR / MODEL_FILES["base"]["name"]))
    parser.add_argument("--acoustic-model", default=str(MODEL_DIR / MODEL_FILES["acoustic"]["name"]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=_positive_int, default=10)
    parser.add_argument("--cfg", type=_positive_finite_float, default=2.0)
    parser.add_argument("--steps", type=_positive_int, default=200)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run VoxCPM2 with a pinned llama.cpp-omni CUDA engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="download pinned sources/models and build the CUDA CLI")
    build_parser.add_argument("--jobs", type=_positive_int, default=24, help="parallel build jobs (default: 24)")
    build_parser.add_argument(
        "--model-source",
        choices=tuple(MODEL_SOURCES),
        default=DEFAULT_MODEL_SOURCE,
        help="pinned model mirror to download from (default: huggingface)",
    )

    infer_parser = subparsers.add_parser("infer", help="synthesize one WAV file")
    _add_infer_arguments(infer_parser)

    test_parser = subparsers.add_parser("test", help="run fixed Chinese/English synthesis and cloning acceptance")
    test_parser.add_argument("--gpu", type=_non_negative_int, default=DEFAULT_GPU, help="physical nvidia-smi GPU index")
    test_parser.add_argument("--runs", type=_positive_int, default=3, help="runs per test case")
    test_parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "acceptance"),
        help="directory for WAVs, logs, metadata, and reports",
    )
    test_parser.add_argument(
        "--offline",
        action="store_true",
        help="enforce Linux seccomp network blocking for every engine process",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return build_command(args)
        if args.command == "infer":
            return infer_command(args)
        if args.command == "test":
            return test_command(args)
        parser.error(f"unknown command: {args.command}")
    except (CliError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
