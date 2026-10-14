"""Regression tests for the standalone VoxCPM2 CUDA wrapper."""

import importlib.util
import struct
import sys
import wave
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "voxcpm2_cuda.py"
SPEC = importlib.util.spec_from_file_location("voxcpm2_cuda", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
voxcpm2_cuda = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = voxcpm2_cuda
SPEC.loader.exec_module(voxcpm2_cuda)


def test_validate_infer_options_rejects_blank_text_and_unpaired_reference_text(tmp_path):
    options = voxcpm2_cuda.InferOptions(text="   ", output=tmp_path / "blank.wav")
    with pytest.raises(voxcpm2_cuda.CliError, match="--text must not be empty"):
        voxcpm2_cuda._validate_infer_options(options)

    options = voxcpm2_cuda.InferOptions(
        text="hello",
        output=tmp_path / "unpaired.wav",
        reference_text="reference transcript",
    )
    with pytest.raises(voxcpm2_cuda.CliError, match="requires --reference-audio"):
        voxcpm2_cuda._validate_infer_options(options)


def test_validate_infer_options_rejects_non_finite_or_non_positive_cfg(tmp_path):
    for cfg in (float("nan"), float("inf"), 0.0, -1.0):
        options = voxcpm2_cuda.InferOptions(text="hello", output=tmp_path / "output.wav", cfg=cfg)
        with pytest.raises(voxcpm2_cuda.CliError, match="finite positive"):
            voxcpm2_cuda._validate_infer_options(options)


def test_validate_infer_options_prevents_overwriting_models_or_reference(tmp_path):
    model = tmp_path / "base.gguf"
    options = voxcpm2_cuda.InferOptions(
        text="hello",
        output=model,
        base_model=model,
        acoustic_model=tmp_path / "acoustic.gguf",
    )
    with pytest.raises(voxcpm2_cuda.CliError, match="must not overwrite"):
        voxcpm2_cuda._validate_infer_options(options)

    reference = tmp_path / "reference.wav"
    options = voxcpm2_cuda.InferOptions(text="hello", output=reference, reference_audio=reference)
    with pytest.raises(voxcpm2_cuda.CliError, match="must not overwrite"):
        voxcpm2_cuda._validate_infer_options(options)


def test_build_engine_command_selects_expected_clone_modes(tmp_path):
    binary = tmp_path / "voxcpm2-cli"
    reference = tmp_path / "reference.wav"
    base_model = tmp_path / "base.gguf"
    acoustic_model = tmp_path / "acoustic.gguf"
    audio_only = voxcpm2_cuda.InferOptions(
        text="spoken words",
        output=tmp_path / "audio.wav",
        reference_audio=reference,
        base_model=base_model,
        acoustic_model=acoustic_model,
    )
    command = voxcpm2_cuda._build_engine_command(binary, audio_only, reference)
    assert command[command.index("-r") + 1] == str(reference)
    assert "--prompt-wav" not in command
    assert command[-2:] == [str(base_model), str(acoustic_model)]
    assert command[command.index("--n-gpu-layers") + 1] == "999"

    with_transcript = voxcpm2_cuda.InferOptions(
        text="spoken words",
        output=tmp_path / "continued.wav",
        reference_audio=reference,
        reference_text="exact transcript",
        base_model=base_model,
        acoustic_model=acoustic_model,
    )
    command = voxcpm2_cuda._build_engine_command(binary, with_transcript, reference)
    assert command[command.index("--prompt-wav") + 1] == str(reference)
    assert command[command.index("--prompt-text") + 1] == "exact transcript"
    assert "-r" not in command


def test_parse_and_audit_engine_cuda_log():
    log = """VoxCPM2Runtime: custom component backend=CUDA0
llama_model_load: offloaded 32/32 layers to GPU
Audio: 1.25s, 60000 samples, 48000 Hz, RMS=0.120, peak=0.500
Elapsed: 2.500s, RTF=2.000
"""
    parsed = voxcpm2_cuda._parse_engine_log(log)
    assert parsed["backend"] == "CUDA0"
    assert parsed["gpu_layers"] == [{"offloaded": 32, "total": 32}]
    assert parsed["elapsed_seconds"] == 2.5
    assert parsed["rtf"] == 2.0
    assert parsed["audio_log"]["sample_rate"] == 48000
    voxcpm2_cuda._audit_cuda(log, parsed)


def test_audit_cuda_requires_every_base_layer_on_gpu():
    log = "VoxCPM2Runtime: custom component backend=CUDA0\nllama_model_load: offloaded 28/29 layers to GPU\n"
    parsed = voxcpm2_cuda._parse_engine_log(log)
    with pytest.raises(voxcpm2_cuda.CliError, match="every BaseLM layer"):
        voxcpm2_cuda._audit_cuda(log, parsed)


def test_audit_engine_rejects_cpu_fallback():
    log = "VoxCPM2Runtime: no GPU backend found, falling back to CPU\nbackend=CPU"
    with pytest.raises(voxcpm2_cuda.CliError, match="fell back to CPU"):
        voxcpm2_cuda._audit_cuda(log, voxcpm2_cuda._parse_engine_log(log))


def test_run_engine_terminates_immediately_on_cpu_backend(tmp_path, monkeypatch):
    log_path = tmp_path / "cpu-fallback.log"
    processes = []

    class FakeProcess:
        def __init__(self, *_args, stdout, **_kwargs):
            self.returncode = None
            self.terminated = False
            stdout.write("VoxCPM2Runtime: custom component backend=CPU\n")
            stdout.flush()
            processes.append(self)

        def wait(self, timeout=None):
            if self.returncode is None:
                raise voxcpm2_cuda.subprocess.TimeoutExpired("engine", timeout)
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def poll(self):
            return self.returncode

    monkeypatch.setattr(voxcpm2_cuda.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(voxcpm2_cuda, "_gpu_snapshot", lambda _uuid: None)
    options = voxcpm2_cuda.InferOptions(text="hello", output=tmp_path / "output.wav")

    with pytest.raises(voxcpm2_cuda.CliError, match="terminated to prevent CPU fallback"):
        voxcpm2_cuda._run_engine(
            command=["fake-engine"],
            options=options,
            gpu={"uuid": "GPU-test"},
            log_path=log_path,
            offline=False,
        )

    assert processes[0].terminated is True


def test_gpu_inventory_parser_uses_physical_index_and_uuid():
    devices = voxcpm2_cuda._parse_gpu_inventory("0, GPU-first, RTX A, 24564\n1, GPU-second, RTX 5090, 32607\n")
    assert devices == [
        {"index": 0, "uuid": "GPU-first", "name": "RTX A", "memory_total_mib": 24564},
        {"index": 1, "uuid": "GPU-second", "name": "RTX 5090", "memory_total_mib": 32607},
    ]


def test_model_check_requires_gguf_header(tmp_path):
    model = tmp_path / "custom.gguf"
    model.write_bytes(b"not a model")
    with pytest.raises(voxcpm2_cuda.CliError, match="GGUF header"):
        voxcpm2_cuda._check_model(model, "Custom")


def test_reference_audio_accepts_standard_mono_pcm16_without_conversion(tmp_path):
    reference = tmp_path / "reference.wav"
    with wave.open(str(reference), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<4h", 0, 200, -200, 0))

    prepared, converted, cleanup = voxcpm2_cuda._prepare_reference_audio(reference, tmp_path / "artifacts")
    try:
        assert prepared == reference.resolve()
        assert converted is False
    finally:
        cleanup()


def test_inspect_wav_requires_48khz_finite_non_silent_pcm16(tmp_path):
    output = tmp_path / "generated.wav"
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(struct.pack("<4h", 1000, -1000, 2000, -2000))

    info = voxcpm2_cuda._inspect_wav(output, required_sample_rate=48000)
    assert info["sample_rate"] == 48000
    assert info["non_silent"] is True
    assert info["finite"] is True


def test_inspect_wav_rejects_silent_audio(tmp_path):
    output = tmp_path / "silent.wav"
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(struct.pack("<8h", *([0] * 8)))

    with pytest.raises(voxcpm2_cuda.CliError, match="appears silent"):
        voxcpm2_cuda._inspect_wav(output, required_sample_rate=48000)


def test_offline_filter_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(voxcpm2_cuda.platform, "system", lambda: "Windows")
    with pytest.raises(voxcpm2_cuda.CliError, match="x86_64 Linux only"):
        voxcpm2_cuda._seccomp_filter_instructions()


def test_model_source_urls_and_build_parser_are_pinned():
    info = {"name": "VoxCPM2-BaseLM-F16.gguf"}
    assert voxcpm2_cuda._model_download_url(info, "huggingface") == (
        "https://huggingface.co/DennisHuang648/VoxCPM2-GGUF/resolve/"
        "169f64d8b98bbaab1761e4ca3a83e6af653456cc/VoxCPM2-BaseLM-F16.gguf?download=true"
    )
    assert voxcpm2_cuda._model_download_url(info, "modelscope") == (
        "https://modelscope.cn/models/DennisHuang/VoxCPM2-GGUF/resolve/"
        "bb319272866df0419de6a2357c9e675421b2688e/VoxCPM2-BaseLM-F16.gguf"
    )
    assert voxcpm2_cuda.make_parser().parse_args(["build"]).model_source == "huggingface"
    modelscope_args = voxcpm2_cuda.make_parser().parse_args(["build", "--model-source", "modelscope"])
    assert modelscope_args.model_source == "modelscope"


def test_model_download_resumes_partial_range_and_checks_digest(tmp_path, monkeypatch):
    import hashlib

    payload = b"GGUF" + bytes(range(64))
    split_at = 19
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    partial = model_dir / "tiny.gguf.part"
    partial.write_bytes(payload[:split_at])
    monkeypatch.setattr(voxcpm2_cuda, "MODEL_DIR", model_dir)
    requested_ranges = []

    class FakeResponse:
        status = 206

        def __init__(self):
            self.headers = {"Content-Range": f"bytes {split_at}-{len(payload) - 1}/{len(payload)}"}
            self.remaining = payload[split_at:]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return self.status

        def read(self, size):
            chunk, self.remaining = self.remaining[:size], self.remaining[size:]
            return chunk

    def fake_urlopen(request, timeout):
        assert timeout == 120
        requested_ranges.append(request.get_header("Range"))
        return FakeResponse()

    monkeypatch.setattr(voxcpm2_cuda, "urlopen", fake_urlopen)
    info = {
        "name": "tiny.gguf",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    result = voxcpm2_cuda._download_model(info)

    assert requested_ranges == [f"bytes={split_at}-"]
    assert (model_dir / "tiny.gguf").read_bytes() == payload
    assert not partial.exists()
    assert result["sha256"] == info["sha256"]
