from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def web_app(monkeypatch):
    """Exercise the real Gradio configuration without importing or downloading models."""
    voxcpm = types.ModuleType("voxcpm")
    voxcpm.VoxCPM = types.SimpleNamespace(from_pretrained=Mock(return_value=object()))
    funasr = types.ModuleType("funasr")
    funasr.AutoModel = Mock(return_value=object())
    utils = types.ModuleType("voxcpm.model.utils")
    utils.resolve_runtime_device = Mock(side_effect=lambda device, configured: device)
    monkeypatch.setitem(sys.modules, "voxcpm", voxcpm)
    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setitem(sys.modules, "voxcpm.model.utils", utils)
    spec = importlib.util.spec_from_file_location("web_demo_test_app", Path(__file__).resolve().parents[1] / "app.py")
    app = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, app)
    spec.loader.exec_module(app)
    return app


@pytest.fixture
def local_models(tmp_path):
    filenames = {
        "model_id": ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors", "audiovae.pth"),
        "asr_model_id": (
            "configuration.json",
            "config.yaml",
            "model.pt",
            "am.mvn",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ),
        "zipenhancer_model_id": ("configuration.json", "pytorch_model.bin"),
    }
    locations = {}
    for argument, files in filenames.items():
        directory = tmp_path / argument
        directory.mkdir()
        for filename in files:
            (directory / filename).write_text("{}")
        locations[argument] = str(directory)
    return locations


def test_local_model_paths_and_device_reach_loaders(web_app, local_models):
    demo = web_app.VoxCPMDemo(**local_models, device="cuda:1", local_files_only=True)
    tts = demo.get_or_load_voxcpm()
    asr = demo.get_or_load_asr_model()
    assert demo.get_or_load_voxcpm() is tts
    assert demo.get_or_load_asr_model() is asr
    web_app.voxcpm.VoxCPM.from_pretrained.assert_called_once_with(
        local_models["model_id"],
        zipenhancer_model_id=local_models["zipenhancer_model_id"],
        local_files_only=True,
        optimize=True,
        device="cuda:1",
    )
    web_app.AutoModel.assert_called_once_with(
        model=local_models["asr_model_id"],
        disable_update=True,
        check_latest=False,
        log_level="DEBUG",
        device="cuda:1",
    )


@pytest.mark.parametrize("argument", ["model_id", "asr_model_id", "zipenhancer_model_id"])
def test_offline_missing_directory_never_calls_a_loader(web_app, local_models, argument):
    local_models[argument] += "-missing"
    with pytest.raises(ValueError, match="existing local model directory"):
        web_app.VoxCPMDemo(**local_models, device="cuda", local_files_only=True)
    web_app.voxcpm.VoxCPM.from_pretrained.assert_not_called()
    web_app.AutoModel.assert_not_called()
    web_app.resolve_runtime_device.assert_not_called()


def test_partial_download_is_rejected_before_model_loading(web_app, local_models):
    weight = Path(local_models["model_id"]) / "model.safetensors"
    weight.rename(weight.with_suffix(".safetensors.incomplete"))
    with pytest.raises(ValueError, match="missing local model file"):
        web_app.VoxCPMDemo(**local_models, device="cuda", local_files_only=True)
    web_app.voxcpm.VoxCPM.from_pretrained.assert_not_called()


def test_default_online_model_ids_remain_available(web_app):
    demo = web_app.VoxCPMDemo(device="cpu")
    demo.get_or_load_voxcpm()
    assert demo.asr_model_id == "iic/SenseVoiceSmall"
    assert web_app.voxcpm.VoxCPM.from_pretrained.call_args.args == ("openbmb/VoxCPM2",)
    assert web_app.voxcpm.VoxCPM.from_pretrained.call_args.kwargs["local_files_only"] is False


def test_disabling_compile_keeps_cuda_inference(web_app, local_models):
    demo = web_app.VoxCPMDemo(**local_models, device="cuda", local_files_only=True, optimize=False)
    demo.get_or_load_voxcpm()
    kwargs = web_app.voxcpm.VoxCPM.from_pretrained.call_args.kwargs
    assert kwargs["device"] == "cuda"
    assert kwargs["optimize"] is False


def test_offline_ui_disables_analytics_and_shares_the_gpu_queue(web_app, local_models):
    demo = web_app.VoxCPMDemo(**local_models, device="cuda", local_files_only=True)
    interface = web_app.create_demo_interface(demo)
    try:
        assert interface.analytics_enabled is False
        functions = {event.fn.__name__: event for event in interface.fns.values() if event.fn}
        for name in ("_generate", "_run_asr_if_needed"):
            assert functions[name].concurrency_id == "voxcpm_gpu"
            assert functions[name].concurrency_limit == 1
        theme = web_app.create_app_theme(local_files_only=True)
        assert "fonts.googleapis.com" not in str(theme._get_theme_css())
    finally:
        interface.close()


def test_offline_launch_disables_public_sharing(web_app, local_models, monkeypatch):
    interface = Mock()
    interface.queue.return_value = interface
    monkeypatch.setattr(web_app, "create_demo_interface", lambda demo: interface)
    web_app.run_demo(**local_models, device="cuda", local_files_only=True, server_port=8088)
    assert interface.launch.call_args.kwargs["share"] is False
    assert interface.launch.call_args.kwargs["server_port"] == 8088
