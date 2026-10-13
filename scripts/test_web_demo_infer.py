"""Manual CUDA/Web acceptance checks; requires all three downloaded demo models.

Run with --offline to start a temporary demo with non-loopback networking blocked
inside the test process, or without it to check the running background service.
"""

import argparse
import ipaddress
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def block_external_network(attempts):
    """Keep local HTTP usable while rejecting external DNS and TCP connections."""
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_lookup = socket.gethostbyname

    def allowed(host):
        if host in (None, "", "localhost", "0.0.0.0", "::"):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def reject(host):
        attempts.append(str(host))
        raise OSError(f"Offline acceptance check blocked external networking: {host}")

    def resolve(host, *args, **kwargs):
        if not allowed(host):
            reject(host)
        return original_resolve(host, *args, **kwargs)

    def connect(connection, address):
        if isinstance(address, tuple) and not allowed(address[0]):
            reject(address[0])
        return original_connect(connection, address)

    def connect_ex(connection, address):
        if isinstance(address, tuple) and not allowed(address[0]):
            reject(address[0])
        return original_connect_ex(connection, address)

    def sendto(connection, data, *args):
        address = args[-1]
        if isinstance(address, tuple) and not allowed(address[0]):
            reject(address[0])
        return original_sendto(connection, data, *args)

    def lookup(host):
        if not allowed(host):
            reject(host)
        return original_lookup(host)

    socket.getaddrinfo = resolve
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.gethostbyname = lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    args = parser.parse_args()
    output = ROOT / "output" / "web_demo" / ("offline_validation" if args.offline else "validation")
    output.mkdir(parents=True, exist_ok=True)
    report = {"model": "VoxCPM2", "device": "cuda", "physical_gpu": 0, "offline_guard": args.offline, "cases": []}
    attempts = []
    interface = None
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["GRADIO_TEMP_DIR"] = str(output / "gradio")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        block_external_network(attempts)

    try:
        import numpy as np
        import soundfile as sf
        from gradio_client import Client, handle_file

        if args.offline:
            import app

            models = ROOT / "pretrained_models"
            demo = app.VoxCPMDemo(
                model_id=str(models / "VoxCPM2"),
                asr_model_id=str(models / "SenseVoiceSmall"),
                zipenhancer_model_id=str(models / "ZipEnhancer"),
                device="cuda",
                local_files_only=True,
                optimize=False,
            )
            interface = app.create_demo_interface(demo)
            interface.queue(max_size=10, default_concurrency_limit=1).launch(
                server_name="127.0.0.1",
                server_port=port,
                prevent_thread_lock=True,
                share=False,
                i18n=app.I18N,
                theme=app.create_app_theme(True),
                css=app._CUSTOM_CSS,
            )
            args.url = f"http://127.0.0.1:{port}"

        client = Client(args.url, verbose=False, download_files=str(output))
        report["url"] = args.url

        def generation_args(
            text="你好，这是本地模型生成的测试语音。",
            control="",
            reference=None,
            ultimate=False,
            transcript="",
            normalize=False,
            denoise=False,
            seed=42,
        ):
            return [
                text,
                control,
                handle_file(reference) if reference else None,
                ultimate,
                transcript,
                2.0,
                normalize,
                denoise,
                10,
                seed,
            ]

        def save_result(name, result, elapsed):
            source, seed = result
            wav, sample_rate = sf.read(source)
            assert sample_rate == 48000 and len(wav) > 0 and np.isfinite(wav).all()
            assert np.max(np.abs(wav)) > 0
            destination = output / f"{name}.wav"
            sf.write(destination, wav, sample_rate)
            report["cases"].append(
                {
                    "name": name,
                    "seconds": round(elapsed, 2),
                    "sample_rate": sample_rate,
                    "audio_seconds": round(len(wav) / sample_rate, 2),
                    "seed": seed,
                    "file": str(destination),
                }
            )
            print(f"PASS {name}: {destination}", flush=True)

        reference = str(ROOT / "examples" / "reference_speaker.wav")
        cases = [
            ("chinese", generation_args()),
            ("voice_design", generation_args(control="年轻女性，声音温柔自然")),
            ("reference_clone", generation_args(reference=reference)),
            ("denoised_clone", generation_args(reference=reference, denoise=True)),
            ("normalize_zh", generation_args(text="现在是2026年10月13日，价格是12.5元。", normalize=True)),
            ("normalize_en", generation_args(text="There are 12 items priced at 3.5 dollars each.", normalize=True)),
        ]
        for name, inputs in cases:
            started = time.monotonic()
            save_result(name, client.predict(*inputs, api_name="/generate"), time.monotonic() - started)

        transcription = client.predict(True, handle_file(reference), api_name="/_run_asr_if_needed")
        transcript = transcription["value"] if isinstance(transcription, dict) else transcription
        assert transcript and transcript.strip(), "ASR did not return a transcript"
        report["transcript"] = transcript
        started = time.monotonic()
        save_result(
            "ultimate_clone",
            client.predict(
                *generation_args(reference=reference, ultimate=True, transcript=transcript), api_name="/generate"
            ),
            time.monotonic() - started,
        )

        started = time.monotonic()
        jobs = [
            client.submit(*generation_args(text=f"这是第{number}个排队测试。", seed=42 + number), api_name="/generate")
            for number in (1, 2)
        ]
        states = []
        while not all(job.done() for job in jobs):
            state = [job.status().code.name for job in jobs]
            if not states or states[-1] != state:
                states.append(state)
            time.sleep(0.1)
        for number, job in enumerate(jobs, 1):
            save_result(f"queued_{number}", job.result(), time.monotonic() - started)
        report["queue_states"] = states
        assert any("IN_QUEUE" in state and "PROCESSING" in state for state in states), states
        assert not attempts, f"Unexpected external network attempts: {attempts}"
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = repr(error)
        raise
    finally:
        if interface is not None:
            interface.close()
        report["blocked_network_attempts"] = attempts
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
