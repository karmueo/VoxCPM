# Local VoxCPM2 F16 CUDA deployment

This guide uses the `tc-mb/llama.cpp-omni` command-line engine to run VoxCPM2 with the F16 BaseLM and Acoustic GGUF files. It covers a pinned CUDA build, local WAV generation, voice cloning, and an offline acceptance run. The deployment helper is separate from the VoxCPM Python inference environment and does not require PyTorch or a project virtual environment.

Run the examples from the VoxCPM repository root. The helper also locates the repository when called from another working directory.

## Requirements and recorded environment

Use a Linux host with an NVIDIA GPU, its existing compatible driver and CUDA Toolkit, CMake, a C/C++ compiler, Git, Python 3, and `ffmpeg`/`ffprobe`. `ffmpeg` is used to normalize reference audio; `ffprobe` can inspect generated WAVs. Do not change the system driver or CUDA installation for this workflow. The helper explicitly requests a CUDA build for NVIDIA compute capability 12.0 and reports a build failure instead of silently selecting CPU inference.

The machine used to prepare this recipe has two RTX 5090 GPUs (32,607 MiB each), driver 595.84, CUDA 13.2.86, CMake 3.28.3, GCC 13.3, Python 3.12.3, and working `ffmpeg`/`ffprobe`. These are environment details, not a promise of a particular synthesis speed or listening quality.

## Pinned engine and model files

The build helper fetches the engine at commit [`722eb4a6724598a8c83d313cd60e744deabb9c24`](https://github.com/tc-mb/llama.cpp-omni/commit/722eb4a6724598a8c83d313cd60e744deabb9c24). Model files come from the pinned [`DennisHuang648/VoxCPM2-GGUF` revision `169f64d8b98bbaab1761e4ca3a83e6af653456cc`](https://huggingface.co/DennisHuang648/VoxCPM2-GGUF/tree/169f64d8b98bbaab1761e4ca3a83e6af653456cc):

| Artifact | Size in bytes | SHA-256 |
| --- | ---: | --- |
| BaseLM F16 | 3,247,980,544 | `8be62e899f8f32b3c2109c99950b46671135a8f40b27b1df720f6823d36cc59e` |
| Acoustic F16 | 1,825,096,352 | `5bde898488ad635ff55d24da53543768fa33d5e5cdc538ce190e5ef831038e85` |

The helper stores source and build files under `output/llama_cpp_omni/src` and `output/llama_cpp_omni/build`, and the two models under `pretrained_models/VoxCPM2-GGUF/`. These locations are ignored by Git. The build records the engine revision, model hashes, and build details alongside its log in the ignored output directory.

## Download and build

The preparation step needs network access to fetch the pinned source and model files. It checks the model hashes before building. Use the existing CUDA, CMake, and compiler installation:

```bash
python3 scripts/voxcpm2_cuda.py build --jobs 16
```

`--jobs` is optional; set it to a value appropriate for the host. If Hugging Face downloads are unreliable, add `--model-source modelscope` to use the repository-recommended [pinned ModelScope revision](https://modelscope.cn/models/DennisHuang/VoxCPM2-GGUF/files?revision=bb319272866df0419de6a2357c9e675421b2688e). Both sources provide identical file sizes and SHA-256 values. The helper resumes interrupted transfers and verifies the complete file. The command builds the `voxcpm2-cli` executable with CUDA enabled for compute capability 12.0. Keep the build log if compilation fails so the CUDA detection or compiler error is available for diagnosis. The helper does not install or upgrade system packages, drivers, or CUDA.

## Offline synthesis

Once the build and downloads finish, inference uses the local executable and model files and does not download anything. By default it uses physical GPU 1 and the pinned F16 model pair. The default sampling settings are seed 42, 10 timesteps, CFG 2.0, and 200 steps.

Ordinary English synthesis:

```bash
python3 scripts/voxcpm2_cuda.py infer \
  --text "Hello. This is a local VoxCPM2 CUDA synthesis." \
  --output output/llama_cpp_omni/hello.wav
```

Ordinary Chinese synthesis:

```bash
python3 scripts/voxcpm2_cuda.py infer \
  --text "你好，这是一次本机 VoxCPM2 CUDA 语音合成。" \
  --output output/llama_cpp_omni/hello_zh.wav
```

To select physical GPU 0, add `--gpu 0`; `--gpu 1` selects physical GPU 1. The helper selects the requested physical GPU by UUID and exposes it to the engine as its CUDA device 0. An existing `CUDA_VISIBLE_DEVICES` setting does not reinterpret the `--gpu` physical index. The F16 models are loaded on the GPU with all model layers requested (`--n-gpu-layers 999`). If CUDA or the requested GPU is unavailable, the command fails instead of falling back to CPU.

You may override the default local model paths when needed:

```bash
python3 scripts/voxcpm2_cuda.py infer \
  --text "A short test." \
  --output output/llama_cpp_omni/custom.wav \
  --base-model pretrained_models/VoxCPM2-GGUF/BASE_FILE.gguf \
  --acoustic-model pretrained_models/VoxCPM2-GGUF/ACOUSTIC_FILE.gguf
```

Replace the two example filenames with the actual local filenames. `infer` accepts `--seed`, `--timesteps`, `--cfg`, and `--steps` for explicit reproducible settings.

## Voice cloning

Reference-audio-only cloning uses the reference voice without supplying a transcript:

```bash
python3 scripts/voxcpm2_cuda.py infer \
  --text "The words to synthesize in the reference voice." \
  --reference-audio examples/reference_speaker.wav \
  --output output/llama_cpp_omni/reference_only.wav
```

For transcript-guided cloning, pass both a reference file and its exact spoken words. Do not guess or paraphrase the transcript. The text must match the audio:

```bash
python3 scripts/voxcpm2_cuda.py infer \
  --text "The new words to synthesize." \
  --reference-audio /path/to/your_reference.wav \
  --reference-text "Replace this with the exact words spoken in your reference recording." \
  --output output/llama_cpp_omni/reference_text.wav
```

The repository's `examples/reference_speaker.wav` is mono PCM16 at 48 kHz, about 12.8 seconds long. Its exact transcript is not supplied here, so use it for reference-audio-only mode unless you independently know the accurate words. The helper automatically normalizes reference audio to mono, 16 kHz PCM16 WAV with the installed `ffmpeg`. You can also convert it first:

```bash
ffmpeg -i /path/to/input_audio -ac 1 -ar 16000 -c:a pcm_s16le output/llama_cpp_omni/reference.wav
```

Then pass the converted WAV with `--reference-audio`.

## Acceptance run and offline check

The acceptance command generates the fixed English and Chinese ordinary-synthesis and reference-audio-only cases three times each by default, checks the WAV outputs for 48 kHz audio with finite, non-silent samples, and saves WAVs and measurements in an ignored directory. It records first and later run times, audio duration, real-time factor, and observed GPU memory; listen to the saved clips to assess voice quality. Each run launches a separate CLI process. Wall time includes startup and model loading; engine time and RTF use the CLI generation-stage timer. GPU memory observations cover the selected physical GPU and can include other processes.

```bash
python3 scripts/voxcpm2_cuda.py test \
  --gpu 1 --runs 3 \
  --output-dir output/llama_cpp_omni/acceptance
```

Repeat with the Linux x86_64 seccomp network filter to verify the prepared deployment runs offline. The filter blocks network socket creation for each engine process and preserves the local sockets required by CUDA:

```bash
python3 scripts/voxcpm2_cuda.py test \
  --gpu 1 --runs 3 \
  --output-dir output/llama_cpp_omni/acceptance-offline \
  --offline
```

The Markdown and JSON summaries are saved as `report.md` and `report.json` in the selected acceptance directory. The generated WAV files are also retained there for listening. Review the report and audio from this machine; no speed or subjective-quality result is assumed by this guide.

## Troubleshooting

- **Build reports that CUDA is unavailable or compilation fails:** check `nvidia-smi`, `nvcc --version`, CMake, and the C/C++ compiler versions. Read the preserved build log. This flow does not change the installed driver or CUDA Toolkit.
- **The latest upstream HEAD fails its first Linux build on `F_NOCACHE`:** its MoE code unconditionally uses a macOS-only flag. Use the compatible commit pinned in this guide; there is no need to change system CUDA.
- **The selected GPU is unavailable or the wrong device is busy:** pass `--gpu 0` or `--gpu 1` using the physical GPU number. Check device visibility and the helper's recorded GPU UUID. CUDA errors stop inference rather than causing CPU fallback.
- **A model is missing or its checksum does not match:** rerun `build` while online and let it verify the pinned files. `infer` and `test` do not fetch replacements.
- **Reference audio cannot be read:** confirm `ffmpeg` is installed so the helper can normalize the input to mono, 16 kHz PCM16 WAV. Check that the file is non-empty and decodable.
- **Transcript-guided cloning sounds misaligned:** ensure `--reference-text` is the exact transcript of `--reference-audio`. Leave out `--reference-text` for reference-audio-only cloning.
- **Offline acceptance cannot start:** complete `build` while online first, then check that both local F16 files and the compiled executable exist. The `--offline` option is for verifying inference after preparation; it does not download missing assets.

The helper and guides live in the repository source directories; models, build products, logs, temporary audio, and default generated output remain in ignored output/model directories. This deployment does not alter the existing Python inference setup.
