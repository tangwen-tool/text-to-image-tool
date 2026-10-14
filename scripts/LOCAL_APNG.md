# Local text-to-APNG verification

This standalone experiment does not use the desktop application's cloud providers.
Use Python 3.12 and an NVIDIA CUDA GPU. Model weights, the text encoder, and the
runtime require many GB of disk space; generation speed and memory must be measured.

## Setup (PowerShell)

```powershell
py -3.12 -m venv .venv-wan
.\.venv-wan\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-wan\Scripts\python.exe -m pip install -r scripts/wan_requirements.txt
.\.venv-wan\Scripts\python.exe scripts/local_apng.py check
.\.venv-wan\Scripts\python.exe scripts/local_apng.py download --source modelscope
```

The default model directory is `models/wan2.1-t2v-1.3b`. Copy the complete directory
to another computer for offline use. Both machines need a compatible runtime.
`--model-dir D:\Models\wan2.1-t2v-1.3b` selects a different location.
Use `download --source modelscope` to download the official model from ModelScope
instead of Hugging Face. Generation remains offline for either download source.

## Generate offline

```powershell
.\.venv-wan\Scripts\python.exe scripts/local_apng.py generate --prompt "Photorealistic ocean waves rolling onto a beach, fixed camera, natural daylight."
```

Defaults: 832x480, 33 frames, 30 inference steps, 10 FPS. This is a verification
configuration, not a guaranteed quality or performance target. Try English prompts
first to isolate inference problems. The PNG is an animated PNG (APNG), not GIF.
Generation keeps the video model and VAE on CUDA and uses VAE tiling. No online
prompt enhancement is used. CPU model offload is disabled to avoid exhausting
system memory during repeated GPU-to-CPU transfers on the verified 16 GB RAM PC.
The text encoder is loaded onto the GPU first and released before the video model
is loaded. This reduces simultaneous RAM use, but text encoding itself needs enough
free VRAM. The current experiment targets the verified 16 GB NVIDIA GPU, not all GPUs.
On Windows, checkpoint reads use the safetensors `pread` backend to avoid native
access violations in mmap-backed torch storage. This requires safetensors 0.8+ and
can increase loading time on machines with little available RAM. Video checkpoints
are staged on CUDA with owned tensor storage before dtype conversion. CPU tensor
operations use one thread. Failed shots record exit codes and per-shot `.log` files;
select a failed shot to view the error.

Outputs:

- `outputs/sample.png`: looping APNG (looping does not guarantee a seamless transition).
- `outputs/sample_frames/`: original PNG frames for later video encoding.
- `outputs/sample.json`: prompt, seed, hardware and generation measurements.

Use a new `--output` name for each run; the frame directory must not already exist.
An APNG-aware viewer is required to see animation. Model files and outputs are ignored
by Git. Installer packaging for the local model runtime has not yet been implemented.

## Desktop verification

Install the desktop dependencies into the same runtime with
`.\.venv-wan\Scripts\python.exe -m pip install -e .`, then double-click
`run_local_apng.bat` or run `.\.venv-wan\Scripts\python.exe main.py`.
The Local APNG window runs inference in a separate process. It supports prompt
input, two resolutions, a 0.5–5 second playback duration input, progress, cancellation,
animated playback, save-as,
and custom model/runtime paths. The seed is fixed internally at 42. Frame counts
are rounded up to 4n+1; APNG frame delays sum exactly to the requested playback
duration. The desktop splits scripts into paragraphs, lines, or sentences and starts
serial generation immediately. Each batch is saved under `outputs/desktop/` in a
unique directory, with `script.txt`, `batch.json`, numbered APNGs and original frames.
Failures do not stop later shots; cancellation preserves completed shots. No project,
storyboard confirmation, or API key is required.

Automated UI tests cover Unicode prompt transport, playback, cancellation and retry.
`scripts/verify_desktop_apng.py` also runs the real model through the desktop page
and saves UI screenshots under `outputs/desktop-qa/`. Add `--render-only` to inspect
the layout using the existing beach sample without running model inference again.

Official model: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers

## Verified on 2026-10-13

Windows, RTX 3080 Ti Laptop GPU (16 GB VRAM), 15.78 GiB RAM, Python 3.12.14,
PyTorch 2.8.0+cu128. ModelScope downloaded 20 model files totaling 26.94 GiB.
Generation ran with offline flags and local-files-only loading.

| Output | Steps | Setup including text encoding | Generation including frame decoding |
| --- | --- | --- | --- |
| 512x288, 17 frames | 20 | 23.2 s | 26.9 s |
| 832x480, 33 frames | 30 | 32.2 s | 224.7 s |

Both runs recorded a peak PyTorch CUDA allocation of 10.80 GiB (not total system
VRAM consumption). The 480P APNG is 22,100,813 bytes, plays for 3.3 seconds at 10 FPS,
and has 33 matching original PNG frames. Frame decoding and pixel comparisons
confirmed that the image is animated. Other devices have not been validated.
