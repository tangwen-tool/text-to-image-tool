"""Download Wan weights separately, then generate APNG with networking disabled."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def emit_event(stage, **details):
    print("APNG_EVENT " + json.dumps({"stage": stage, **details}), flush=True)


def frame_durations(frame_count, fps, duration=None):
    if duration is None:
        return [round(1000 / fps)] * frame_count
    total_ms = round(duration * 1000)
    return [
        round((index + 1) * total_ms / frame_count) - round(index * total_ms / frame_count)
        for index in range(frame_count)
    ]


def configure_weight_reader():
    if sys.platform != "win32":
        return
    import safetensors
    import safetensors.torch

    original_open = safetensors.safe_open

    # Windows mmap-backed torch storage can crash when reading large checkpoints.
    def open_without_mmap(*args, **kwargs):
        kwargs["backend"] = "pread"
        return original_open(*args, **kwargs)

    def load_owned_tensors(filename, device="cpu", *, backend="pread"):
        import torch

        target = "cuda:0" if str(device) == "cpu" and torch.cuda.is_available() else device
        with original_open(filename, framework="pt", device=target, backend="pread") as reader:
            # Keep the reader alive until each returned tensor owns its storage.
            names = reader.keys()
            return {key: reader.get_tensor(key).clone() for key in names}

    safetensors.safe_open = open_without_mmap
    safetensors.torch.safe_open = open_without_mmap
    safetensors.torch.load_file = load_owned_tensors


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=["download", "check", "generate"])
    result.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    result.add_argument("--model-dir", type=Path, default=ROOT / "models" / "wan2.1-t2v-1.3b")
    result.add_argument(
        "--prompt",
        default="Photorealistic ocean waves rolling onto a beach, fixed camera, natural daylight.",
    )
    result.add_argument("--output", type=Path, default=ROOT / "outputs" / "sample.png")
    result.add_argument("--width", type=int, default=832)
    result.add_argument("--height", type=int, default=480)
    result.add_argument("--frames", type=int, default=33)
    result.add_argument("--steps", type=int, default=30)
    result.add_argument("--fps", type=int, default=10)
    result.add_argument("--duration", type=float, help="Target APNG playback duration in seconds")
    result.add_argument("--seed", type=int, default=42)
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.action == "download":
        patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]
        if args.source == "modelscope":
            from modelscope import snapshot_download

            snapshot_download(MODEL_ID, local_dir=str(args.model_dir), allow_file_pattern=patterns)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                MODEL_ID, local_dir=args.model_dir, max_workers=2, allow_patterns=patterns
            )
        print(f"Model downloaded: {args.model_dir}")
        return

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import psutil
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    hardware = {
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "available_ram_gb": round(psutil.virtual_memory().available / 2**30, 2),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        hardware.update(
            gpu=torch.cuda.get_device_name(),
            vram_gb=round(total / 2**30, 2),
            free_vram_gb=round(free / 2**30, 2),
        )
    print(json.dumps(hardware, indent=2), flush=True)
    if args.action == "check":
        return
    if not torch.cuda.is_available():
        cli.error("This verification requires a CUDA-compatible NVIDIA GPU.")
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        cli.error("Width and height must be positive multiples of 16.")
    if args.frames < 5 or (args.frames - 1) % 4:
        cli.error("Frame count must be 4n+1 and at least 5.")
    if args.steps < 1 or not 1 <= args.fps <= 100 or not args.prompt.strip():
        cli.error("Provide a prompt, positive steps, and FPS between 1 and 100.")
    if args.duration is not None and not 0.5 <= args.duration <= 5:
        cli.error("Duration must be between 0.5 and 5 seconds.")
    if args.output.suffix.lower() not in (".png", ".apng"):
        cli.error("Output must end with .png or .apng.")
    if not (args.model_dir / "model_index.json").is_file():
        cli.error("Model not found. Run the download action or import a complete model directory.")
    frame_dir = args.output.parent / (args.output.stem + "_frames")
    if args.output.exists() or frame_dir.exists():
        cli.error("Output already exists. Choose a new --output name.")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    configure_weight_reader()
    from diffusers import (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
    )
    from PIL import Image
    from transformers import AutoTokenizer, UMT5EncoderModel

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print("Encoding text on GPU...", flush=True)
    emit_event("encoding")
    encoder = UMT5EncoderModel.from_pretrained(
        args.model_dir / "text_encoder",
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer", local_files_only=True)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.model_dir, subfolder="scheduler", local_files_only=True
    )
    # Encode with the official pipeline, then release the large text encoder.
    text_pipe = WanPipeline(
        tokenizer=tokenizer, text_encoder=encoder, vae=None, scheduler=scheduler, transformer=None
    )
    with torch.inference_mode():
        prompt_embeds, negative_embeds = text_pipe.encode_prompt(
            prompt=args.prompt,
            negative_prompt="blurry, static, cartoon, illustration, subtitles, watermark",
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
            max_sequence_length=512,
        )
    del text_pipe, encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    encoded = time.perf_counter()
    print("Loading video generator...", flush=True)
    emit_event("loading")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    pipe = WanPipeline.from_pretrained(
        args.model_dir,
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    loaded = time.perf_counter()
    emit_event("generating", current=0, total=args.steps)

    def on_step(pipeline, step, timestep, callback_kwargs):
        emit_event("generating", current=step + 1, total=args.steps)
        if step + 1 == args.steps:
            emit_event("decoding")
        return callback_kwargs

    with torch.inference_mode():
        frames = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=6.0,
            generator=torch.Generator("cpu").manual_seed(args.seed),
            output_type="pil",
            callback_on_step_end=on_step,
        ).frames[0]
    generated = time.perf_counter()
    emit_event("exporting")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(exist_ok=False)
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"{index:04d}.png")
    durations = frame_durations(len(frames), args.fps, args.duration)
    frames[0].save(
        args.output,
        format="PNG",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=0,
        blend=0,
    )
    with Image.open(args.output) as image:
        if not image.is_animated or image.n_frames != len(frames):
            raise RuntimeError("APNG verification failed")
    report = {
        **hardware,
        "model": MODEL_ID,
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": len(frames),
        "steps": args.steps,
        "fps": args.fps,
        "playback_seconds": sum(durations) / 1000,
        "effective_fps": len(frames) * 1000 / sum(durations),
        "load_seconds": loaded - started,
        "text_encoding_seconds": encoded - started,
        "generation_seconds": generated - loaded,
        "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    emit_event("completed", output=str(args.output.resolve()))


if __name__ == "__main__":
    main()
