"""Tiny inference stand-in for testing the desktop process lifecycle."""

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("action")
parser.add_argument("--output", type=Path)
parser.add_argument("--prompt")
parser.add_argument("--frames", type=int)
parser.add_argument("--duration", type=float)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--seed", type=int)
args, _ = parser.parse_known_args()
print('APNG_EVENT {"stage":"encoding"}', flush=True)
if args.prompt == "WAIT":
    time.sleep(60)
if args.prompt == "FAIL":
    print("Simulated inference error", file=sys.stderr)
    raise SystemExit(1)
print('APNG_EVENT {"stage":"generating","current":1,"total":2}', flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
frames = [Image.new("RGB", (64, 32), color) for color in ("red", "blue")]
frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=100, loop=0)
args.output.with_suffix(".json").write_text(
    json.dumps({key: value for key, value in vars(args).items() if key != "output"}),
    encoding="utf-8",
)
print('APNG_EVENT {"stage":"completed"}', flush=True)
