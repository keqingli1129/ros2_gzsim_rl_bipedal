#!/usr/bin/env bash
# Sync the RL/training venv and deterministically pin cv2 to the headless
# build. `uv sync` alone installs opencv-python *and* opencv-python-headless
# side by side (stable-baselines3[extra] requires the GUI build; we require
# headless) and whichever finishes writing to site-packages/cv2/ last wins
# the import - not reliably headless. Re-installing opencv-python-headless
# last makes that deterministic instead of a race.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync
uv pip install --reinstall-package opencv-python-headless opencv-python-headless

uv run python -c "
import cv2, numpy as np
# opencv-python-headless still exposes cv2.imshow as a stub; it raises at
# call time since no GUI backend is compiled in. That's the real signal
# headless won the site-packages/cv2/ install race (hasattr() alone can't
# tell the two builds apart).
try:
    cv2.imshow('t', np.zeros((1, 1, 3), dtype='uint8'))
    raise SystemExit('GUI opencv-python won the cv2 install race, expected headless')
except cv2.error:
    print(f'cv2 {cv2.__version__} OK (headless build confirmed)')
"
