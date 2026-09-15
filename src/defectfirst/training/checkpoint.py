from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from defectfirst.io import read_json, sha256, write_json


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume RNG CUDA device layout differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
        write_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": sha256(path)})
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, contract: dict) -> dict:
    if sha256(path) != read_json(path.with_suffix(path.suffix + ".sha256.json"))["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    # Full resume contains Python/NumPy RNG states. Only load our trusted local checkpoints.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["contract"] != contract:
        raise ValueError("Refusing resume with changed config/data/schedule/model assets")
    return state
