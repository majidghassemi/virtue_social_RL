"""Checkpoints: the full runner state (to resume) and the policy parameters (to evaluate)."""

from __future__ import annotations

import json
from pathlib import Path

from flax import serialization


def save_runner(path: Path, runner) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(serialization.to_bytes(runner))
    tmp.replace(path)                       # atomic, so a job killed mid-write keeps the old file


def load_runner(path: Path, template):
    return serialization.from_bytes(template, path.read_bytes())


def save_params(path: Path, params, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def load_params(path: Path, template):
    """`template`: params of a freshly initialised network with the same architecture."""
    return serialization.from_bytes(template, Path(path).read_bytes())


def read_meta(path: Path) -> dict:
    return json.loads(Path(path).with_suffix(".json").read_text())
