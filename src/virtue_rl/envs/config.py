"""Build the ethics environment from a YAML config."""

from __future__ import annotations

from pathlib import Path

import yaml

from virtue_rl.envs.roles import RoleConfig
from virtue_rl.envs.wrapper import EthicsHarvest


def make_from_config(cfg: dict | str | Path) -> EthicsHarvest:
    if not isinstance(cfg, dict):
        cfg = yaml.safe_load(Path(cfg).read_text())
    roles = RoleConfig.from_dict(cfg["roles"])
    env_cfg = dict(cfg.get("env", {}))
    add_teacher_channel = bool(env_cfg.pop("add_teacher_channel", True))
    if "grid_size" in env_cfg:
        env_cfg["grid_size"] = tuple(env_cfg["grid_size"])
    return EthicsHarvest(roles, add_teacher_channel=add_teacher_channel, **env_cfg)
