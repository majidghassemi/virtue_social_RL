"""Run the Phase 3 protocol and log raw episode counts.

Usage:
    python scripts/run_protocol.py                                    # full config
    python scripts/run_protocol.py --units 4 --envs-per-unit 8 \\
        --out-dir runs/protocol_v1_pilot                              # quick pilot

For every (map, teacher condition, disposition, w, eps) job it rolls out
units x envs_per_unit episodes and writes one row per (episode, agent) with
the raw counts from `virtue_rl.metrics.behaviour.COUNT_FIELDS`. One compressed
CSV per map, so an interrupted run resumes by skipping finished maps.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import yaml

from virtue_rl.agents import ScriptedPolicy, make_params
from virtue_rl.envs import RoleConfig
from virtue_rl.metrics.behaviour import COUNT_FIELDS, episode_counts
from virtue_rl.runner import MapSuite, rollout


@partial(jax.jit, static_argnames=("env", "policy", "num_envs", "num_steps"))
def run_counts(env, policy, key, num_envs, num_steps, params):
    traj = rollout(env, policy, key, num_envs=num_envs, num_steps=num_steps,
                   policy_params=params)
    return episode_counts(traj["ethics"])


def jobs_for(entry, condition: str, cfg: dict) -> list[dict]:
    """Jobs for one (map, condition): a list of learner-group descriptions.

    Two config styles: `groups` (protocol v2, named groups with rule, cue and
    view_only) or `ladder` plus `noise` (protocol v1 and the crossed ladder).
    """
    if "groups" in cfg:
        return [{"group": g["name"], "disposition": g["disposition"],
                 "w": float(g.get("w", 0.0)), "eps": float(g.get("eps", 0.0)),
                 "rule": g.get("rule", "minimal"), "cue": g.get("cue", "in_view"),
                 "view_only": bool(g.get("view_only", False))}
                for g in cfg["groups"] if condition in g["conditions"]]
    tuples = []
    if cfg["ladder"].get("include_base", True):   # eps = 0 rungs
        tuples += [("mixture", float(w), 0.0) for w in cfg["ladder"]["mixture_w"]]
    tuples += [(d, 0.0, 0.0) for d in cfg["ladder"].get("extra", [])]
    nz = cfg.get("noise") or {}
    if entry.split in nz.get("splits", []) and condition in nz.get("conditions", []):
        tuples += [("mixture", float(w), float(e))
                   for e in nz["eps"] for w in cfg["ladder"]["mixture_w"]]
    return [{"group": None, "disposition": d, "w": w, "eps": e, "rule": "minimal",
             "cue": "in_view", "view_only": False} for d, w, e in tuples]


def conditions_of(cfg: dict) -> list[str]:
    if "groups" in cfg:
        return [c for c in ("absent", "present") if any(c in g["conditions"] for g in cfg["groups"])]
    return cfg["conditions"]


def job_key(seed: int, *parts) -> jax.Array:
    h = zlib.crc32("|".join(map(str, parts)).encode())
    return jax.random.fold_in(jax.random.PRNGKey(seed), h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/protocol_v1.yaml"))
    p.add_argument("--units", type=int, default=None)
    p.add_argument("--envs-per-unit", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--maps", nargs="*", default=None, help="restrict to these map names")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    units = args.units or int(cfg["units"])
    per_unit = args.envs_per_unit or int(cfg["envs_per_unit"])
    out_dir = args.out_dir or Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    num_envs, num_steps = units * per_unit, int(cfg["num_steps"])

    roles = RoleConfig.from_dict(cfg["roles"])
    role_of = np.array(["teacher"] + ["learner"] * roles.num_learners
                       + ["bystander"] * roles.num_bystanders)
    suite = MapSuite(cfg["suite"], roles, num_inner_steps=num_steps)
    entries = [e for e in suite.entries if e.split in cfg["splits"]
               and (args.maps is None or e.name in args.maps)]

    (out_dir / "run_config.json").write_text(json.dumps(
        cfg | {"units": units, "envs_per_unit": per_unit}, indent=1))
    t_start = time.perf_counter()
    for i, entry in enumerate(entries):
        path = out_dir / f"counts_{entry.name}.csv.gz"
        if path.exists():
            print(f"[{i + 1}/{len(entries)}] {entry.name}: exists, skipping")
            continue
        frames = []
        for condition in conditions_of(cfg):
            env = suite.env(entry.name, condition == "present")
            policy = ScriptedPolicy(env)
            for job in jobs_for(entry, condition, cfg):
                disp, w, eps = job["disposition"], job["w"], job["eps"]
                t0 = time.perf_counter()
                params = make_params(roles, disp, w, eps, rule=job["rule"], cue=job["cue"],
                                     view_only=job["view_only"])
                if job["group"] is None:   # v1 keys, unchanged so old runs reproduce
                    key = job_key(int(cfg["seed"]), entry.name, condition, disp, w, eps)
                else:
                    key = job_key(int(cfg["seed"]), entry.name, condition, job["group"])
                counts = jax.device_get(run_counts(env, policy, key, num_envs, num_steps, params))
                e_idx, a_idx = np.meshgrid(np.arange(num_envs), np.arange(roles.num_agents),
                                           indexing="ij")
                df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in counts.items()})
                df.insert(0, "agent", a_idx.ravel())
                df.insert(0, "role", role_of[a_idx.ravel()])
                df.insert(0, "unit", e_idx.ravel() // per_unit)
                df.insert(0, "episode", e_idx.ravel())
                if job["group"] is not None:
                    for col in ("view_only", "cue", "rule", "group"):
                        df.insert(0, col, job[col])
                for col, val in (("eps", eps), ("w", w), ("disposition", disp),
                                 ("condition", condition), ("band", entry.band),
                                 ("split", entry.split), ("map", entry.name)):
                    df.insert(0, col, val)
                frames.append(df)
                label = job["group"] or disp
                print(f"[{i + 1}/{len(entries)}] {entry.name} {condition:<7} {label:<16} "
                      f"w={w:.2f} eps={eps:.2f}  {time.perf_counter() - t0:5.1f}s")
        pd.concat(frames).to_csv(path, index=False, compression="gzip")
    print(f"done in {time.perf_counter() - t_start:.0f}s, counts in {out_dir}")


if __name__ == "__main__":
    main()
