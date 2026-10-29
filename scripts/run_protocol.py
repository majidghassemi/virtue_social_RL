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


def jobs_for(entry, condition: str, cfg: dict) -> list[tuple[str, float, float]]:
    jobs = [("mixture", float(w), 0.0) for w in cfg["ladder"]["mixture_w"]]
    jobs += [(d, 0.0, 0.0) for d in cfg["ladder"]["extra"]]
    nz = cfg.get("noise") or {}
    if entry.split in nz.get("splits", []) and condition in nz.get("conditions", []):
        jobs += [("mixture", float(w), float(e))
                 for e in nz["eps"] for w in cfg["ladder"]["mixture_w"]]
    return jobs


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
        for condition in cfg["conditions"]:
            env = suite.env(entry.name, condition == "present")
            policy = ScriptedPolicy(env)
            for disp, w, eps in jobs_for(entry, condition, cfg):
                t0 = time.perf_counter()
                params = make_params(roles, disp, w, eps)
                key = job_key(int(cfg["seed"]), entry.name, condition, disp, w, eps)
                counts = jax.device_get(run_counts(env, policy, key, num_envs, num_steps, params))
                e_idx, a_idx = np.meshgrid(np.arange(num_envs), np.arange(roles.num_agents),
                                           indexing="ij")
                df = pd.DataFrame({k: np.asarray(counts[k]).ravel() for k in COUNT_FIELDS})
                df.insert(0, "agent", a_idx.ravel())
                df.insert(0, "role", role_of[a_idx.ravel()])
                df.insert(0, "unit", e_idx.ravel() // per_unit)
                df.insert(0, "episode", e_idx.ravel())
                for col, val in (("eps", eps), ("w", w), ("disposition", disp),
                                 ("condition", condition), ("band", entry.band),
                                 ("split", entry.split), ("map", entry.name)):
                    df.insert(0, col, val)
                frames.append(df)
                print(f"[{i + 1}/{len(entries)}] {entry.name} {condition:<7} {disp:<12} "
                      f"w={w:.2f} eps={eps:.1f}  {time.perf_counter() - t0:5.1f}s")
        pd.concat(frames).to_csv(path, index=False, compression="gzip")
    print(f"done in {time.perf_counter() - t_start:.0f}s, counts in {out_dir}")


if __name__ == "__main__":
    main()
