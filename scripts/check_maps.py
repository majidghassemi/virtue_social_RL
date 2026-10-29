"""Run a random policy on every map in both teacher conditions and check invariants.

Usage:
    python scripts/check_maps.py                                  # 16 envs x 300 steps
    python scripts/check_maps.py --num-envs 64 --num-steps 1000 --out results/map_check.json

Checks, per (map, condition):
  - absent: the teacher never leaves the bay and is never in anyone's view
  - present: the teacher is seen at least once
  - apples are collected, apples x num_agents equals reward on valid steps
Also records compile time, so the cost of one-environment-per-map is visible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from virtue_rl.envs import TEACHER, RoleConfig
from virtue_rl.envs import maps as M
from virtue_rl.runner import MapSuite, random_policy, rollout


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="maps/suite_v1.json")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=300)
    p.add_argument("--out", type=Path, default=Path("results/map_check_v1.json"))
    args = p.parse_args()

    suite = MapSuite(args.suite, RoleConfig(3, 3))
    rows, failures = [], []
    for e in suite.entries:
        for present in (True, False):
            env = suite.env(e.name, present)
            t0 = time.perf_counter()
            traj = jax.block_until_ready(rollout(env, random_policy, jax.random.PRNGKey(0),
                                                 num_envs=args.num_envs,
                                                 num_steps=args.num_steps, keep_states=True))
            compile_s = time.perf_counter() - t0
            eth = traj["ethics"]
            valid = eth.valid
            teacher = traj["states"].agent_locs[:, :, TEACHER, :2]
            in_bay = jnp.all(teacher == jnp.array([M.BAY_ROW, M.BAY_COL]))
            seen = bool(jnp.any(eth.teacher_in_view))
            rewards_ok = bool(jnp.array_equal(eth.apples * env.num_agents,
                                              jnp.where(valid, traj["rewards"], 0).astype(jnp.int32)))
            row = {
                "map": e.name, "split": e.split, "band": e.band, "teacher_present": present,
                "compile_plus_run_s": round(compile_s, 2),
                "apples_per_env": round(float(jnp.sum(eth.apples)) / args.num_envs, 1),
                "harm_zap_per_env": round(float(jnp.sum(eth.harm_zap)) / args.num_envs, 2),
                "harm_commons_per_env": round(float(jnp.sum(eth.harm_commons)) / args.num_envs, 2),
                "teacher_seen": seen, "teacher_in_bay_all_steps": bool(in_bay),
                "rewards_match_apples": rewards_ok,
            }
            ok = rewards_ok and row["apples_per_env"] > 0
            ok &= (bool(in_bay) and not seen) if not present else seen
            row["ok"] = bool(ok)
            rows.append(row)
            if not ok:
                failures.append(row)
            print(f"{e.name:<32} {'present' if present else 'absent ':<8} ok={ok} "
                  f"compile+run {compile_s:5.1f}s apples/env {row['apples_per_env']:6.1f} "
                  f"zap/env {row['harm_zap_per_env']:5.2f} commons/env {row['harm_commons_per_env']:5.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"num_envs": args.num_envs, "num_steps": args.num_steps,
                                    "rows": rows}, indent=1) + "\n")
    total = sum(r["compile_plus_run_s"] for r in rows)
    print(f"\n{len(rows) - len(failures)}/{len(rows)} passed, total {total:.0f}s, saved {args.out}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
