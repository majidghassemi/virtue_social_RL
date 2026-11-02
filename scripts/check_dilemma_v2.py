"""Social dilemma check v2: milder defection and discounted value.

Usage:
    python scripts/check_dilemma_v2.py               # configs/dilemma_check_v2.yaml

Scripted populations only, no ethics metric. See the config for the criteria.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from check_dilemma import paired_ci, population, unit_means  # noqa: E402  (same folder)
from virtue_rl.agents import ScriptedPolicy
from virtue_rl.envs import RoleConfig
from virtue_rl.envs.maps import load_suite
from virtue_rl.envs.wrapper import EthicsHarvest
from virtue_rl.runner import rollout

SWITCHER = 1


@partial(jax.jit, static_argnames=("env", "policy", "num_envs", "num_steps"))
def run(env, policy, key, num_envs, num_steps, params, gamma):
    e = rollout(env, policy, key, num_envs=num_envs, num_steps=num_steps,
                policy_params=params)["ethics"]
    apples = e.apples.astype(jnp.float32)                         # (T, E, N), 0 on invalid steps
    disc = gamma ** jnp.arange(num_steps, dtype=jnp.float32)
    return {"undiscounted": apples.sum(0), "discounted": jnp.einsum("t,ten->en", disc, apples)}


def populations(defector: str) -> dict[str, tuple[str, str | None]]:
    return {"all_sustainable": ("sustainable", None),
            "all_d": (defector, None),
            "d_among_sustainable": ("sustainable", defector),
            "sustainable_among_d": (defector, "sustainable")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/dilemma_check_v2.yaml"))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    units, per_unit, steps = int(cfg["units"]), int(cfg["envs_per_unit"]), int(cfg["num_steps"])
    num_envs, gamma = units * per_unit, float(cfg["gamma"])
    rng = np.random.default_rng(0)
    pops = {}                          # unique populations across defectors
    for d in cfg["defectors"]:
        pops.update({(base, sw): None for base, sw in populations(d).values()})
    results = {}
    t_start = time.perf_counter()

    for set_name, ms in cfg["map_sets"].items():
        maps = ([("upstream_default", None)] if ms.get("default_map") else
                [(e.name, e) for e in load_suite(ms["suite"]) if e.split == ms["split"]])
        for n in cfg["num_agents"]:
            roles = RoleConfig(num_learners=1, num_bystanders=n - 2)
            vals = {pk: {v: {"total": [], "switcher": []} for v in ("undiscounted", "discounted")}
                    for pk in pops}
            for mi, (name, entry) in enumerate(maps):
                t0 = time.perf_counter()
                kw = {} if entry is None else {"map_ASCII": entry.ascii(ms["teacher_present_version"]),
                                               "grid_size": entry.grid_size}
                env = EthicsHarvest(roles, num_inner_steps=steps, **kw)
                policy = ScriptedPolicy(env)
                key = jax.random.fold_in(jax.random.PRNGKey(int(cfg["seed"])), mi * 100 + n)
                for (base, sw) in pops:
                    r = jax.device_get(run(env, policy, key, num_envs, steps, population(n, base, sw), gamma))
                    for v in ("undiscounted", "discounted"):
                        a = np.asarray(r[v], float)
                        vals[(base, sw)][v]["total"].append(unit_means(a.sum(1), units))
                        vals[(base, sw)][v]["switcher"].append(unit_means(a[:, SWITCHER], units))
                print(f"{set_name:<9} n={n} {name:<18} {time.perf_counter() - t0:5.1f}s", flush=True)
            for d in cfg["defectors"]:
                P = populations(d)
                for v in ("undiscounted", "discounted"):
                    T = {k: {m: np.stack(x) for m, x in vals[P[k]][v].items()} for k in P}
                    c1 = paired_ci(T["all_sustainable"]["total"], T["all_d"]["total"], 2000, rng)
                    c2 = paired_ci(T["d_among_sustainable"]["switcher"], T["all_sustainable"]["switcher"], 2000, rng)
                    c3 = paired_ci(T["all_d"]["switcher"], T["sustainable_among_d"]["switcher"], 2000, rng)
                    holds = {"C1": c1[1] > 0, "C2": c2[1] > 0, "C3": c3[1] > 0}
                    results[f"{set_name}|{n}|{d}|{v}"] = {
                        "map_set": set_name, "num_agents": n, "defector": d, "value": v,
                        "C1": c1, "C2": c2, "C3": c3, "holds": {k: bool(x) for k, x in holds.items()},
                        "dilemma": bool(all(holds.values())),
                        "means": {k: {m: float(x.mean()) for m, x in T[k].items()} for k in P}}

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")

    def cell(r, k):
        e, lo, hi = r[k]
        return f"{e:.2f} [{lo:.2f}, {hi:.2f}] {'holds' if r['holds'][k] else 'no'}"

    lines = ["# Social dilemma check v2", "",
             f"{units} units x {per_unit} episodes per cell, {steps} steps, gamma {gamma}. "
             f"Criteria in `{args.config}`.", "",
             "| Map set | Agents | Defector | Value | C1 collective benefit | C2 temptation | C3 sucker's cost | Dilemma |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results.values():
        lines.append(f"| {r['map_set']} | {r['num_agents']} | {r['defector']} | {r['value']} | "
                     f"{cell(r, 'C1')} | {cell(r, 'C2')} | {cell(r, 'C3')} | {'yes' if r['dilemma'] else 'no'} |")
    lines += ["", "C1 in total value of all agents, C2 and C3 in value of the switching agent (apples, "
                  "discounted or not)."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
