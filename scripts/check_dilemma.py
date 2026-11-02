"""Check whether Commons Harvest is a social dilemma at 3, 5 and 7 agents.

Usage:
    python scripts/check_dilemma.py                  # configs/dilemma_check.yaml

Scripted populations only, no ethics metric. See the config for the criteria.
Writes a JSON file and a Markdown report next to it.
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

from virtue_rl.agents import ScriptedPolicy
from virtue_rl.agents.scripted import RULES, ScriptedParams
from virtue_rl.envs import RoleConfig
from virtue_rl.envs.maps import load_suite
from virtue_rl.envs.wrapper import EthicsHarvest
from virtue_rl.metrics.behaviour import episode_counts
from virtue_rl.runner import rollout

SWITCHER = 1


def population(n: int, base: str, switcher: str | None = None) -> ScriptedParams:
    """All agents follow `base` ("greedy", "minimal" or "sustainable"), agent 1 optionally `switcher`."""
    kinds = [base] * n
    if switcher is not None:
        kinds[SWITCHER] = switcher
    w = np.array([0.0 if k == "greedy" else 1.0 for k in kinds], np.float32)
    rule = np.array([RULES["sustainable"] if k == "sustainable" else RULES["minimal"] for k in kinds], np.int32)
    z = jnp.zeros(n, bool)
    return ScriptedParams(w=jnp.asarray(w), performative=z, aggressive=z,
                          eps=jnp.zeros(n, jnp.float32), rule=jnp.asarray(rule),
                          cue=jnp.zeros(n, jnp.int32), view_only=z)


POPULATIONS = {
    "all_greedy": ("greedy", None),
    "all_minimal": ("minimal", None),
    "all_sustainable": ("sustainable", None),
    "greedy_among_sustainable": ("sustainable", "greedy"),
    "sustainable_among_greedy": ("greedy", "sustainable"),
}


@partial(jax.jit, static_argnames=("env", "policy", "num_envs", "num_steps"))
def run(env, policy, key, num_envs, num_steps, params):
    c = episode_counts(rollout(env, policy, key, num_envs=num_envs, num_steps=num_steps,
                               policy_params=params)["ethics"])
    return {"apples": c["apples"], "stock": c["stock"][:, 0], "regrown": c["regrown"][:, 0],
            "steps": c["steps"][:, 0]}


def unit_means(x: np.ndarray, units: int) -> np.ndarray:
    """Per-episode values (E,) -> per-unit means (units,). Episodes are grouped in order."""
    return x.reshape(units, -1).mean(1)


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple[float, float, float]:
    """Mean over maps of per-map mean(a - b) with a unit-paired bootstrap CI.

    a, b: (maps, units) per-unit means, paired by unit (same random keys).
    """
    diff = a - b
    est = float(diff.mean())
    m, u = diff.shape
    idx = rng.integers(0, u, (n_boot, m, u))
    boots = np.take_along_axis(np.broadcast_to(diff, (n_boot, m, u)), idx, axis=2).mean(axis=(1, 2))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/dilemma_check.yaml"))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    units, per_unit, steps = int(cfg["units"]), int(cfg["envs_per_unit"]), int(cfg["num_steps"])
    num_envs = units * per_unit
    rng = np.random.default_rng(0)
    results = {}
    t_start = time.perf_counter()

    for set_name, ms in cfg["map_sets"].items():
        if ms.get("default_map"):
            maps = [("upstream_default", None)]
        else:
            maps = [(e.name, e) for e in load_suite(ms["suite"]) if e.split == ms["split"]]
        for n in cfg["num_agents"]:
            roles = RoleConfig(num_learners=1, num_bystanders=n - 2)
            # per population: arrays (maps, units) of per-unit means
            table = {k: {"total": [], "switcher": [], "stock": [], "regrown": []} for k in POPULATIONS}
            for mi, (name, entry) in enumerate(maps):
                t0 = time.perf_counter()
                kw = {} if entry is None else {"map_ASCII": entry.ascii(ms["teacher_present_version"]),
                                               "grid_size": entry.grid_size}
                env = EthicsHarvest(roles, num_inner_steps=steps, **kw)
                policy = ScriptedPolicy(env)
                key = jax.random.fold_in(jax.random.PRNGKey(int(cfg["seed"])), mi * 100 + n)
                for pop, (base, sw) in POPULATIONS.items():
                    r = jax.device_get(run(env, policy, key, num_envs, steps, population(n, base, sw)))
                    apples = np.asarray(r["apples"], float)                    # (E, N)
                    table[pop]["total"].append(unit_means(apples.sum(1), units))
                    table[pop]["switcher"].append(unit_means(apples[:, SWITCHER], units))
                    table[pop]["stock"].append(unit_means(
                        np.asarray(r["stock"], float) / np.asarray(r["steps"], float), units))
                    table[pop]["regrown"].append(unit_means(np.asarray(r["regrown"], float), units))
                print(f"{set_name:<9} n={n} {name:<18} {time.perf_counter() - t0:5.1f}s", flush=True)
            T = {k: {m: np.stack(v) for m, v in d.items()} for k, d in table.items()}
            c1 = paired_ci(T["all_sustainable"]["total"], T["all_greedy"]["total"], 2000, rng)
            c2 = paired_ci(T["greedy_among_sustainable"]["switcher"], T["all_sustainable"]["switcher"], 2000, rng)
            c3 = paired_ci(T["all_greedy"]["switcher"], T["sustainable_among_greedy"]["switcher"], 2000, rng)
            holds = {k: bool(v[1] > 0) for k, v in (("C1", c1), ("C2", c2), ("C3", c3))}
            per_map = {
                "C1": float(np.mean(T["all_sustainable"]["total"].mean(1) > T["all_greedy"]["total"].mean(1))),
                "C2": float(np.mean(T["greedy_among_sustainable"]["switcher"].mean(1) > T["all_sustainable"]["switcher"].mean(1))),
                "C3": float(np.mean(T["all_greedy"]["switcher"].mean(1) > T["sustainable_among_greedy"]["switcher"].mean(1))),
            }
            results[f"{set_name}|{n}"] = {
                "map_set": set_name, "num_agents": n, "maps": len(maps),
                "C1_collective_benefit": c1, "C2_temptation": c2, "C3_sucker_cost": c3,
                "holds": holds, "dilemma": all(holds.values()), "share_of_maps": per_map,
                "means": {pop: {m: float(v.mean()) for m, v in d.items()} for pop, d in T.items()},
            }

    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + "\n")

    def f(x, nd=2):
        return f"{x:.{nd}f}"

    lines = ["# Social dilemma check", "",
             f"{units} units x {per_unit} episodes per cell, {steps} steps. Criteria in `{args.config}`.", "",
             "| Map set | Agents | C1 collective benefit | C2 temptation | C3 sucker's cost | Dilemma |",
             "|---|---|---|---|---|---|"]
    for r in results.values():
        cells = []
        for k, lab in (("C1_collective_benefit", "C1"), ("C2_temptation", "C2"), ("C3_sucker_cost", "C3")):
            e, lo, hi = r[k]
            cells.append(f"{f(e)} [{f(lo)}, {f(hi)}] ({'holds' if r['holds'][lab] else 'no'}, "
                         f"{r['share_of_maps'][lab]:.0%} of maps)")
        lines.append(f"| {r['map_set']} | {r['num_agents']} | " + " | ".join(cells) +
                     f" | {'yes' if r['dilemma'] else 'no'} |")
    lines += ["", "Units: C1 in total apples per episode (all agents), C2 and C3 in apples per episode "
                  "for the switching agent.", "",
              "| Map set | Agents | Population | Total apples | Switcher apples | Mean stock | Regrown |",
              "|---|---|---|---|---|---|---|"]
    for r in results.values():
        for pop, m in r["means"].items():
            lines.append(f"| {r['map_set']} | {r['num_agents']} | {pop} | {f(m['total'], 1)} | "
                         f"{f(m['switcher'])} | {f(m['stock'])} | {f(m['regrown'])} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
