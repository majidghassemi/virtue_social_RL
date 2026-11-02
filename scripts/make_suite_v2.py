"""Build map suite v2: held-out splits balanced on the measured cost of virtue.

Usage:
    python scripts/make_suite_v2.py                    # configs/maps_v2.yaml
    python scripts/make_suite_v2.py --preview results/maps_suite_v2.png

Steps:
1. Take the seen maps from suite v1 unchanged and measure their empirical
   cost of virtue: (apples(w = 0) - apples(w = 1)) / apples(w = 0) for the
   scripted learners, teacher absent, eps 0.
2. Band cut = median of the seen maps' costs (two bands, low and high).
3. For each held-out split, walk candidate seeds in order, apply the v1
   constraints (family, apple overlap with seen maps for held-out position),
   measure each candidate's cost and accept it while its (family, band)
   quota is open.

Measurements are cached by (family, seed), so reruns are cheap and the
selection is deterministic given the config.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from make_maps import apple_cells, make_entry, overlap, preview  # noqa: E402

from virtue_rl.agents import ScriptedPolicy, make_params  # noqa: E402
from virtue_rl.envs import RoleConfig  # noqa: E402
from virtue_rl.envs import maps as M  # noqa: E402
from virtue_rl.envs.wrapper import EthicsHarvest  # noqa: E402
from virtue_rl.metrics.behaviour import episode_counts  # noqa: E402
from virtue_rl.runner import rollout  # noqa: E402

ROLES = RoleConfig(num_learners=3, num_bystanders=3)


@partial(jax.jit, static_argnames=("env", "policy", "num_envs", "num_steps"))
def learner_apples(env, policy, key, num_envs, num_steps, params):
    c = episode_counts(rollout(env, policy, key, num_envs=num_envs, num_steps=num_steps,
                               policy_params=params)["ethics"])
    return c["apples"]


def measure_cost(entry: M.MapEntry, mc: dict) -> dict:
    """Empirical cost of virtue with a unit-level standard error."""
    units, per_unit, steps = int(mc["units"]), int(mc["envs_per_unit"]), int(mc["num_steps"])
    env = EthicsHarvest(ROLES, map_ASCII=entry.ascii(False), grid_size=entry.grid_size,
                        num_inner_steps=steps)
    policy = ScriptedPolicy(env)
    key = jax.random.fold_in(jax.random.PRNGKey(int(mc["seed"])), entry.seed)
    learners = np.asarray(ROLES.learner_ids)
    per_unit_apples = {}
    for w in (0.0, 1.0):
        a = np.asarray(jax.device_get(learner_apples(env, policy, key, units * per_unit, steps,
                                                     make_params(ROLES, "mixture", w, 0.0))))
        per_unit_apples[w] = a[:, learners].reshape(units, per_unit * len(learners)).mean(1)
    a0, a1 = per_unit_apples[0.0], per_unit_apples[1.0]
    cost = (a0.mean() - a1.mean()) / a0.mean()
    # Unit-level bootstrap SE (paired by unit, since both rungs share keys).
    rng = np.random.default_rng(0)
    idx = rng.integers(0, units, (1000, units))
    boots = (a0[idx].mean(1) - a1[idx].mean(1)) / a0[idx].mean(1)
    return {"apples_indifferent": float(a0.mean()), "apples_virtuous": float(a1.mean()),
            "empirical_virtue_cost": float(cost), "cost_se": float(boots.std())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/maps_v2.yaml"))
    p.add_argument("--preview", type=Path, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    mc = cfg["cost_measurement"]
    cache_path = Path(cfg["cache"])
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def cost_of(entry: M.MapEntry) -> dict:
        k = f"{entry.family}:{entry.seed}"
        if k not in cache:
            t0 = time.perf_counter()
            cache[k] = measure_cost(entry, mc)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1) + "\n")
            print(f"  measured {entry.name:<28} seed {entry.seed:>5}: cost "
                  f"{cache[k]['empirical_virtue_cost']:+.3f} (se {cache[k]['cost_se']:.3f})"
                  f"  {time.perf_counter() - t0:.0f}s", flush=True)
        return cache[k]

    n = int(cfg["num_agents"])
    seen = [e for e in M.load_suite(cfg["splits"]["seen"]["from_suite"]) if e.split == "seen"]
    for e in seen:
        e.stats = e.stats | cost_of(e)
    cut = float(np.median([e.stats["empirical_virtue_cost"] for e in seen]))
    for e in seen:
        e.band = "low" if e.stats["empirical_virtue_cost"] <= cut else "high"
    seen_apples = [apple_cells(e.ascii_present) for e in seen]
    entries = list(seen)
    report = {}

    for split in ("heldout_position", "heldout_structure"):
        sc = cfg["splits"][split]
        families = sc["families"]
        per_cell = int(sc["count"]) // (len(families) * len(cfg["bands"]))
        quota = {(f, b): per_cell for f in families for b in cfg["bands"]}
        seed, made, tried = int(sc["first_seed"]), [], 0
        rejected_overlap = 0
        while any(v > 0 for v in quota.values()):
            if tried >= int(cfg["max_candidates_per_split"]):
                raise RuntimeError(f"{split}: quotas still open after {tried} candidates: {quota}")
            family = families[(seed - int(sc["first_seed"])) % len(families)]
            cand_seed = seed
            seed += 1
            if all(quota[(family, b)] == 0 for b in cfg["bands"]):
                continue
            e = make_entry(split, len(made), family, cand_seed, cfg["patch"], n)
            if split == "heldout_position":
                cells = apple_cells(e.ascii_present)
                if any(overlap(cells, s) > sc["max_apple_overlap"] for s in seen_apples):
                    rejected_overlap += 1
                    continue
            tried += 1
            e.stats = e.stats | cost_of(e)
            e.band = "low" if e.stats["empirical_virtue_cost"] <= cut else "high"
            if quota[(family, e.band)] == 0:
                continue
            quota[(family, e.band)] -= 1
            made.append(e)
        report[split] = {"candidates_measured": tried, "rejected_for_overlap": rejected_overlap}
        entries.extend(made)

    meta = {"config": cfg, "band_cut_empirical_cost": cut,
            "band_note": "empirical cost of virtue, two bands split at the median of the seen maps",
            "selection": report, "grid_size": list(entries[0].grid_size)}
    M.save_suite(entries, cfg["out"], meta)
    print(f"wrote {len(entries)} maps to {cfg['out']}, band cut {cut:.3f}")
    for e in entries:
        s = e.stats
        print(f"  {e.name:<30} cost {s['empirical_virtue_cost']:+.3f} (se {s['cost_se']:.3f}) "
              f"structural {s['structural_virtue_cost']:.3f} band {e.band}")
    print(json.dumps(report))
    if args.preview:
        preview(entries, args.preview, "empirical_virtue_cost", "empirical cost")


if __name__ == "__main__":
    main()
