"""Train a population of agents with a disposition reward (Phase 4a teachers).

Usage:
    python scripts/train.py --config configs/train_prosocial_v1.yaml --lam 0.5 --seed 0
    python scripts/train.py --config configs/train_prosocial_v1.yaml --array-index 4
        # picks (lam, seed) from the config grid, lam-major: index = i_lam * n_seeds + i_seed
    python scripts/train.py --config configs/train_prosocial_v1.yaml --bench 3
        # compiles, times 3 updates, saves nothing

Output, one folder per run (`<out_root>/<name>/lam<lam>_seed<seed>/`):
    config.json           resolved config, git commit, maps, device
    metrics.jsonl         one row per PPO update (apples, harm, regrowth, losses)
    runner.msgpack        latest full training state, used to resume
    params_u<k>.msgpack   policy snapshots every `snapshot_every` updates
    params_final.msgpack  final policy, with a .json file of metadata

A run that is restarted with the same arguments resumes from runner.msgpack.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import yaml

from virtue_rl.envs import RoleConfig
from virtue_rl.envs.maps import load_suite
from virtue_rl.envs.wrapper import EthicsHarvest
from virtue_rl.train import PPOConfig, RewardConfig, Trainer, metrics_to_rows
from virtue_rl.train import checkpoint as ckpt


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def resolve(cfg: dict, args) -> tuple[float, int]:
    if args.array_index is not None:
        lams, seeds = cfg["grid"]["lam"], cfg["grid"]["seed"]
        i = args.array_index
        if not 0 <= i < len(lams) * len(seeds):
            raise SystemExit(f"array index {i} outside 0..{len(lams) * len(seeds) - 1}")
        return float(lams[i // len(seeds)]), int(seeds[i % len(seeds)])
    if args.lam is None or args.seed is None:
        raise SystemExit("give --lam and --seed, or --array-index")
    return float(args.lam), int(args.seed)


def build(cfg: dict, lam: float):
    roles = RoleConfig.from_dict(cfg["roles"])
    ppo = PPOConfig(**cfg["ppo"])
    entries = [e for e in load_suite(cfg["suite"]) if e.split == cfg["split"]]
    if cfg.get("maps"):                         # optional subset, e.g. for quick checks
        entries = [e for e in entries if e.name in set(cfg["maps"])]
    envs = [EthicsHarvest(roles, num_inner_steps=ppo.num_steps,
                          map_ASCII=e.ascii(bool(cfg["teacher_present"])), grid_size=e.grid_size)
            for e in entries]
    n = roles.num_agents
    rc = cfg["reward"]
    reward = RewardConfig(kind=rc["kind"], lam=(lam,) * n,
                          svo_w=None if rc.get("svo_w") is None else (float(rc["svo_w"]),) * n)
    return Trainer(envs, ppo, reward, train_slots=tuple(cfg["train_slots"])), entries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--lam", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--array-index", type=int)
    p.add_argument("--bench", type=int, default=0, help="time this many updates and exit")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.bench and args.lam is None:
        args.lam, args.seed = cfg["grid"]["lam"][-1], cfg["grid"]["seed"][0]
    lam, seed = resolve(cfg, args)

    trainer, entries = build(cfg, lam)
    steps_per_update = trainer.num_envs * trainer.cfg.num_steps
    print(f"lam={lam} seed={seed} maps={len(entries)} envs={trainer.num_envs} "
          f"updates={trainer.num_updates} batch={trainer.batch_size} device={jax.devices()[0]}",
          flush=True)
    runner = trainer.init(jax.random.PRNGKey(seed))

    if args.bench:
        t0 = time.perf_counter()
        runner, m = trainer.run(runner, 1)
        jax.block_until_ready(m)
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        runner, m = trainer.run(runner, args.bench)
        jax.block_until_ready(m)
        dt = time.perf_counter() - t0
        est = trainer.num_updates * dt / args.bench
        print(f"first call (compile + 1 update) {t_compile:.0f}s, {dt / args.bench:.1f}s per update, "
              f"{steps_per_update * args.bench / dt:,.0f} env steps/s, "
              f"full run about {est / 3600:.1f} h", flush=True)
        return

    out = Path(cfg["out_root"]) / cfg["name"] / f"lam{lam:g}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    runner_path, metrics_path = out / "runner.msgpack", out / "metrics.jsonl"
    meta = {"config": cfg, "lam": lam, "seed": seed, "git": git_state(),
            "maps": [e.name for e in entries], "device": str(jax.devices()[0]),
            "num_updates": trainer.num_updates, "env_steps_per_update": steps_per_update}

    if runner_path.exists():
        runner = ckpt.load_runner(runner_path, runner)
        done = int(runner.update)
        rows = [r for r in (json.loads(x) for x in metrics_path.read_text().splitlines() if x)
                if r["update"] <= done] if metrics_path.exists() else []
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"resumed at update {done}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(meta, indent=1) + "\n")
        metrics_path.write_text("")

    chunk, snap = int(cfg["chunk_updates"]), int(cfg["snapshot_every"])
    t_start = time.perf_counter()
    while int(runner.update) < trainer.num_updates:
        start = int(runner.update)
        n = min(chunk, trainer.num_updates - start)
        runner, m = trainer.run(runner, n)
        rows = metrics_to_rows(m, start, steps_per_update)
        ckpt.save_runner(runner_path, runner)          # state first, then metrics (see resume)
        with metrics_path.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        u = int(runner.update)
        if u // snap > start // snap:
            ckpt.save_params(out / f"params_u{u}.msgpack", runner.train_state.params,
                             {**meta, "update": u})
        last = rows[-1]
        print(f"update {u}/{trainer.num_updates} apples {sum(last['apples']):.1f} "
              f"harm_commons {sum(last['harm_commons']):.2f} regrown {last['regrown']:.2f} "
              f"entropy {last['entropy']:.3f} ({time.perf_counter() - t_start:.0f}s)", flush=True)

    ckpt.save_params(out / "params_final.msgpack", runner.train_state.params,
                     {**meta, "update": int(runner.update)})
    print(f"done, {out}", flush=True)


if __name__ == "__main__":
    main()
