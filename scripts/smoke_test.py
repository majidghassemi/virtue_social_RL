"""Phase 0 smoke test: random-policy rollout on SocialJax Commons Harvest Open.

Compiles a jit + vmap rollout (lax.scan over time, vmap over environments),
then reports compile time and steady-state environment steps per second.

Usage:
    python scripts/smoke_test.py                       # defaults: 1024 envs x 1000 steps
    python scripts/smoke_test.py --num-envs 64 --num-steps 200
    python scripts/smoke_test.py --out results/speed_cpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import socialjax


def make_rollout(env, num_envs: int, num_steps: int):
    num_agents = env.num_agents
    num_actions = env.action_space(0).n

    def rollout(key):
        key, reset_key = jax.random.split(key)
        obs, state = jax.vmap(env.reset)(jax.random.split(reset_key, num_envs))

        def step(carry, _):
            state, key = carry
            key, act_key, step_key = jax.random.split(key, 3)
            actions = jax.random.randint(act_key, (num_envs, num_agents), 0, num_actions)
            obs, state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(step_key, num_envs), state, actions
            )
            return (state, key), jnp.sum(reward)

        (_, _), rewards = jax.lax.scan(step, (state, key), None, length=num_steps)
        return obs, jnp.sum(rewards)

    return jax.jit(rollout)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="harvest_common_open")
    p.add_argument("--num-agents", type=int, default=7)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    env = socialjax.make(args.env, num_agents=args.num_agents, shared_rewards=False)
    obs, _ = env.reset(jax.random.PRNGKey(args.seed))
    print(f"jax {jax.__version__} | devices: {jax.devices()}")
    print(f"env {args.env} | agents {env.num_agents} | actions {env.action_space(0).n} "
          f"| obs per env {tuple(obs.shape)}")

    rollout = make_rollout(env, args.num_envs, args.num_steps)

    t0 = time.perf_counter()
    jax.block_until_ready(rollout(jax.random.PRNGKey(args.seed)))
    compile_and_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, total_reward = jax.block_until_ready(rollout(jax.random.PRNGKey(args.seed + 1)))
    run = time.perf_counter() - t0

    env_steps = args.num_envs * args.num_steps
    result = {
        "env": args.env,
        "num_agents": env.num_agents,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "device": str(jax.devices()[0]),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "compile_plus_first_run_s": round(compile_and_first, 2),
        "run_s": round(run, 2),
        "env_steps_per_s": round(env_steps / run),
        "agent_steps_per_s": round(env_steps * env.num_agents / run),
        "random_policy_total_reward": float(total_reward),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
