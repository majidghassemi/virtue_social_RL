# virtue_social_RL

Virtue ethics with social reinforcement learning. The first study asks whether behavioural ethics metrics validly measure dispositions in multi-agent agents, using SocialJax Commons Harvest as the environment. Later work scales the same pipeline to Multi-Agent Craftax.

## Setup

Requires Python 3.10 to 3.12 (SocialJax supports `>=3.10,<3.13`).

```bash
git clone --recurse-submodules https://github.com/majidghassemi/virtue_social_RL.git
cd virtue_social_RL

# with uv (recommended)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# or with plain pip
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

If you cloned without `--recurse-submodules`, run `git submodule update --init`.

For a CUDA 12 GPU, install the GPU extra instead: `pip install -e ".[dev,gpu]"`. On Compute Canada, check the cluster's documentation for JAX first, since its wheelhouse and CUDA modules may need a different install route.

## Check the install

```bash
.venv/bin/python -m pytest -q          # import, reset and jit/vmap step checks
.venv/bin/python scripts/smoke_test.py # random-policy speed benchmark
```

## Layout

| Path | Contents |
|---|---|
| `third_party/SocialJax` | SocialJax, pinned as a git submodule at commit `18e2171` |
| `src/virtue_rl` | this project's code |
| `scripts/` | runnable entry points |
| `tests/` | unit tests |
| `results/` | small result files (benchmarks, summaries) |
| `NOTES.md` | environment facts this project relies on, with source line numbers |

SocialJax is not packaged upstream, so `pyproject.toml` exposes its `socialjax` package directly from the submodule. Dependency versions follow SocialJax's own pins.
