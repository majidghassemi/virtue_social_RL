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

## Maps

```bash
.venv/bin/python scripts/make_maps.py --preview results/maps_suite_v1.png  # regenerate the suite
.venv/bin/python scripts/check_maps.py                                    # invariants on every map
```

## Layout

| Path | Contents |
|---|---|
| `third_party/SocialJax` | SocialJax, pinned as a git submodule at commit `18e2171` |
| `src/virtue_rl` | this project's code (`envs/` holds the ethics wrapper) |
| `scripts/` | runnable entry points |
| `tests/` | unit tests |
| `configs/` | YAML configs (roles, environment, map suite) |
| `maps/` | generated map suites (JSON) |
| `results/` | small result files (benchmarks, map checks, previews) |
| `NOTES.md` | environment facts this project relies on, with source line numbers |

SocialJax is not packaged upstream, so `pyproject.toml` exposes its `socialjax` package directly from the submodule. Dependency versions follow SocialJax's own pins.

## Progress

| Phase | Status | What it delivers |
|---|---|---|
| 0. Setup | Done | Pinned SocialJax, installable package, smoke tests, CPU speed benchmark (about 20,900 env steps per second on 2 cores) |
| 1. Ethics wrapper | Done | Per-agent ethics record on every step, teacher channel in observations, role configuration |
| 2. Maps and splits | Done | Seeded map generator, 24-map suite in three splits, teacher-absent condition, provisional cost bands, per-map runner |
| 3. Scripted ladder and metrics | Next | Scripted agents with known dispositions, behaviour and validity metrics, full protocol run |
| 4. Learned agents | Planned | PPO, PPO with harm penalty, SVO, behaviour cloning, social learner (needs the GPU benchmark first) |
| 5. Analysis | Planned | Figures and a script that regenerates every reported number from raw logs |

Test status: 51 tests pass (`pytest -q`).

## Design decisions

Each choice below has a short reason. `NOTES.md` has the full detail with source line numbers.

### Environment

| Decision | Reason |
|---|---|
| SocialJax Commons Harvest Open | GPU-native (JAX), a standard social dilemma benchmark, configurable number of agents, and two kinds of harm (zapping others and destroying the commons) |
| SocialJax pinned at commit `18e2171`, not release v1.1.0 | v1.1.0 is only needed to reproduce the SocialJax paper. Later commits fix an agent-overlap bug |
| SocialJax wrapped, never edited | Upstream fixes can be pulled freely, and reviewers see an unmodified standard environment |
| `shared_rewards=False` enforced | The upstream default gives every agent the summed reward, which removes the dilemma |

### Ethics record (`src/virtue_rl/envs/wrapper.py`)

| Field | Definition | Why this definition |
|---|---|---|
| `harm_zap` | zapped while another agent stood in one of the 4 zap cells | Uses the environment's own hit test, so harm means the victim really was sent back to spawn |
| `opp_zap` | another agent stood in the zap cells at the start of the step | An opportunity is something the agent could act on, so it is measured at decision time |
| `harm_commons` | collected an apple with no other apple within radius 2 | Matches the regrowth rule: a cell with no apple neighbours has zero chance of regrowing |
| `opp_commons` | such a "last apple" was in one of the 4 cells the agent could move into | Strict reading of "had the chance this step", and it guarantees harm never exceeds opportunity |
| `apples` | apples collected this step | Reported next to harm so that doing nothing never counts as virtue |
| `teacher_in_view` | teacher inside the agent's view in the observation it acted on | Performative behaviour depends on being watched at the moment of choice |
| `valid` | False on the last step of an episode | SocialJax resets the state and zeroes rewards inside that step, so it cannot be measured |

Commons harm is judged on the grid the agent saw, because the study is about dispositions, which are choices given what the agent knew.

Other wrapper choices:

| Decision | Reason |
|---|---|
| A teacher channel is appended to observations | The native observation puts all other agents in one channel, so a learner could not tell the teacher apart |
| Apple counts taken modulo 256 | The upstream inventory is an 8-bit integer that wraps after 127 apples |
| Default roles: 1 teacher, 3 learners, 3 bystanders | 7 agents is the standard SocialJax setting. Bystanders allow harm to others without involving the teacher |
| Plain YAML configs | Small configs and no extra dependency. Hydra can replace it when Phase 4 sweeps grow |

### Maps (`src/virtue_rl/envs/maps.py`, `maps/suite_v1.json`)

| Decision | Reason |
|---|---|
| 16 x 22 arena, apple patches at the top, spawn cells at the bottom | Same scale as the standard Harvest Open map |
| Patches of 1, 5 or 13 apples that never share a regrowth neighbourhood | Each patch is exactly one regrowth component, which makes the cost of virtue exact |
| Families: `open` for training, `divided` and `rooms` held out | The structure split tests walls and doorways the agents never trained with |
| Held-out-position maps share at most half their apple cells with any training map | Prevents near-duplicates of training layouts |
| Wall ring around every map | Neutralises an upstream bug where zaps at the map edge wrap to the opposite edge or hit the zapper itself |
| Sealed teacher bay plus a single teacher start cell | Gives a teacher-absent condition without changing the number of agents. Both conditions share the same grid except two cells |
| Bay placed 9 columns beyond the arena | Walls do not block vision in SocialJax, so distance is the only way to hide the teacher. Checked exactly for every reachable cell and heading |
| Structural cost of virtue = patches / apples | The share of the starting stock a virtuous agent must leave: within a patch every apple but the last can be taken harmlessly |
| Cost bands from tertiles of the training maps, held-out splits stratified by band | A held-out result then reflects the layout shift, not a change in how costly virtue is. Bands are provisional until Phase 3 measures the cost with scripted agents |

### Upstream issues found

| Issue | How it is handled |
|---|---|
| Zaps at the map edge wrap to the opposite edge or hit the zapper | Wall ring on every generated map, regression test, self-hits never count as harm |
| `step_env` resets internally on the last step and zeroes its rewards | That step is marked `valid = False` and excluded from metrics |
| Inventory stored as `int8` | Apple counts computed modulo 256 |
| Other agents collapsed into one observation channel | Teacher channel added by the wrapper |

### Open choices to confirm before Phase 3

1. Commons harm judged on the grid the agent saw, rather than the outcome after everyone moved.
2. The strict "adjacent" definition of `opp_commons`.
3. The 1 teacher, 3 learners, 3 bystanders split.
