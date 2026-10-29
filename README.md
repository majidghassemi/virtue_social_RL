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
| 3. Scripted ladder and metrics | Code and pilot done, full run 22 of 24 maps (stopped at the 2-hour job limit) | Scripted agents with known dispositions, behaviour and validity metrics, pre-registered protocol, pilot results |
| 4. Learned agents | Planned | PPO, PPO with harm penalty, SVO, behaviour cloning, social learner (needs the GPU benchmark first) |
| 5. Analysis | Planned | Figures and a script that regenerates every reported number from raw logs |

Test status: 77 tests pass (`pytest -q`).

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

### Phase 3 decisions (approved 29 Oct 2026)

| Decision | Choice | Reason |
|---|---|---|
| Commons harm timing | Judged on the grid the agent saw | Dispositions are choices given what the agent knew |
| `opp_commons` | Strict: a last apple in one of the 4 cells the agent can move into | Clear, and harm never exceeds opportunity |
| Roles | 1 teacher, 3 learners, 3 bystanders | Standard 7-agent setting |
| Watched cue | Performative agents react to `teacher_in_view`. `seen_by_teacher` is recorded as well | The agent can only perceive the first. The second tests the proposal's assumption that presence is the relevant cue |
| Bystanders | Scripted indifferent | Realistic opportunities, less noise than random agents |
| Teacher | Scripted virtuous | Its behaviour is known exactly |
| Ladder | Mixture w in {0, 0.25, 0.5, 0.75, 1}, plus performative and aggressive | Graded known order, plus separate groups for the teacher gap and zap metrics |
| Noise | eps in {0.1, 0.2}: random move or stay, never a zap, cancelled if a virtuous-mode agent would step onto a last apple | Lowers competence without changing morality |
| Run size | 10 seed units x 32 episodes per job, 1,000-step episodes | Reliability curves then show whether that is enough |
| Success criteria | Fixed in `configs/protocol_v1.yaml` before any run | Proposal Q8. Failures are findings |
| Cost of virtue | Empirical: apples(indifferent) minus apples(virtuous), as a share, per map | Replaces the provisional structural bands |

### Phase 3 full run, preliminary (22 of 24 maps, 10 seed units x 32 episodes)

The full run stopped at the 2-hour limit for background jobs after finishing 22 maps. The two missing maps are `heldout_structure_04_rooms` and `heldout_structure_05_divided`, both in the low-cost band, so the held-out-structure split is currently unbalanced. `scripts/run_protocol.py` resumes by skipping finished maps. Full report: `results/protocol_v1_partial22_summary.md`.

| Pre-registered check | Result |
|---|---|
| Rank validity, all three metrics | Pass (tau-b 1.0 in every split) |
| Competence independence | Harm per episode fails (0.25), harm per opportunity fails (0.16), harm per apple passes (at most 0.05) |
| Teacher gap detects performative agents | Pass (gap 0.83, 95% CI 0.81 to 0.84) |
| Teacher gap null for virtuous and indifferent agents | Fails: the indifferent rung has a small but significant gap (-0.007, CI -0.011 to -0.004) |

With ten times the pilot's data, the small ecological effect of the teacher's presence becomes significant. It is about 1% of the performative gap. The exploratory onset metric ranks the ladder perfectly too, gives a much smaller performative gap (0.39) because idle opportunities no longer inflate the denominator, and narrowly fails competence independence (0.12). The cue result holds: for performative agents the within-episode gap is 0.95 by "teacher in view" but 0.24 by "seen by the teacher".

### Phase 3 pilot results (4 seed units x 8 episodes, all 24 maps)

Full report: `results/protocol_v1_pilot_summary.md`. These are pilot numbers. The pre-registered full run replaces them.

| Finding | Detail |
|---|---|
| All three harm metrics rank the ladder perfectly | Kendall tau-b = 1.0 in every split. Graded mixtures were not enough to make the scripted ladder hard to rank |
| Only harm per apple is competence-independent | Relative change under noise: harm per episode up to 0.24 (fails), harm per opportunity up to 0.14 (fails), harm per apple at most 0.06 (passes) |
| The teacher gap detects performative agents | Gap 0.74 for performative agents. The virtuous and indifferent rungs have gaps whose intervals include zero |
| The teacher gap is not perfectly clean | The w = 0.5 rung shows a small but significant negative gap (-0.03), so the teacher's own presence changes the ecology slightly |
| The cue matters | For performative agents the within-episode gap is 0.96 when split by "teacher in view" but only 0.36 when split by "actually seen by the teacher" |
| Reliability | About 64 agent-episodes per map and rung give a standard error of 0.05 for harm per opportunity |
| Zap metrics | Aggressive agents separate perfectly from all others |
| Structural cost is a poor proxy | Spearman correlation 0.25 with the empirical cost of virtue, so the empirical bands should be used |
| Per-step opportunity counts are fragile | A performative agent waiting next to a last apple while watched adds an opportunity every step (up to about 1,000 in one episode), which skews pooled ratios and their bootstrap intervals. An exploratory onset-based metric (each temptation counted once) was added before the full run |
