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
.venv/bin/python scripts/make_suite_v2.py --preview results/maps_suite_v2.png  # suite v2 (measures the cost of virtue, about 30 min on 2 CPU cores)
```

## Phase 3 protocol

```bash
.venv/bin/python scripts/run_protocol.py                                         # pre-registered ladder (configs/protocol_v1.yaml)
.venv/bin/python scripts/analyze_protocol.py                                     # results/protocol_v1_summary.{json,md}
.venv/bin/python scripts/run_protocol.py --config configs/protocol_crossed_v1.yaml  # crossed virtue x competence ladder
.venv/bin/python scripts/analyze_crossed.py                                      # results/crossed_v1_summary.{json,md}
```

Both run scripts skip maps that are already finished, so an interrupted run resumes where it stopped.

## Layout

| Path | Contents |
|---|---|
| `third_party/SocialJax` | SocialJax, pinned as a git submodule at commit `18e2171` |
| `src/virtue_rl` | this project's code (`envs/` ethics wrapper and maps, `agents/` scripted and learned-slot policies, `metrics/`, `train/` PPO trainers) |
| `scripts/` | runnable entry points |
| `tests/` | unit tests |
| `configs/` | YAML configs (roles, environment, map suite) |
| `maps/` | generated map suites (JSON): `suite_v1` for Phase 3, `suite_v2` for Phase 4 |
| `results/` | small result files (benchmarks, map checks, previews) |
| `NOTES.md` | environment facts this project relies on, with source line numbers |

SocialJax is not packaged upstream, so `pyproject.toml` exposes its `socialjax` package directly from the submodule. Dependency versions follow SocialJax's own pins.

## Progress

| Phase | Status | What it delivers |
|---|---|---|
| 0. Setup | Done | Pinned SocialJax, installable package, smoke tests, CPU speed benchmark (about 20,900 env steps per second on 2 cores) |
| 1. Ethics wrapper | Done | Per-agent ethics record on every step, teacher channel in observations, role configuration |
| 2. Maps and splits | Done | Seeded map generator, 24-map suite in three splits, teacher-absent condition, provisional cost bands, per-map runner |
| 3. Scripted ladder and metrics | Done | Scripted agents with known dispositions, behaviour and validity metrics, pre-registered protocol, full run on 24 maps, crossed virtue x competence ladder, map suite v2 balanced on the measured cost of virtue, protocol v2 with known groups not defined by the metric's rule, social dilemma check (3 agents chosen), protocol v2 repeated at 3 agents |
| 4. Learned agents | In progress | Code done and tested on CPU: trainer for teachers with a prosocial disposition (4a) and the social learner of Ndousse et al. 2021 (4b). Training runs on GPU next. Later: PPO with harm penalty, behaviour cloning |
| 5. Analysis | Planned | Figures and a script that regenerates every reported number from raw logs |

Test status: 104 tests pass (`pytest -q`).

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

### Phase 3 full run (all 24 maps, 10 seed units x 32 episodes)

Full report: `results/protocol_v1_summary.md`. About 1.02 million agent-episodes, 438,000 of them learners. Adding the last two maps did not change any verdict from the 22-map preliminary analysis.

| Pre-registered check | Result |
|---|---|
| Rank validity, all three metrics | Pass (tau-b 1.0 in every split) |
| Competence independence | Harm per episode fails (0.25), harm per opportunity fails (0.16), harm per apple passes (at most 0.05) |
| Teacher gap detects performative agents | Pass (gap 0.84, 95% CI 0.81 to 0.84) |
| Teacher gap null for virtuous and indifferent agents | Fails: the indifferent rung has a small but significant gap (-0.007, CI -0.010 to -0.004) |

Other results: about 64 agent-episodes per map and rung give a standard error of 0.05, zap metrics separate aggressive agents perfectly, and the structural cost of virtue correlates only weakly with the measured cost (Spearman 0.30). The exploratory onset metric gives a performative gap of 0.40 instead of 0.84, because idle opportunities no longer inflate the denominator. For performative agents the within-episode gap is 0.95 by "teacher in view" but 0.24 by "seen by the teacher".

### Phase 3 completion (decided after the full run)

| Addition | Status | Result |
|---|---|---|
| Crossed virtue x competence ladder: 5 w values x 5 eps values (0 to 0.5), seen maps, teacher absent | Pre-registered (criteria and predictions committed in `636a29c` before the run) | All three metrics reach the gamma threshold of 0.8, but harm per episode only exactly at it (see below) |
| Teacher gap minus the indifferent rung's gap | Post hoc | Does not remove the ecological effect, because that effect differs between rungs |
| Equivalence view of the teacher gap (95% CI inside +-0.05) | Post hoc, proposed for the Phase 4 pre-registration | Every non-performative gap is inside the margin, the performative gap is far outside it, for both opportunity metrics |
| Map suite v2 with two empirical cost bands | Design change, used from Phase 4 on | See below |

Crossed ladder (`results/crossed_v1_summary.md`, 288,000 learner agent-episodes). Gamma is computed over the 250 pairs of agent types with different w.

| Metric | Prediction | Gamma | Misordered pairs | Worst misorder |
|---|---|---|---|---|
| Harm per episode | Fails | 0.800 (threshold) | 10.0% | A clumsy indifferent agent (w 0, eps 0.5) scores 0.62 below a skilled w 0.25 agent |
| Harm per opportunity | Unsure | 0.952 | 2.4% | Same pair, by 0.15 |
| Harm per apple | Passes | 1.000 | 0% | None |
| Harm per onset (exploratory) | Not predicted | 0.928 | 3.6% | Same pair, by 0.15 |

Harm per episode passes only exactly at the threshold (25 of 250 pairs misordered), and its worst map has gamma 0.72. By the pre-registered rule it is a pass, so the predicted failure did not happen. The confound still shows in the size of the worst misorder, close to half the ladder's range. Harm per opportunity also falls with noise, because a random lapse next to a last apple turns an opportunity into a non-harm. Harm per apple orders every pair correctly: it is the only metric that is independent of competence on both tests. Caveat: competence here is lowered by random lapses only. Learned agents may be less competent in other ways (slow routing, poor exploration), so Phase 4 should check this again.

Teacher-gap nulls: the indifferent-rung adjustment works for the per-opportunity metric (the w 0.25 to 0.75 rungs then include zero, and the virtuous rung, which never harms, is only shifted by the control gap), but under the onset metric the mixture rungs at w 0.5 and 0.75 have larger ecological gaps (-0.023) than the indifferent rung (-0.007), so the additive adjustment leaves a significant -0.016. A fixed equivalence margin is simpler and robust: the largest non-performative gap is 0.023 in absolute value (its CI reaches 0.031), while the performative gaps are 0.40 and 0.84.

Map suite v2 (`maps/suite_v2.json`, preview `results/maps_suite_v2.png`). Phase 4 uses it. Phase 3 results stay on suite v1.

| Item | Value |
|---|---|
| Seen maps | The 12 v1 seen maps, unchanged |
| Held-out maps | 8 per split, selected in seed order until each band quota was filled (held-out position 4 + 4, held-out structure 2 per family and band) |
| Bands | Two, split at the median measured cost of the seen maps (0.139) |
| Cost measurement | Scripted learners w 0 against w 1, eps 0, teacher absent, 320 episodes, both rungs on the same random keys. Median SE 0.009 |
| Reproducibility | r = 0.94 against the protocol_v1 measurement on the seen maps |
| Map checks | 56 of 56 pass |

Why two bands: 8 maps per split balance evenly into two bands, not three. Balanced band counts do not mean matched distributions: the held-out-structure costs spread much more widely (SD 0.19 against 0.07 for seen maps), and one divided map has a cost of -0.28 (virtue pays there). Phase 4 should therefore use each map's measured cost as a covariate, not only its band.

### Protocol v2: known groups not defined by the metric's rule

Design principle: the environment and the tests should neither be built to confirm our assumptions nor be built to break them. An audit of Phase 3 against it (full table in `NOTES.md`) found that the Phase 3 "virtuous" agent avoids exactly the last apples the metric counts, so it is really a "minimal compliance" agent, and that performative agents only reacted to the cue later used in the analysis. Protocol v2 (`configs/protocol_v2.yaml`) adds groups whose rules come from the environment, not from the metric. The groups, questions, predictions and criteria were committed in `e924e33` and the analysis script in `b3386c3`, both before the run. Full report: `results/protocol_v2_summary.md`. Suite v2, 28 maps, 10 seed units x 16 episodes, 215,000 learner agent-episodes.

| Group | Definition |
|---|---|
| Indifferent | Takes the nearest apple of any kind |
| Minimal | Never takes a last apple (the Phase 3 "virtuous" agent, unchanged) |
| Sustainable | Only takes an apple from a patch with more than 3 apples (fastest regrowth needs 3 or more apple neighbours) |
| View-only variants | See apples only inside their own 11 x 11 view |
| Performative, three cues | Act like minimal agents only when the teacher is in their view, when they are in the teacher's view, or when the teacher is within distance 5 |

| Question | Prediction | Result |
|---|---|---|
| Q1 Does the care order show in what happens to the commons? | minimal above indifferent, sustainable unsure | Yes. Apples regrown per episode: indifferent 0.53, minimal 0.69, sustainable 1.16. Full order on 26 of 28 maps, sustainable above minimal on all 28 |
| Q2 How much do the harm metrics miss? | No prediction | Minimal and sustainable both score 0 on every harm metric (by construction), yet the regrowth gain from minimal to sustainable is 2.8 times the gain from indifferent to minimal (CI 2.2 to 3.8). The sustainable rule costs 18% of apples |
| Q3 Competence independence under limited view (max 0.10) | Harm per episode fails, others unsure | Harm per episode fails (0.42 and 0.33). Harm per opportunity (0.015 at most), harm per apple (0.066 at most) and harm per onset (0.049 at most) pass, although view-only agents collect 39 to 46% fewer apples |
| Q4 Teacher gap for three cues (detection above +0.05, nulls inside +-0.05) | All detected | All 12 checks pass. Gaps per opportunity 0.84 to 0.85 for all three cues, per onset 0.41 to 0.43. The indifferent null is -0.011 (CI -0.016 to -0.006): inside the margin, but it would again fail a zero-null |
| Q5 Within-episode gaps by recorded cue | Each group largest on its own cue | Yes. In-view agents: 0.95 by their cue, 0.22 by the other. Seen agents: 0.95 against 0.27. Near agents: 0.31 and 0.27, so an analyst with either recorded cue underestimates them about threefold |

What this adds to Phase 3:

* The harm metrics measure compliance with the last-apple rule, not care for the commons. They rank indifferent against minimal correctly but cannot see the larger difference between minimal and sustainable.
* Competence independence depends on the kind of low competence. Harm per opportunity failed under random lapses (0.16 in Phase 3) but passes under limited view (0.015). Harm per apple passes under both.
* The condition-level teacher gap detects performative agents whatever cue they use. Within-episode splits only work when the analyst records the agent's own cue.

Caveat on the environment: the commons is nearly empty for most of each episode in every group (mean stock about 0.5 to 0.6 apples, about 1 apple regrown per 1,000 steps). The same happens on the upstream default map (about 1.1 apples, 2 to 3.5 regrown per episode with the same agents), so it comes mainly from having 4 or more greedy agents out of 7, not only from our maps. Our maps also hold fewer apples than the upstream map (median 30 against 64). See the open decision in `NOTES.md`.

### Protocol v2 at 3 agents (1 teacher, 1 learner, 1 bystander)

The dilemma check chose 3 agents for Phase 4, so protocol v2 was repeated in that setting with the same groups, questions and criteria (`configs/protocol_v2_3agents.yaml`, committed in `8001726` before the run). The only changes are the roles, a new seed, 32 episodes per seed unit instead of 16, and the prediction for the indifferent null in Q4, which was changed to "unsure" because the teacher is now a third of the population. Full report: `results/protocol_v2_3agents_summary.md`. Suite v2, 28 maps, 10 seed units x 32 episodes, 143,000 learner agent-episodes.

| Question | 7 agents | 3 agents |
|---|---|---|
| Q1 Apples regrown per episode (indifferent, minimal, sustainable) | 0.53, 0.69, 1.16. Full order on 26 of 28 maps | 0.9, 1.4, 2.9. Full order on all 28 maps |
| Q1 Mean apple stock (same groups) | 0.52, 0.59, 0.61 | 0.87, 0.99, 1.06 |
| Q2 Blind ratio (regrowth gain from minimal to sustainable over the gain from indifferent to minimal) | 2.8 (CI 2.2 to 3.8) | 3.1 (CI 2.8 to 3.4) |
| Q2 Apples given up by the sustainable rule | 18% | 13.5% |
| Q3 Largest relative change under limited view, harm per episode (max 0.10) | 0.42, fails | 0.52, fails |
| Q3 Same, harm per opportunity, per apple, per onset | 0.015, 0.066, 0.049, all pass | 0.009, 0.074, 0.072, all pass |
| Q4 Teacher gap per opportunity, three cues | 0.84 to 0.85 | 0.94 to 0.95 |
| Q4 Teacher gap per onset, three cues | 0.41 to 0.43 | 0.47 to 0.51 |
| Q4 Indifferent null (margin +-0.05) | -0.011 (CI -0.016 to -0.006), passes | +0.003 (CI 0.001 to 0.004), passes |
| Q5 Near agents, gaps by the two recorded cues | 0.31 and 0.27 | 0.30 and 0.33 |

All 16 pass/fail checks give the same verdict as at 7 agents. The conclusions in the previous section hold in the Phase 4 setting: the harm metrics see compliance with the last-apple rule but not the larger gain from the sustainable rule, competence independence holds under limited view for every metric except harm per episode, and the condition-level teacher gap detects all three cues while within-episode splits miss the near cue about threefold. With one bystander instead of three, the commons holds more apples and the order of the groups is cleaner, but the mean stock is still about 1 apple, so the commons is mostly depleted at 3 agents as well.

### Phase 4: learned teachers and social learners (code ready, not yet trained)

Why: every ethical agent in Phases 3 and protocol v2 follows a hand-written rule, the teacher included. A virtuous disposition is not a rule, so the thesis needs a teacher whose ethics sits in its objective and a learner that is never given that objective.

| Stage | Agent | Training signal | Config |
|---|---|---|---|
| 4a teachers | All 3 agents share one network and one disposition lam (0, 0.5 or 1), 3 seeds each | Own apples + lam x mean apples of the other agents | `configs/train_prosocial_v1.yaml` |
| 4b learners | Slot 1 learns. Slot 0 is a frozen 4a teacher, slot 2 a frozen lam 0 agent | Own apples only (lam 0), plus the auxiliary next-observation loss | `configs/train_social_v1.yaml` (draft) |

Learner conditions: social learner with a lam 1 teacher (main), with a lam 0 teacher (separates acquiring virtue from copying competence), the same network without the auxiliary loss (social learning against plain RL), and with the teacher walled in (no teacher).

Implementation (`src/virtue_rl/train/`):

* `ppo.py`: independent PPO following SocialJax's IPPO baseline (same network, loss, GAE and learning-rate schedule, SocialJax's Harvest defaults). Adds several maps per run (one group of parallel envs per map), the ethics wrapper, disposition rewards, frozen slots, and chunked, resumable training.
* `social.py`: recurrent PPO with the auxiliary predictive loss of Ndousse et al. 2021 (conv layers, 192-unit LSTM, decoder that predicts the next observation from the LSTM output and the action, mean absolute error, c_aux = 3). The choices the paper leaves open are listed in the module docstring.
* `rewards.py`: the prosocial reward above and the SVO reward of McKee et al. 2020 (as an option, not used yet).
* `scripts/train.py` runs one (condition, seed) or a Slurm array index. `scripts/slurm/train_prosocial.sh` and `train_social.sh` submit all runs.

Measured on this 2-core CPU: one update over the 12 training maps compiles in about 2 minutes and runs at about 460 environment steps per second, so a 1e8-step run needs a GPU. Run `scripts/train.py --bench 3` on a GPU node first to set the job time.

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
