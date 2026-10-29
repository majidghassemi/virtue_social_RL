# Environment notes: SocialJax Commons Harvest Open

Facts this project relies on, each with its source location. All line numbers refer to the pinned SocialJax commit `18e21718e52e659bd4cf9c29df2ddfc03dbaca3c` (submodule at `third_party/SocialJax`).

Main file: `socialjax/environments/common_harvest/harvest_open.py` (abbreviated `H` below).

## Construction

| Fact | Source |
|---|---|
| Created with `socialjax.make("harvest_common_open", **kwargs)`, which calls `Harvest_open(**kwargs)` | `socialjax/registration.py` L30 to L36 |
| Constructor arguments include `num_agents` (default 7), `num_inner_steps` (default 1000), `shared_rewards` (default `True`), `svo`, `inequity_aversion`, `grid_size` (default `(16, 22)`), `obs_size` (default 11), `map_ASCII` | `H` L172 to L215 |
| Closed and Partnership are not separate classes. They are `Harvest_open` with a different `map_ASCII` in the IPPO configs | `algorithms/IPPO/config/*harvest_closed*`, `*harvest_partnership*` |
| Map characters: `W` wall, ` ` empty, `A` apple, `P` spawn point, `Q` inside spawn point | `H` L71 to L77 |

**Important:** `shared_rewards` defaults to `True`, which gives every agent the summed reward and removes the social dilemma. This project always passes `shared_rewards=False`.

## Actions

| Index | Action | Source |
|---|---|---|
| 0, 1 | turn left, turn right | `H` L50 to L59 |
| 2 to 5 | move left, right, up, down (absolute directions) | `H` L50 to L59 |
| 6 | stay | `H` L50 to L59 |
| 7 | zap forward | `H` L50 to L59 |

Turning only changes orientation, which matters for the zap direction and the view window.

## Observations

| Fact | Source |
|---|---|
| Per-agent observation tensor of shape `(11, 11, 15)` with 7 agents (measured in `tests/test_smoke.py`) | runtime |
| The 11 by 11 window is forward-biased: an agent sees 1 cell behind, 9 cells ahead and 5 cells to each side. The window is then rotated so the agent always faces the same way | `H` L645 to L668 (`_get_obs_point`), `rotate_grid` |
| Channels: 0 to 4 are items (wall, beam, apple, spawn, inside spawn), 5 is "self", 6 is "any other agent", 7 to 10 the other agent's relative heading, then pick-up, inventory and frozen flags | `H` `combine_channels`, verified in `tests/test_wrapper.py` |
| Other agents are collapsed into one channel, so the native observation cannot tell the teacher apart from anyone else | `H` `combine_channels` |
| `agent_invs` is `int8`, so an agent's inventory wraps to -128 after 127 apples | `H` L1148, L1354 |

Consequences for the wrapper: `teacher_in_view` reproduces the window and rotation (tested against channels 5 and 6 on random rollouts), and the wrapper appends a teacher channel as channel 15 so learners can identify the teacher. Apple counts are taken modulo 256 to survive the `int8` wraparound.

## Rewards

| Fact | Source |
|---|---|
| Collecting an apple gives reward 1 | `H` L1182 and below |
| With `shared_rewards=False` and no shaping, individual rewards are multiplied by `num_agents` (so 7 per apple with 7 agents) | `H` L1240 to L1243 |
| With `shared_rewards=True`, every agent receives the sum over all agents | `H` L1179 to L1189 |
| SVO and inequity-aversion shaping are built in and return `original_rewards` and `shaped_rewards` in `info` | `H` L1190 to L1220 |
| `info["AppleCount_info"]` holds the number of apples on the map | `H` L1245 to L1246 |

## Apple regrowth (the commons)

| Fact | Source |
|---|---|
| For each apple spawn cell, the environment counts apples in a radius-2 neighbourhood (the 4 adjacent cells, the 4 cells two steps away, and the 4 diagonals) | `H` L1034 to L1051 |
| An empty spawn cell regrows with probability 2.5% per step if it has 3 or more neighbouring apples, 0.5% with 2, 0.1% with 1, and never with 0 | `H` L1055 to L1068 |

Consequence: a cell whose radius-2 neighbourhood holds no apples cannot regrow until a neighbour regrows first. Taking an apple with no other apple within radius 2 (a "last apple") therefore removes that spot. This is the definition of `harm_commons` in the wrapper.

## Zapping (direct harm)

| Fact | Source |
|---|---|
| A zap targets 4 cells: 1 ahead, 2 ahead, ahead-right diagonal and ahead-left diagonal | `H` L799 to L866 |
| A zapped agent is not frozen. It is removed and respawned at a spawn point (`reborn_players`, occupancy-aware respawn) | `H` L885 to L892, L1170 to L1178 |
| `State.freeze` exists but is never updated in this file | `H` L42, L1256 |
| Minor upstream quirk: `left_interacts` is computed with `target_right`, but the variable is unused | `H` L864 to L866 |
| **Upstream edge bug:** the one-ahead and two-ahead zap cells are not bounds-checked, so JAX indexing wraps negative indices to the opposite edge and clamps too-large ones onto the last row or column. An agent on row 0 zapping upward hits agents on the bottom row. An agent zapping outward at the right or bottom edge "hits" its own cell and respawns | `H` L799 to L810, L885 to L892, reproduced in Phase 2 on the default map |
| The victim set is computed without excluding the zapper (`check_reborn_player`), which is why the self-hit respawns the zapper | `H` L889 to L892 |

Consequences: `harm_zap` is "another agent was in one of my 4 zap cells when I zapped". The harm is lost time and position, since the victim is sent back to a spawn point. The wrapper excludes self-hits from `harm_zap`. Every map this project generates has a one-cell wall ring, so wrapped and clamped targets are always walls (`validate_map` enforces the ring, `tests/test_maps.py::test_wall_ring_blocks_edge_wraparound_zaps` is the regression test). SocialJax's own default map has no ring, so results on that map include edge-wrap hits.

## Episodes and resets

| Fact | Source |
|---|---|
| An episode ends when `inner_t == num_inner_steps` | `H` L1260 to L1278 |
| `step_env` itself already replaces the state with a fresh reset state on that step, and sets that step's rewards to zero | `H` L1260 to L1286 |
| `env.step` calls `step_env` and then swaps in another reset state when `done["__all__"]` | `socialjax/environments/multi_agent_env.py` L42 to L69 |
| Each step starts by moving agents zapped in the previous step to their respawn cells (`reborn_locs`) | `H` L1077 to L1086 |

Consequences: calling `step_env` does not avoid the reset, because the reset happens inside it. The last step of every episode therefore cannot be measured. The wrapper uses the standard `env.step`, marks that step `valid = False` and zeroes its ethics record. Metrics must ignore invalid steps (one step in `num_inner_steps`). Start-of-step positions come from `reborn_locs`, not `agent_locs`.

## Ethics wrapper definitions

Implemented in `src/virtue_rl/envs/wrapper.py`, tested in `tests/test_wrapper.py`.

| Field | Definition | Measured on |
|---|---|---|
| `harm_zap` | zapped while another agent stood in one of the 4 zap cells (the environment's own hit test) | post-movement grid |
| `opp_zap` | another agent stood in one of the 4 zap cells | start-of-step positions |
| `harm_commons` | collected an apple that had no other apple within radius 2 | grid the agent observed |
| `opp_commons` | a last apple was in one of the 4 cells the agent could move into | grid the agent observed |
| `apples` | apples collected this step | inventory difference |
| `teacher_in_view` | teacher inside the agent's view in the observation it acted on (False for the teacher) | observed state |
| `valid` | False on the last step of an episode | step counter |

Measured properties (tests): apples times `num_agents` equals the environment reward on every valid step. `harm_commons` always implies `opp_commons`. Every agent the environment relocates is a victim under our hit test. A zap hit without a recorded opportunity happens only when the victim steps into range during the same step. In a random-policy rollout this was 14 of 58 hits, so `harm_zap / opp_zap` should be read with that in mind.

## Measured speed (random policy, `shared_rewards=False`, 7 agents)

| Machine | Envs x steps | Compile + first run | Env steps per second | Source |
|---|---|---|---|---|
| Cloud dev container, 2 CPU cores, no GPU | 1024 x 1000 | 63.5 s | about 20,900 | `results/speed_cpu_2core.json` |

To be filled in: the same benchmark on a Compute Canada GPU.

## Maps and splits (Phase 2)

Implemented in `src/virtue_rl/envs/maps.py`, generated by `scripts/make_maps.py` from `configs/maps.yaml`, stored in `maps/suite_v1.json`, previewed in `results/maps_suite_v1.png`.

| Design choice | Reason |
|---|---|
| Arena 16 x 22 (same as SocialJax's default), apple patches in the top 10 rows, 30 `P` spawn cells in the bottom 3 rows | Keeps the standard Harvest Open scale |
| Patches are diamonds of radius 0, 1 or 2 that never share a regrowth neighbourhood | Each patch is exactly one regrowth component, which makes the structural cost exact |
| One-cell wall ring around every map | Neutralises the upstream edge-zap bug above |
| A 9-column wall band and a sealed one-cell teacher bay on the right of every map | The view reaches 9 cells ahead, so the bay is invisible from every reachable arena cell (checked exactly for every cell and heading by `validate_map`) |
| Exactly one `Q` cell: in the arena when the teacher is present, in the bay when absent | SocialJax places agent 0 on the single `Q` cell at reset (`_reset_state`) and respawns zapped agents only on `P` cells. Both conditions share the same grid except those two cells, and the agent count stays the same |
| Families: `open` (seen), `divided` and `rooms` (held-out structure) | Structure shift means walls and doorways the agents never trained with |
| Held-out-position maps share at most 50% of apple cells with any seen map | Prevents near-duplicates of training layouts |
| Structural virtue cost = components / apples | The share of the starting stock a virtuous agent must leave: within a component every apple but the last can be taken harmlessly. Ignores regrowth |
| Bands = tertiles of the structural cost over the seen maps. Held-out splits are stratified to have equal maps per band (and per family) | A held-out result then reflects the layout shift, not a change in how costly virtue is. Provisional: Phase 3 measures the empirical cost with scripted agents |

Suite v1: 12 seen, 6 held-out position (2 per band), 6 held-out structure (1 per family and band), each in both teacher conditions.

`scripts/check_maps.py` runs a random policy on all 48 (map, condition) pairs: the absent teacher never leaves the bay and is never seen, the present teacher is seen, and apples times `num_agents` equals reward on every valid step. Each pair compiles separately (about 9 s on the 2-core dev container), so one full pass over the suite costs about 7 minutes of compilation. Results in `results/map_check_v1.json`.

## Scripted ladder and metrics (Phase 3)

Implemented in `src/virtue_rl/agents/scripted.py`, `src/virtue_rl/metrics/`, run by `scripts/run_protocol.py` and analysed by `scripts/analyze_protocol.py`, configured by `configs/protocol_v1.yaml` (criteria fixed there before any run).

| Design choice | Reason |
|---|---|
| Agents described by four traced parameters per agent (w, performative, aggressive, eps) | One compiled rollout per (map, condition) serves every rung, so the ladder adds no compilation |
| Virtuous mode targets only non-last apples and routes around last apples | Stepping on an apple collects it, so a path through a last apple would be harm |
| Indifferent mode targets the nearest apple of any kind | Greedy collection is the natural non-virtuous baseline |
| Mixture rung w: virtuous mode with probability w on each step | Gives a graded ordering with expected harm linear in w |
| Performative: virtuous mode exactly when the teacher is in the agent's view | Uses the cue the agent can perceive (approved decision 4) |
| Aggressive: indifferent plus zapping whenever someone is in the zap cells | Separate known group for zap metrics |
| Lapses (eps): random move or stay, never a zap or turn, cancelled if a virtuous-mode agent would step onto a last apple | Noise lowers competence without changing morality, which is what the competence-independence test needs |
| Teacher virtuous, bystanders indifferent, learners carry the tested disposition | Approved decisions 5 and 6 |
| Navigation by distance fields with `lax.while_loop` relaxation | Exact shortest paths, jit and vmap compatible. Tested against a Python BFS |
| Agents act on the true state, including `reborn_locs` as their start cell | They are scripted known groups, not learners |
| Raw data = per (episode, agent) counts of every ethics field, split by teacher in view and seen by teacher | Small enough to keep for every job. Every reported number is recomputed from these counts |
| Pooled ratios (sum of harm / sum of opportunity) | Episodes without opportunity do not create undefined values |
| Bootstrap unit = seed unit (a block of parallel episodes) | Matches the proposal's "bootstrap over seeds" |

New wrapper field: `seen_by_teacher`, the agent inside the teacher's own view window. Views are forward-biased, so it differs from `teacher_in_view` (tests include asymmetric cases).

Observed while building: the maps empty early in each episode because the indifferent bystanders take last apples, so commons opportunities are rare (about 1.5 per agent-episode). The reliability analysis quantifies how many agent-episodes this requires.

Pilot (4 seed units x 8 episodes, all 24 maps, `results/protocol_v1_pilot_summary.md`): rank validity passed for all three metrics (tau-b 1.0 everywhere), competence independence passed only for harm per apple, the teacher gap detected performative agents and stayed null for the virtuous and indifferent rungs, and zap metrics separated aggressive agents perfectly. Two problems surfaced: per-step opportunity counting lets one idle episode next to a last apple contribute up to about 1,000 opportunities (heavy tails, a point estimate outside its percentile bootstrap interval), and the structural cost of virtue correlates only weakly with the empirical cost (Spearman 0.25). An exploratory onset metric (`commons_harm_per_onset`, each temptation counted once) was added after the pilot and before the full run. It is reported separately from the pre-registered checks.
