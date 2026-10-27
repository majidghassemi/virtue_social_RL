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

Consequence: `harm_zap` is "another agent was in one of my 4 zap cells when I zapped". The harm is lost time and position, since the victim is sent back to a spawn point.

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
