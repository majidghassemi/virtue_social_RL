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
| The 11 by 11 window is forward-biased: its corner is shifted by `obs_size // 2 - 1` cells in the facing direction, so an agent sees farther ahead than behind | `H` L645 to L668 (`_get_obs_point`) |

Consequence for the wrapper: `teacher_in_view` must reproduce this window logic and be tested against the teacher's channel in the observation tensor.

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

Consequence: taking an apple that leaves an area with no apples within radius 2 removes that area permanently. This is the definition of `harm_commons` for the wrapper.

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
| `env.step` calls `step_env` and then silently swaps in a fresh reset state when `done["__all__"]` | `socialjax/environments/multi_agent_env.py` L42 to L69 |

Consequence: the ethics wrapper must call `step_env` directly and handle the reset itself. Otherwise the last transition of each episode is read from the reset state.

## Measured speed (random policy, `shared_rewards=False`, 7 agents)

| Machine | Envs x steps | Compile + first run | Env steps per second | Source |
|---|---|---|---|---|
| Cloud dev container, 2 CPU cores, no GPU | 1024 x 1000 | 63.5 s | about 20,900 | `results/speed_cpu_2core.json` |

To be filled in: the same benchmark on a Compute Canada GPU.
