"""Pure JAX geometry helpers that mirror SocialJax Commons Harvest internals.

Every function here reproduces a specific piece of
`socialjax/environments/common_harvest/harvest_open.py` (pinned commit 18e2171)
so that the ethics wrapper measures exactly what the environment does.
Line references are to that file.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from socialjax.environments.common_harvest.harvest_open import STEP, Items

# Radius-2 neighbourhood used by apple regrowth (harvest_open.py L1034-L1051):
# the 4 adjacent cells, the 4 cells two steps away, and the 4 diagonals.
APPLE_NEIGHBOURHOOD = jnp.array(
    [
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-2, 0], [2, 0], [0, -2], [0, 2],
        [-1, -1], [-1, 1], [1, -1], [1, 1],
    ],
    dtype=jnp.int32,
)

# The four cells an agent can reach with one move (moves are absolute
# directions, harvest_open.py STEP_MOVE).
ADJACENT = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def apple_neighbour_counts(grid: jnp.ndarray) -> jnp.ndarray:
    """Number of apples in each cell's radius-2 neighbourhood, shape (H, W).

    Cells outside the map count as non-apples. The upstream count has a few
    inconsistent bounds checks at the map border (L1044-L1047). Those only
    differ from this function for apples on the last row or column, which the
    standard maps do not have.
    """
    is_apple = (grid == Items.apple).astype(jnp.int32)
    padded = jnp.pad(is_apple, 2)
    h, w = grid.shape

    def shifted(offset):
        return jax.lax.dynamic_slice(padded, (2 + offset[0], 2 + offset[1]), (h, w))

    return jnp.sum(jax.vmap(shifted)(APPLE_NEIGHBOURHOOD), axis=0)


def last_apple_mask(grid: jnp.ndarray) -> jnp.ndarray:
    """True where a cell holds an apple with no other apple within radius 2.

    Taking such an apple leaves its cell with zero apple neighbours, and the
    regrowth rule (L1055-L1060) gives a cell with zero neighbours probability
    zero of regrowing.
    """
    return (grid == Items.apple) & (apple_neighbour_counts(grid) == 0)


def lookup(grid: jnp.ndarray, rc: jnp.ndarray, fill: int = 0) -> jnp.ndarray:
    """grid[r, c] with out-of-map cells returning `fill`."""
    h, w = grid.shape
    inside = (rc[..., 0] >= 0) & (rc[..., 0] < h) & (rc[..., 1] >= 0) & (rc[..., 1] < w)
    r = jnp.clip(rc[..., 0], 0, h - 1)
    c = jnp.clip(rc[..., 1], 0, w - 1)
    return jnp.where(inside, grid[r, c], fill)


def zap_target_cells(loc: jnp.ndarray, grid_rows: int, grid_cols: int) -> jnp.ndarray:
    """The 4 cells a zap from `loc` = (row, col, dir) hits, shape (4, 3).

    Reproduces L799-L840 exactly: one ahead, two ahead, ahead-right and
    ahead-left, where a diagonal that falls outside the map is replaced by the
    one-ahead cell. Rows keep the (row, col, dir) layout upstream uses so that
    the out-of-bounds test is identical.
    """
    d = loc[2].astype(jnp.int32)
    loc = loc.astype(jnp.int32)
    step = STEP.astype(jnp.int32)
    one = loc + step[d]
    two = loc + 2 * step[d]
    right = loc + step[d] + step[(d + 1) % 4]
    left = loc + step[d] + step[(d - 1) % 4]

    def oob(t):
        return (t[0] > grid_rows - 1) | (t[1] > grid_cols - 1) | jnp.any(t < 0)

    right = jnp.where(oob(right), one, right)
    left = jnp.where(oob(left), one, left)
    return jnp.stack([one, two, right, left])


def upstream_grid_lookup(grid: jnp.ndarray, cells: jnp.ndarray) -> jnp.ndarray:
    """Index `grid` at `cells[..., :2]` with upstream's plain jnp indexing.

    Upstream reads zap targets with `state.grid[t[0], t[1]]` and no bounds
    check for the one-ahead and two-ahead cells, so JAX's default indexing
    applies (negative indices wrap, too-large indices clamp). Using the same
    indexing keeps `harm_zap` identical to the environment's own hit test,
    including that edge behaviour.
    """
    return grid[cells[..., 0], cells[..., 1]]


def obs_window_corner(loc: jnp.ndarray, obs_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-left corner of an agent's view in the padded grid (L645-L668)."""
    padding = obs_size - 1
    half = obs_size // 2
    x, y, direction = loc[0], loc[1], loc[2]
    x = x + padding - half
    y = y + padding - half
    x = jnp.where(direction == 0, x + half - 1, x)
    y = jnp.where(direction == 1, y + half - 1, y)
    x = jnp.where(direction == 2, x - half + 1, x)
    y = jnp.where(direction == 3, y - half + 1, y)
    return x, y


def agent_views(grid: jnp.ndarray, agent_locs: jnp.ndarray, obs_size: int) -> jnp.ndarray:
    """Raw grid values inside every agent's egocentric view, shape (N, O, O).

    Same padding (walls), window placement and rotation as upstream
    `_get_obs` (L669 onward), before the one-hot encoding.
    """
    padding = obs_size - 1
    padded = jnp.pad(grid, ((padding, padding), (padding, padding)),
                     constant_values=Items.wall)

    def one_view(loc):
        x, y = obs_window_corner(loc, obs_size)
        view = jax.lax.dynamic_slice(padded, (x, y), (obs_size, obs_size))
        view = jnp.where(loc[2] == 1, jnp.rot90(view, k=1, axes=(0, 1)), view)
        view = jnp.where(loc[2] == 2, jnp.rot90(view, k=2, axes=(0, 1)), view)
        view = jnp.where(loc[2] == 3, jnp.rot90(view, k=3, axes=(0, 1)), view)
        return view

    return jax.vmap(one_view)(agent_locs)
