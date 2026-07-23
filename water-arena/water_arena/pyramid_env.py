"""Single-agent RL environment for the pyramid supply chain.

You control **one trader node**; every other node runs a scripted strategy. The
env follows the Gymnasium API (``reset`` / ``step`` -> ``(obs, reward,
terminated, truncated, info)``) so Stable-Baselines3, RLlib, CleanRL, etc. plug
in directly. Gymnasium and numpy are optional; without Gymnasium the env still
runs with the same signatures and lightweight fallback spaces.

Action (2-vector in [-1, 1]):
    ``action[0]`` -- price: markup over input cost. sell price =
                     cheapest_parent_price * (1 + markup), markup in
                     [0, ``markup_cap``] as a0 sweeps [-1, 1].
    ``action[1]`` -- replenishment: order up to a target inventory of
                     (a1+1)/2 * storage_capacity, sourced cheapest-parent-first
                     within pipeline / cash limits.

Observation (9-vector), lightly normalised:
    [inventory/cap, cash/initial_cash, input_cost, last_sell_price,
     last_sold/buyer_need, supplier_price, buyer_WTP-or-0, is_bottom,
     fraction_of_episode]

Reward: the controlled node's profit this step (its change in cash).
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

from .pyramid import PyramidConfig, PyramidWorld, SupplyAction, SupplyAgent, SupplyObservation, build_pyramid
from .supply_agents import _order_up_to, default_supply_roster

try:  # optional
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # optional
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
    _Base = gym.Env
except ImportError:  # pragma: no cover
    _HAS_GYM = False

    class _Base:  # minimal stand-in so the class body is always valid
        pass


OBS_DIM = 9
ACT_DIM = 2


# --------------------------------------------------------------------------- #
# shared observe / decode helpers (used by the single-agent and MARL envs)
# --------------------------------------------------------------------------- #
def supplier_price_now(world) -> float:
    """The supplier's price for the world's current step, without side effects."""
    cfg = world.config
    if world._custom_price_fn is not None:
        return max(0.0, world._custom_price_fn(world.step_count))
    mode = cfg.supplier_price_mode
    if mode == "constant":
        return cfg.supplier_price
    if mode == "sine":
        phase = 2 * math.pi * world.step_count / max(1.0, cfg.supplier_price_period)
        return max(0.05, cfg.supplier_price * (1.0 + cfg.supplier_price_amp * math.sin(phase)))
    return world._sp  # random_walk: current level


def decode_action(obs: SupplyObservation, raw, markup_cap: float) -> SupplyAction:
    """Map a raw ``[markup_knob, target_stock_knob]`` in [-1,1] to a SupplyAction."""
    a0, a1 = raw
    cost = obs.cheapest_parent_price
    markup = 0.5 * (a0 + 1.0) * markup_cap  # a0 in [-1,1] -> [0, cap]
    price = max(1e-3, cost * (1.0 + markup))
    target = 0.5 * (a1 + 1.0) * obs.storage_capacity  # a1 in [-1,1] -> [0, cap]
    return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


def observe_node(world, nid: int):
    """Build the 9-dim observation vector for trader node ``nid``."""
    cfg = world.config
    topo = world.topology
    s = world.states[nid]
    supplier_price = supplier_price_now(world)
    parent_prices = [
        supplier_price if p == topo.supplier_id else world.states[p].sell_price
        for p in topo.parents[nid]
    ]
    input_cost = min(parent_prices) if parent_prices else supplier_price
    is_bottom = 1.0 if nid in topo.bottom_ids else 0.0
    wtp = cfg.buyer_willingness_to_pay if nid in topo.bottom_ids else 0.0
    frac = world.step_count / max(1, cfg.n_steps)
    vec = [
        s.inventory / cfg.storage_capacity,
        s.cash / cfg.initial_cash,
        input_cost,
        s.sell_price,
        s.last_sold_v / max(1e-9, cfg.buyer_need),
        supplier_price,
        wtp,
        is_bottom,
        frac,
    ]
    return np.asarray(vec, dtype=np.float32)


class _LearnerSeat(SupplyAgent):
    """Occupies the controlled node. Its action is injected by the env; it
    decodes the raw policy output against the *true* current observation the
    world hands it, so ordering respects the real prices and limits."""

    def __init__(self, env: "PyramidTradingEnv"):
        super().__init__(name="RL")
        self.env = env
        self.pending_raw = (0.0, 0.0)

    def act(self, obs: SupplyObservation) -> SupplyAction:
        return self.env._decode(obs, self.pending_raw)


class PyramidTradingEnv(_Base):
    """Gymnasium-style env: you control one pyramid trader, bots fill the rest."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[PyramidConfig] = None,
        learner_node: Optional[int] = None,
        opponents: Optional[Dict[int, SupplyAgent]] = None,
        markup_cap: float = 1.5,
    ):
        super().__init__()
        if not _HAS_NUMPY:
            raise ImportError("PyramidTradingEnv requires numpy; `pip install numpy`")
        self._config = config or PyramidConfig()
        self._topo = build_pyramid(self._config.n_layers, self._config.pipeline_capacity)
        # default: an interior bottom node -- two parents, faces the buyer directly
        self.learner_node = learner_node if learner_node is not None else self._topo.bottom_ids[1]
        if self.learner_node not in self._topo.trader_ids:
            raise ValueError(f"learner_node {self.learner_node} is not a trader node")
        self._opponents = opponents
        self.markup_cap = markup_cap
        self._seat = _LearnerSeat(self)
        self._world: Optional[PyramidWorld] = None
        self._seed_rng = random.Random(self._config.seed)

        if _HAS_GYM:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
            high = np.full(OBS_DIM, np.inf, dtype=np.float32)
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        else:  # simple fallbacks
            self.action_space = _Box(-1.0, 1.0, (ACT_DIM,))
            self.observation_space = _Box(-float("inf"), float("inf"), (OBS_DIM,))

    # -- Gym API ------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            # explicit seed -> exact, reproducible episode (tests, evaluation)
            self._config.seed = seed
            self._seed_rng = random.Random(seed)
        else:
            # auto-reset -> a fresh world each episode so training sees variety
            # (reproducible: the stream is anchored by the last explicit seed)
            self._config.seed = self._seed_rng.randrange(1, 2**31 - 1)
        roster: Dict[int, SupplyAgent] = (
            dict(self._opponents) if self._opponents is not None
            else default_supply_roster(self._topo.trader_ids)
        )
        roster[self.learner_node] = self._seat  # take our seat
        self._world = PyramidWorld(roster, self._config)
        return self._observe(), {}

    def step(self, action):
        assert self._world is not None, "call reset() first"
        self._seat.pending_raw = (float(action[0]), float(action[1]))
        info = self._world.step()
        reward = float(info.profit.get(self.learner_node, 0.0))
        state = self._world.states[self.learner_node]
        terminated = not state.alive
        truncated = self._world.done() and state.alive
        return self._observe(), reward, terminated, truncated, {"step_info": info}

    def render(self):  # pragma: no cover - convenience only
        if self._world is not None:
            print(self._world.summary())

    # -- action decoding & observation (delegate to shared helpers) ---- #
    def _decode(self, obs: SupplyObservation, raw) -> SupplyAction:
        return decode_action(obs, raw, self.markup_cap)

    def _observe(self):
        assert self._world is not None
        return observe_node(self._world, self.learner_node)


class _Box:
    """Very small Box stand-in used when Gymnasium is not installed."""

    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):
        import random

        return [random.uniform(-1.0, 1.0) for _ in range(self.shape[0])]

    def __repr__(self) -> str:
        return f"_Box(low={self.low}, high={self.high}, shape={self.shape})"
