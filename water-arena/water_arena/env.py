"""Single-agent RL environment wrapping the multi-agent :class:`World`.

One *learning* agent trades against a roster of scripted opponents. The env
follows the Gymnasium API (``reset`` / ``step`` returning
``(obs, reward, terminated, truncated, info)``) so RL libraries such as
Stable-Baselines3 or RLlib work out of the box.

Gymnasium and numpy are optional. If Gymnasium is installed the env subclasses
``gymnasium.Env`` and exposes real ``spaces``; otherwise it still runs with the
same method signatures and lightweight fallback spaces.

Action (2-vector, both in roughly [-1, 1]):
    ``action[0]`` -- price offset: clearing/reference price * (1 + 0.5 * a0)
    ``action[1]`` -- signed size: fraction of storage headroom (buy, a1>0) or
                     of held water (sell, a1<0)

Observation (numpy vector), all lightly normalised:
    [cash, water, storage_headroom, demand, last_price,
     price_momentum, fraction_of_episode_elapsed]
"""

from __future__ import annotations

from typing import List, Optional

from .agents import Action, Agent, Observation
from .world import World, WorldConfig

try:  # optional dependency
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # optional dependency
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
    _Base = gym.Env
except ImportError:  # pragma: no cover
    _HAS_GYM = False

    class _Base:  # minimal stand-in so the class body is always valid
        pass


OBS_DIM = 7
ACT_DIM = 2


class _ExternalAgent(Agent):
    """Placeholder occupying the learner's seat in the World.

    Its action is injected each step by the environment rather than computed
    from the observation, so we stash the latest observation for the env to read.
    """

    def __init__(self, name: str = "RL"):
        super().__init__(name)
        self.pending: Action = Action.hold()
        self.last_obs: Optional[Observation] = None

    def act(self, obs: Observation) -> Action:
        self.last_obs = obs
        return self.pending


class WaterTradingEnv(_Base):
    """Gymnasium-style env: you control agent 0, scripted bots fill the rest."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponents: Optional[List[Agent]] = None,
        config: Optional[WorldConfig] = None,
    ):
        super().__init__()
        if not _HAS_NUMPY:
            raise ImportError("WaterTradingEnv requires numpy; `pip install numpy`")
        if opponents is None:
            # default field of scripted opponents
            from .agents import Hoarder, MeanReversionTrader, NeedsTrader, RandomAgent

            opponents = [
                NeedsTrader(name="Needs"),
                MeanReversionTrader(name="MeanRev"),
                Hoarder(name="Hoarder"),
                RandomAgent(name="Random", seed=1),
            ]
        self._learner = _ExternalAgent()
        self._opponents = opponents
        self._config = config or WorldConfig()
        self._world: Optional[World] = None

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
            self._config.seed = seed
        agents: List[Agent] = [self._learner, *self._opponents]
        self._world = World(agents, self._config)
        # prime an observation without advancing time
        obs = self._peek_observation()
        return obs, {}

    def step(self, action):
        assert self._world is not None, "call reset() first"
        self._learner.pending = self._decode_action(action)
        info = self._world.step()
        reward = float(info.rewards.get(0, 0.0))
        obs = self._peek_observation()
        state = self._world.states[0]
        terminated = not state.alive
        truncated = self._world.done() and state.alive
        return obs, reward, terminated, truncated, {"step_info": info}

    def render(self):  # pragma: no cover - convenience only
        if self._world is not None:
            print(self._world.summary())

    # -- helpers ------------------------------------------------------- #
    def _decode_action(self, action) -> Action:
        a0 = float(action[0])
        a1 = float(action[1])
        obs = self._learner.last_obs
        ref = obs.reference_price if obs is not None else self._config.initial_price
        price = max(1e-3, ref * (1.0 + 0.5 * a0))
        if obs is None:
            return Action(price=price, quantity=0.0)
        if a1 >= 0:
            qty = a1 * obs.storage_headroom
        else:
            qty = a1 * obs.water  # a1 < 0 -> negative -> sell
        return Action(price=price, quantity=qty)

    def _peek_observation(self):
        """Build the learner's observation vector from current world state."""
        world = self._world
        assert world is not None
        s = world.states[0]
        hist = world.price_history
        last = hist[-1] if hist else self._config.initial_price
        if len(hist) >= 2:
            momentum = (hist[-1] - hist[-2]) / (abs(hist[-2]) + 1e-9)
        else:
            momentum = 0.0
        frac = world.step_count / max(1, self._config.n_steps)
        headroom = max(0.0, s.storage_capacity - s.water)
        demand = self._config.base_demand
        vec = [s.cash, s.water, headroom, demand, last, momentum, frac]
        return np.asarray(vec, dtype=np.float32)


class _Box:
    """Very small Box stand-in used when Gymnasium is not installed."""

    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):  # uniform-ish sample, numpy required
        import random

        return [random.uniform(-1.0, 1.0) for _ in range(self.shape[0])]

    def __repr__(self) -> str:
        return f"_Box(low={self.low}, high={self.high}, shape={self.shape})"
