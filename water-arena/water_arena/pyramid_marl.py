"""Multi-agent RL environment: every trader node learns at once.

A PettingZoo ``ParallelEnv`` in which **each trader node is an agent**. On every
step all agents submit an action simultaneously, the supply chain clears, and
each receives its own observation and its own profit as reward. This is where
the real supply-chain game lives -- undercutting wars for the cheapest-first
buyer, margin discipline, and the bullwhip effect under mutual adaptation.

Agents share the same observation/action layout as the single-agent env
(:mod:`water_arena.pyramid_env`), so a parameter-sharing policy is natural.

PettingZoo and numpy are optional; import fails loudly only if you construct the
env without them.
"""

from __future__ import annotations

import functools
import random
from typing import Dict, Optional

from .pyramid import PyramidConfig, PyramidWorld, SupplyAction, SupplyAgent, SupplyObservation, build_pyramid
from .pyramid_env import ACT_DIM, OBS_DIM, decode_action, observe_node

try:  # optional
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # optional
    from gymnasium import spaces
    from pettingzoo import ParallelEnv

    _HAS_PZ = True
    _Base = ParallelEnv
except ImportError:  # pragma: no cover
    _HAS_PZ = False

    class _Base:  # stand-in so the class body is valid without PettingZoo
        pass


class _MultiSeat(SupplyAgent):
    """One node's seat. Reads the action the env staged for this node and
    decodes it against the true observation the world hands it."""

    def __init__(self, env: "PyramidParallelEnv", nid: int):
        super().__init__(name=f"RL{nid}")
        self.env = env
        self.nid = nid

    def act(self, obs: SupplyObservation) -> SupplyAction:
        return decode_action(obs, self.env._pending[self.nid], self.env.markup_cap)


class PyramidParallelEnv(_Base):
    """PettingZoo ParallelEnv where every pyramid trader is a learning agent."""

    metadata = {"render_modes": ["human"], "name": "water_pyramid_v0", "is_parallelizable": True}

    def __init__(self, config: Optional[PyramidConfig] = None, markup_cap: float = 1.5):
        if not _HAS_NUMPY:
            raise ImportError("PyramidParallelEnv requires numpy; `pip install numpy`")
        if not _HAS_PZ:
            raise ImportError("PyramidParallelEnv requires pettingzoo; `pip install pettingzoo`")
        self._config = config or PyramidConfig()
        self._topo = build_pyramid(self._config.n_layers, self._config.pipeline_capacity)
        self.markup_cap = markup_cap

        self._nids = list(self._topo.trader_ids)
        self._label = {nid: self._topo.label(nid) for nid in self._nids}
        self._nid_of = {self._label[nid]: nid for nid in self._nids}
        self.possible_agents = [self._label[nid] for nid in self._nids]
        self.agents = list(self.possible_agents)

        self._seats = {nid: _MultiSeat(self, nid) for nid in self._nids}
        self._pending: Dict[int, tuple] = {nid: (0.0, 0.0) for nid in self._nids}
        self._seed_rng = random.Random(self._config.seed)
        self._world: Optional[PyramidWorld] = None
        self.render_mode = None

    # -- spaces (identical for every agent) ---------------------------- #
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        return spaces.Box(low=-high, high=high, dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

    # -- PettingZoo parallel API --------------------------------------- #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._config.seed = seed
            self._seed_rng = random.Random(seed)
        else:
            self._config.seed = self._seed_rng.randrange(1, 2**31 - 1)
        roster = dict(self._seats)  # every seat is a learner
        self._world = PyramidWorld(roster, self._config)
        self.agents = list(self.possible_agents)
        obs = {a: observe_node(self._world, self._nid_of[a]) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        assert self._world is not None, "call reset() first"
        for a in self.agents:
            nid = self._nid_of[a]
            act = actions[a]
            self._pending[nid] = (float(act[0]), float(act[1]))

        info = self._world.step()
        acting = list(self.agents)  # agents that were present this step

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        horizon = self._world.done()
        for a in acting:
            nid = self._nid_of[a]
            alive = self._world.states[nid].alive
            observations[a] = observe_node(self._world, nid)
            rewards[a] = float(info.profit.get(nid, 0.0))
            terminations[a] = not alive
            truncations[a] = horizon and alive
            infos[a] = {}

        # drop agents that terminated or truncated
        self.agents = [a for a in acting if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self._world is not None:
            print(self._world.summary())

    def close(self):
        pass

    # -- convenience --------------------------------------------------- #
    @property
    def world(self) -> Optional[PyramidWorld]:
        return self._world

    def node_of(self, agent) -> int:
        return self._nid_of[agent]
