"""Tests for the multi-agent (PettingZoo ParallelEnv) pyramid env."""

import os
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pettingzoo")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig
from water_arena.pyramid_marl import PyramidParallelEnv


def _cfg():
    return PyramidConfig(n_layers=3, n_steps=40, seed=0, allow_bankruptcy=False)


def test_all_traders_are_agents():
    env = PyramidParallelEnv(config=_cfg())
    # 2 + 3 + 4 trader nodes
    assert len(env.possible_agents) == 9


def test_reset_returns_obs_for_every_agent():
    env = PyramidParallelEnv(config=_cfg())
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    for a in env.possible_agents:
        assert obs[a].shape == (9,)


def test_step_returns_five_dicts():
    env = PyramidParallelEnv(config=_cfg())
    env.reset(seed=0)
    actions = {a: [0.2, 0.4] for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    for d in (obs, rewards, terms, truncs, infos):
        assert set(d) == set(env.possible_agents)
    assert all(isinstance(r, float) for r in rewards.values())


def test_episode_truncates_together_at_horizon():
    cfg = _cfg()
    env = PyramidParallelEnv(config=cfg)
    env.reset(seed=0)
    steps = 0
    while env.agents:
        env.step({a: [0.0, 0.3] for a in env.agents})
        steps += 1
    assert steps == cfg.n_steps  # no bankruptcy -> all run to the horizon


def test_determinism_same_seed_same_actions():
    def run():
        env = PyramidParallelEnv(config=_cfg())
        env.reset(seed=0)
        totals = {a: 0.0 for a in env.possible_agents}
        for _ in range(40):
            _, rewards, _, _, _ = env.step({a: [0.1, 0.5] for a in env.agents})
            for a, r in rewards.items():
                totals[a] += r
        return totals

    assert run() == run()


def test_pettingzoo_parallel_api():
    from pettingzoo.test import parallel_api_test

    parallel_api_test(PyramidParallelEnv(config=_cfg()), num_cycles=100)
