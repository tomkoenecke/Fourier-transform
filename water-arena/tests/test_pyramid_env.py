"""Tests for the pyramid single-agent RL environment."""

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig
from water_arena.pyramid_env import ACT_DIM, OBS_DIM, PyramidTradingEnv


def _cfg():
    return PyramidConfig(n_layers=3, n_steps=60, seed=0)


def test_reset_returns_obs_of_right_shape():
    env = PyramidTradingEnv(config=_cfg())
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(info, dict)


def test_step_contract():
    env = PyramidTradingEnv(config=_cfg())
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([0.2, 0.5])
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "step_info" in info


def test_episode_truncates_at_horizon():
    cfg = _cfg()
    env = PyramidTradingEnv(config=cfg)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step([0.0, 0.3])
        steps += 1
        if terminated or truncated:
            break
    assert steps <= cfg.n_steps
    assert truncated or terminated


def test_action_space_shape():
    env = PyramidTradingEnv(config=_cfg())
    assert env.action_space.shape == (ACT_DIM,)


def test_determinism_same_seed_same_actions():
    actions = [[0.1 * (i % 5) - 0.2, 0.3] for i in range(60)]

    def run():
        env = PyramidTradingEnv(config=_cfg())
        env.reset(seed=0)
        rs = []
        for a in actions:
            _, r, term, trunc, _ = env.step(a)
            rs.append(r)
            if term or trunc:
                break
        return rs

    assert run() == run()


def test_can_control_a_chosen_node():
    cfg = _cfg()
    env = PyramidTradingEnv(config=cfg, learner_node=1)  # a layer-1 node
    obs, _ = env.reset(seed=0)
    assert env.learner_node == 1
    # layer-1 node is not a bottom node -> WTP feature is zero
    assert obs[6] == 0.0
