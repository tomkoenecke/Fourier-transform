"""Tests for the World simulation: invariants and determinism."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.agents import Action, Agent, NeedsTrader, RandomAgent
from water_arena.world import World, WorldConfig


class FixedAgent(Agent):
    """Always submits the same order -- handy for deterministic checks."""

    def __init__(self, price, quantity, name=None):
        super().__init__(name)
        self._action = Action(price, quantity)

    def act(self, obs):
        return self._action


def test_determinism_with_seed():
    cfg = WorldConfig(n_steps=50, seed=42)
    a = World([NeedsTrader(), RandomAgent(seed=0)], cfg)
    a.run()
    b = World([NeedsTrader(), RandomAgent(seed=0)], WorldConfig(n_steps=50, seed=42))
    b.run()
    assert [s.net_worth(a.reference_price) for s in a.states] == [
        s.net_worth(b.reference_price) for s in b.states
    ]


def test_cash_water_conservation_in_trade():
    # A pure buyer and a pure seller; no production or consumption so the only
    # balance change comes from trading -> cash and water are conserved.
    cfg = WorldConfig(
        n_steps=1,
        well_capacity=0.0,
        production_noise=0.0,
        drought_prob=0.0,
        base_demand=0.0,
        demand_noise=0.0,
        initial_cash=100.0,
        initial_water=10.0,
        seed=1,
    )
    buyer = FixedAgent(price=2.0, quantity=5, name="buyer")
    seller = FixedAgent(price=1.0, quantity=-5, name="seller")
    world = World([buyer, seller], cfg)
    world.step()
    total_cash = sum(s.cash for s in world.states)
    total_water = sum(s.water for s in world.states)
    assert abs(total_cash - 200.0) < 1e-9  # cash only moves between agents
    assert abs(total_water - 20.0) < 1e-9  # water only moves between agents


def test_storage_capacity_respected():
    cfg = WorldConfig(
        n_steps=20,
        storage_capacity=15.0,
        well_capacity=10.0,
        production_noise=0.0,
        drought_prob=0.0,
        base_demand=0.0,
        seed=3,
    )
    world = World([NeedsTrader(), RandomAgent(seed=2)], cfg)
    world.run()
    for s in world.states:
        assert s.water <= cfg.storage_capacity + 1e-9


def test_cannot_sell_more_than_held():
    cfg = WorldConfig(n_steps=1, well_capacity=0.0, base_demand=0.0, seed=0)
    seller = FixedAgent(price=0.5, quantity=-1000, name="seller")  # oversell attempt
    buyer = FixedAgent(price=1.0, quantity=1000, name="buyer")
    world = World([seller, buyer], cfg)
    world.step()
    for s in world.states:
        assert s.water >= -1e-9  # never negative water


def test_bankruptcy_removes_agent():
    # An agent that always fails to consume gets penalised into bankruptcy.
    cfg = WorldConfig(
        n_steps=100,
        initial_cash=5.0,
        initial_water=0.0,
        well_capacity=0.0,
        base_demand=4.0,
        demand_noise=0.0,
        unmet_penalty=5.0,
        seed=0,
    )
    from water_arena.agents import DoNothingAgent

    world = World([DoNothingAgent(name="doomed")], cfg)
    world.run()
    assert not world.states[0].alive
    assert world.states[0].died_at is not None
