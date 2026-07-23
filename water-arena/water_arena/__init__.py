"""Water Arena -- a small economy simulation for RL and scripted strategies.

Agents own wells that produce water, must consume water each step to survive,
and trade surpluses/deficits in a uniform-price call auction. The package is a
compact arena for pitting reinforcement-learning agents against scripted
heuristics.

Quick start::

    from water_arena import World, WorldConfig, default_roster

    world = World(default_roster(), WorldConfig(n_steps=200, seed=0))
    world.run()
    print(world.summary())
"""

from .agents import (
    Action,
    Agent,
    DoNothingAgent,
    Hoarder,
    MeanReversionTrader,
    NeedsTrader,
    Observation,
    RandomAgent,
    TrendFollower,
    default_roster,
)
from .market import MarketResult, Order, Trade, clear_market
from .pyramid import (
    NodeState,
    PyramidConfig,
    PyramidStep,
    PyramidWorld,
    SupplyAction,
    SupplyAgent,
    SupplyObservation,
    Topology,
    build_pyramid,
)
from .supply_agents import (
    CostPlusTrader,
    Discounter,
    Monopolist,
    RandomTrader,
    Speculator,
    default_supply_roster,
)
from .world import AgentState, StepInfo, World, WorldConfig

__version__ = "0.2.0"

__all__ = [
    # flat market arena
    "Action",
    "Agent",
    "AgentState",
    "DoNothingAgent",
    "Hoarder",
    "MarketResult",
    "MeanReversionTrader",
    "NeedsTrader",
    "Observation",
    "Order",
    "RandomAgent",
    "StepInfo",
    "Trade",
    "TrendFollower",
    "World",
    "WorldConfig",
    "clear_market",
    "default_roster",
    # pyramid supply-chain arena
    "NodeState",
    "PyramidConfig",
    "PyramidStep",
    "PyramidWorld",
    "SupplyAction",
    "SupplyAgent",
    "SupplyObservation",
    "Topology",
    "build_pyramid",
    "CostPlusTrader",
    "Discounter",
    "Monopolist",
    "RandomTrader",
    "Speculator",
    "default_supply_roster",
]
