"""Pit the advanced strategies head-to-head on the pyramid.

    python examples/advanced_showdown.py

Every node runs an advanced strategy (EWMA, Adaptive, InventoryAware, JIT,
Bandit), cycled across the pyramid, so you can see which approach wins where.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig, PyramidWorld, build_pyramid
from water_arena.supply_agents import advanced_supply_roster


def main() -> None:
    config = PyramidConfig(n_layers=3, n_steps=400, seed=3)
    topo = build_pyramid(config.n_layers, config.pipeline_capacity)
    world = PyramidWorld(advanced_supply_roster(topo.trader_ids), config)
    world.run()
    print(world.summary())


if __name__ == "__main__":
    main()
