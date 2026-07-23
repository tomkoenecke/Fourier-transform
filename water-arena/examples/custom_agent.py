"""Minimal example of writing your own strategy and dropping it into the arena.

    python examples/custom_agent.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena import (
    Action,
    Agent,
    Hoarder,
    MeanReversionTrader,
    NeedsTrader,
    World,
    WorldConfig,
)


class UndercutSeller(Agent):
    """Covers its own thirst, then sells any surplus by slightly undercutting
    the last clearing price so its offers keep getting hit."""

    def act(self, obs):
        reserve = obs.demand * 2  # keep two steps of water in the tank
        surplus = obs.water - reserve
        if surplus > 0 and obs.last_price is not None:
            return Action(price=obs.last_price * 0.98, quantity=-surplus)
        if obs.water < obs.demand:  # top up if we're short
            return Action(price=obs.reference_price * 1.1, quantity=obs.demand)
        return Action.hold()


def main() -> None:
    roster = [
        UndercutSeller(name="Undercut"),
        NeedsTrader(name="Needs"),
        Hoarder(name="Hoarder"),
        MeanReversionTrader(name="MeanRev"),
    ]
    world = World(roster, WorldConfig(n_steps=250, seed=3))
    world.run()
    print(world.summary())


if __name__ == "__main__":
    main()
