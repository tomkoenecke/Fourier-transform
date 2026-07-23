"""Run a scripted-agent tournament and print the leaderboard.

    python examples/tournament.py

If matplotlib is installed, also writes a price + net-worth chart to
``fig/tournament.png``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena import World, WorldConfig, default_roster


def main() -> None:
    config = WorldConfig(n_steps=300, seed=7)
    world = World(default_roster(), config)
    world.run(verbose=False)
    print(world.summary())
    _maybe_plot(world)


def _maybe_plot(world: World) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to also get a chart)")
        return

    steps = [h.step for h in world.history]
    prices = [h.reference_price for h in world.history]

    fig, (ax_price, ax_wealth) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax_price.plot(steps, prices, color="tab:blue")
    for h in world.history:
        if h.drought:
            ax_price.axvline(h.step, color="tab:orange", alpha=0.15)
    ax_price.set_ylabel("clearing price")
    ax_price.set_title("Water Arena -- price (orange bands = droughts)")

    for i, s in enumerate(world.states):
        wealth = [h.net_worth.get(i, 0.0) for h in world.history]
        ax_wealth.plot(steps, wealth, label=s.name)
    ax_wealth.set_ylabel("net worth")
    ax_wealth.set_xlabel("step")
    ax_wealth.legend(loc="upper left", fontsize=8, ncol=2)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "tournament.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nchart written to {out}")


if __name__ == "__main__":
    main()
