"""Run the pyramid supply chain and print the leaderboard.

    python examples/pyramid_run.py

With matplotlib installed it also writes fig/pyramid.png (supplier price, buyer
fill rate, and per-node inventory).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig, PyramidWorld, build_pyramid
from water_arena.supply_agents import default_supply_roster


def main() -> None:
    config = PyramidConfig(n_layers=3, n_steps=300, seed=7)
    topo = build_pyramid(config.n_layers, config.pipeline_capacity)
    world = PyramidWorld(default_supply_roster(topo.trader_ids), config)

    print(_describe_topology(world))
    print()
    world.run(verbose=False)
    print(world.summary())
    print()
    print(world.price_cascade_str())
    _maybe_plot(world)


def _describe_topology(world: PyramidWorld) -> str:
    topo = world.topology
    lines = ["Topology (node -> children):"]
    for L, layer in enumerate(topo.layers):
        names = ", ".join(topo.label(n) for n in layer)
        lines.append(f"  layer {L}: {names}")
    return "\n".join(lines)


def _maybe_plot(world: PyramidWorld) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to also get a chart)")
        return

    steps = [h.step for h in world.history]
    supply = [h.supplier_price for h in world.history]
    buyer_price = [h.buyer_avg_price if h.buyer_avg_price is not None else float("nan") for h in world.history]
    fill = [
        100.0 * h.buyer_filled / h.buyer_need if h.buyer_need > 0 else 0.0
        for h in world.history
    ]

    fig, (ax_p, ax_f, ax_i) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax_p.plot(steps, supply, label="supplier price", color="tab:blue")
    ax_p.plot(steps, buyer_price, label="buyer avg price", color="tab:red")
    ax_p.set_ylabel("price")
    ax_p.legend(loc="upper left", fontsize=8)
    ax_p.set_title("Pyramid Arena -- prices propagate down the chain")

    ax_f.plot(steps, fill, color="tab:green")
    ax_f.set_ylabel("buyer fill %")
    ax_f.set_ylim(0, 105)

    topo = world.topology
    for nid in topo.trader_ids:
        inv = [h.inventory.get(nid, 0.0) for h in world.history]
        ax_i.plot(steps, inv, label=topo.label(nid), alpha=0.7)
    ax_i.set_ylabel("inventory")
    ax_i.set_xlabel("step")
    ax_i.legend(loc="upper left", fontsize=7, ncol=3)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pyramid.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nchart written to {out}")


if __name__ == "__main__":
    main()
