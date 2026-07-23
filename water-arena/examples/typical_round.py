"""Plot price, stock, and flow developments during a typical round.

    python examples/typical_round.py

Writes two figures to fig/:
  * round_overview.png -- chain-level price / stock / flow over the round
  * node_detail.png    -- one representative mid-chain node up close
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig, PyramidWorld, build_pyramid
from water_arena.supply_agents import default_supply_roster

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")


def _moving_avg(xs, k=5):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        window = [v for v in xs[lo : i + 1] if v == v]  # skip NaNs
        out.append(sum(window) / len(window) if window else float("nan"))
    return out


def _layer_price(hist, layer_ids):
    """Volume-weighted average sell price of a layer, per step (NaN if idle)."""
    series = []
    for h in hist:
        num = sum(h.sell_price[n] * h.sold[n] for n in layer_ids)
        vol = sum(h.sold[n] for n in layer_ids)
        series.append(num / vol if vol > 1e-9 else float("nan"))
    return series


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("this example needs matplotlib: pip install matplotlib")
        return

    cfg = PyramidConfig(n_layers=3, n_steps=160, seed=7)
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    world = PyramidWorld(default_supply_roster(topo.trader_ids), cfg)
    world.run()
    hist = world.history
    steps = [h.step for h in hist]
    os.makedirs(FIG_DIR, exist_ok=True)

    layer_ids = {L: topo.layers[L] for L in range(1, cfg.n_layers + 1)}
    colors = {1: "tab:blue", 2: "tab:green", 3: "tab:red"}

    # ---- Figure 1: chain-level price / stock / flow ---------------------- #
    fig, (axp, axs, axf) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # prices
    axp.plot(steps, [h.supplier_price for h in hist], color="black", lw=2, label="supplier")
    for L in layer_ids:
        axp.plot(steps, _layer_price(hist, layer_ids[L]), color=colors[L], lw=1.3, label=f"layer {L}")
    axp.plot(
        steps,
        [h.buyer_avg_price if h.buyer_avg_price is not None else float("nan") for h in hist],
        color="tab:purple", lw=1.6, ls="--", label="buyer paid",
    )
    axp.set_ylabel("price")
    axp.set_title("Price — margins stack down the chain, tracking the supplier cycle")
    axp.legend(loc="upper left", fontsize=8, ncol=5)

    # stock (inventory) aggregated per layer
    for L in layer_ids:
        stock = [sum(h.inventory[n] for n in layer_ids[L]) for h in hist]
        axs.plot(steps, stock, color=colors[L], lw=1.3, label=f"layer {L}")
    axs.set_ylabel("stock (inventory)")
    axs.set_title("Stock — base-stock ordering makes inventories saw-tooth")
    axs.legend(loc="upper left", fontsize=8, ncol=3)

    # flow: demand vs delivered, and per-layer throughput (smoothed)
    axf.plot(steps, [h.buyer_need for h in hist], color="grey", lw=1.0, ls=":", label="buyer need")
    axf.plot(steps, _moving_avg([h.buyer_filled for h in hist]), color="tab:purple", lw=1.6, label="delivered to buyer")
    for L in layer_ids:
        flow = _moving_avg([sum(h.sold[n] for n in layer_ids[L]) for h in hist])
        axf.plot(steps, flow, color=colors[L], lw=1.1, alpha=0.8, label=f"layer {L} outflow")
    axf.set_ylabel("flow (units / step)")
    axf.set_xlabel("step")
    axf.set_title("Flow — throughput through each stage (5-step moving average)")
    axf.legend(loc="upper left", fontsize=8, ncol=5)

    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, "round_overview.png")
    fig.savefig(out1, dpi=110)

    # ---- Figure 2: one representative interior node up close ------------- #
    node = topo.layers[2][1]  # an interior layer-2 trader
    name = world.agents[node].name
    fig2, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    a1.plot(steps, [h.supplier_price for h in hist], color="grey", lw=1.0, ls=":", label="supplier price")
    a1.plot(steps, [h.sell_price[node] for h in hist], color="tab:red", lw=1.5, label="its sell price")
    a1.set_ylabel("price")
    a1.set_title(f"Node {topo.label(node)} ({name}) — price")
    a1.legend(loc="upper left", fontsize=8)

    a2.plot(steps, [h.inventory[node] for h in hist], color="tab:green", lw=1.5)
    a2.axhline(cfg.storage_capacity, color="grey", lw=0.8, ls="--")
    a2.set_ylabel("stock")
    a2.set_title("Stock (inventory)")

    a3.plot(steps, [h.bought[node] for h in hist], color="tab:blue", lw=1.2, label="inflow (bought)")
    a3.plot(steps, [h.sold[node] for h in hist], color="tab:orange", lw=1.2, label="outflow (sold)")
    a3.set_ylabel("flow (units / step)")
    a3.set_xlabel("step")
    a3.set_title("Flow — inflow lands one step later than the order (lead time)")
    a3.legend(loc="upper left", fontsize=8)

    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "node_detail.png")
    fig2.savefig(out2, dpi=110)

    print(f"wrote {out1}")
    print(f"wrote {out2}")
    print()
    print(world.price_cascade_str())


if __name__ == "__main__":
    main()
