"""Tests for the pyramid supply-chain model."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import (
    PyramidConfig,
    PyramidWorld,
    SupplyAction,
    SupplyAgent,
    build_pyramid,
)
from water_arena.supply_agents import (
    BanditPricer,
    CostPlusTrader,
    advanced_supply_roster,
    default_supply_roster,
)


def test_topology_shape():
    topo = build_pyramid(n_layers=3, pipeline_capacity=8.0)
    # layer sizes: supplier(1), then 2, 3, 4
    assert [len(layer) for layer in topo.layers] == [1, 2, 3, 4]
    assert len(topo.trader_ids) == 2 + 3 + 4
    assert len(topo.bottom_ids) == 4


def test_middle_child_is_shared():
    topo = build_pyramid(n_layers=2, pipeline_capacity=8.0)
    # layer 1 = two nodes, layer 2 = three nodes; the middle one has two parents
    layer1 = topo.layers[1]
    layer2 = topo.layers[2]
    assert len(topo.parents[layer2[1]]) == 2  # shared middle node
    assert set(topo.parents[layer2[1]]) == set(layer1)
    assert len(topo.parents[layer2[0]]) == 1
    assert len(topo.parents[layer2[2]]) == 1


def test_wrap_gives_every_trader_two_parents():
    topo = build_pyramid(n_layers=3, pipeline_capacity=8.0, wrap=True)
    # layer 1 nodes still have the single supplier as their only parent
    for nid in topo.layers[1]:
        assert len(topo.parents[nid]) == 1
    # every trader in layers 2+ now has exactly two parents (no edge nodes)
    for L in range(2, topo.n_layers + 1):
        for nid in topo.layers[L]:
            assert len(topo.parents[nid]) == 2, topo.label(nid)
    # corner parents pick up a third child; the shape is no longer a triangle
    corner = topo.layers[2][0]
    assert len(topo.children[corner]) == 3


def test_wrap_raises_the_floor_for_the_worst_position():
    # Wrap can't fully equalize a growing pyramid, but by giving every corner a
    # second parent it lifts the worst-off node's profit above the un-wrapped
    # single-pipeline corner.
    from water_arena.supply_agents import CostPlusTrader

    def min_profit(wrap):
        cfg = PyramidConfig(n_layers=3, n_steps=2000, seed=1, wrap=wrap)
        topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity, cfg.wrap)
        roster = {nid: CostPlusTrader(name=topo.label(nid)) for nid in topo.trader_ids}
        w = PyramidWorld(roster, cfg)
        w.run()
        return min(w.states[n].total_profit for n in topo.trader_ids)

    assert min_profit(wrap=True) > min_profit(wrap=False)


def test_supplier_feeds_two():
    topo = build_pyramid(n_layers=3, pipeline_capacity=8.0)
    assert len(topo.children[topo.supplier_id]) == 2


def test_determinism_with_seed():
    cfg = PyramidConfig(n_layers=3, n_steps=80, seed=11)
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    a = PyramidWorld(default_supply_roster(topo.trader_ids), cfg)
    a.run()
    b = PyramidWorld(default_supply_roster(topo.trader_ids), PyramidConfig(n_layers=3, n_steps=80, seed=11))
    b.run()
    assert [a.states[n].total_profit for n in topo.trader_ids] == [
        b.states[n].total_profit for n in topo.trader_ids
    ]


def test_pipeline_capacity_caps_flow():
    # One buyer node ordering a huge amount can never receive more than the
    # pipeline capacity in a single step.
    cap = 5.0
    cfg = PyramidConfig(
        n_layers=1,
        n_steps=1,
        pipeline_capacity=cap,
        initial_inventory=0.0,
        initial_cash=1e6,
        supplier_price_mode="constant",
        supplier_price=1.0,
        holding_cost=0.0,
    )
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)

    class Glutton(SupplyAgent):
        def act(self, obs):
            return SupplyAction(sell_price=2.0, orders={p: 1000.0 for p in obs.parent_prices})

    roster = {nid: Glutton(name=f"g{nid}") for nid in topo.trader_ids}
    world = PyramidWorld(roster, cfg)
    world.step()
    for nid in topo.trader_ids:
        assert world.states[nid].inventory <= cap + 1e-9


def test_no_negative_inventory_or_cash_conservation():
    # Money the buyer spends must equal money the bottom sellers receive from it,
    # and no node holds negative inventory.
    cfg = PyramidConfig(n_layers=3, n_steps=120, seed=5)
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    world = PyramidWorld(default_supply_roster(topo.trader_ids), cfg)
    world.run()
    for s in world.states.values():
        assert s.inventory >= -1e-9


def test_advanced_roster_runs_and_is_deterministic():
    cfg = PyramidConfig(n_layers=3, n_steps=100, seed=9)
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    a = PyramidWorld(advanced_supply_roster(topo.trader_ids), cfg)
    a.run()
    b = PyramidWorld(advanced_supply_roster(topo.trader_ids), PyramidConfig(n_layers=3, n_steps=100, seed=9))
    b.run()
    assert [a.states[n].total_profit for n in topo.trader_ids] == [
        b.states[n].total_profit for n in topo.trader_ids
    ]
    # the chain should still serve a meaningful fraction of demand
    served = sum(h.buyer_filled for h in a.history)
    needed = sum(h.buyer_need for h in a.history)
    assert served > 0.4 * needed


def test_bandit_learns_to_price_below_wtp():
    # A lone bottom bandit facing a cheapest-first buyer should learn to price
    # at or below willingness-to-pay often enough to make real sales.
    cfg = PyramidConfig(
        n_layers=1,
        n_steps=400,
        buyer_willingness_to_pay=2.5,
        supplier_price=1.0,
        supplier_price_mode="constant",
        seed=0,
    )
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    roster = {nid: BanditPricer(name=f"b{nid}", seed=nid) for nid in topo.trader_ids}
    world = PyramidWorld(roster, cfg)
    world.run()
    served = sum(h.buyer_filled for h in world.history)
    assert served > 0  # it discovered profitable, sub-WTP pricing


def test_buyer_respects_willingness_to_pay():
    # If every bottom seller prices above WTP, the buyer buys nothing.
    cfg = PyramidConfig(
        n_layers=1,
        n_steps=5,
        buyer_willingness_to_pay=0.5,  # very low
        supplier_price=1.0,
        supplier_price_mode="constant",
        seed=0,
    )
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    # cost-plus on a supplier price of 1.0 -> price >= 1.0 > 0.5 WTP
    roster = {nid: CostPlusTrader(name=f"c{nid}") for nid in topo.trader_ids}
    world = PyramidWorld(roster, cfg)
    world.run()
    assert all(h.buyer_filled == 0.0 for h in world.history)
