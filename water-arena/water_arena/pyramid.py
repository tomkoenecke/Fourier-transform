"""Pyramid supply-chain model: water flows down a Pascal's-triangle network.

A single **supplier** sits at the apex and posts a price. Below it are trader
layers that grow 2, 3, 4, ...: node *i* in a layer feeds children *i* and *i+1*
in the layer below, so adjacent traders share the middle child. Every edge is a
**pipeline with a maximum flow rate**.

Each trader is a middleman: it buys water from its parent(s), holds inventory,
and resells to its children at a price it chooses. Purchases arrive with a
one-step lead time (inventory buffers absorb the mismatch), so *when* and *how
much* to stock -- and at what margin to sell -- is the core decision.

At the bottom, a single **buyer** fills a per-step need by taking the cheapest
water on offer, up to a willingness-to-pay cap. That price competition is what
propagates scarcity signals back up the pyramid.

The engine is pure standard library and deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
@dataclass
class Topology:
    """The pyramid network: who feeds whom, and pipeline capacities."""

    n_layers: int
    layers: List[List[int]]  # layers[L] = node ids in layer L (layer 0 = supplier)
    parents: Dict[int, List[int]]
    children: Dict[int, List[int]]
    capacity: Dict[tuple, float]  # (src, dst) -> max flow per step
    supplier_id: int
    trader_ids: List[int]
    bottom_ids: List[int]  # traders that sell to the buyer
    buyer_id: int
    pos: Dict[int, int]  # node id -> position within its layer
    layer_of: Dict[int, int]

    def label(self, node_id: int) -> str:
        if node_id == self.supplier_id:
            return "Supplier"
        if node_id == self.buyer_id:
            return "Buyer"
        return f"L{self.layer_of[node_id]}#{self.pos[node_id]}"


def build_pyramid(n_layers: int, pipeline_capacity: float) -> Topology:
    """Build a pyramid with ``n_layers`` trader layers of sizes 2, 3, ..., n+1."""
    if n_layers < 1:
        raise ValueError("need at least one trader layer")

    layers: List[List[int]] = [[0]]  # layer 0 = supplier (id 0)
    next_id = 1
    for L in range(1, n_layers + 1):
        size = L + 1  # layer 1 -> 2 nodes, layer 2 -> 3, ...
        layers.append(list(range(next_id, next_id + size)))
        next_id += size
    buyer_id = next_id

    pos = {nid: i for layer in layers for i, nid in enumerate(layer)}
    layer_of = {nid: L for L, layer in enumerate(layers) for nid in layer}

    parents: Dict[int, List[int]] = {}
    children: Dict[int, List[int]] = {}
    capacity: Dict[tuple, float] = {}
    for nid in range(next_id):
        parents.setdefault(nid, [])
        children.setdefault(nid, [])
    parents[buyer_id] = []
    children[buyer_id] = []

    # node i in layer L feeds children i and i+1 in layer L+1
    for L in range(0, n_layers):
        for i, nid in enumerate(layers[L]):
            for child in (layers[L + 1][i], layers[L + 1][i + 1]):
                children[nid].append(child)
                parents[child].append(nid)
                capacity[(nid, child)] = pipeline_capacity

    # bottom trader layer feeds the buyer
    bottom_ids = list(layers[n_layers])
    for nid in bottom_ids:
        children[nid].append(buyer_id)
        parents[buyer_id].append(nid)
        capacity[(nid, buyer_id)] = pipeline_capacity

    trader_ids = [nid for L in range(1, n_layers + 1) for nid in layers[L]]

    return Topology(
        n_layers=n_layers,
        layers=layers,
        parents=parents,
        children=children,
        capacity=capacity,
        supplier_id=0,
        trader_ids=trader_ids,
        bottom_ids=bottom_ids,
        buyer_id=buyer_id,
        pos=pos,
        layer_of=layer_of,
    )


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class PyramidConfig:
    """Parameters of the pyramid economy."""

    n_layers: int = 3  # trader layers below the supplier (sizes 2, 3, 4, ...)
    n_steps: int = 300

    pipeline_capacity: float = 8.0  # max flow per pipeline per step
    storage_capacity: float = 30.0
    initial_cash: float = 100.0
    initial_inventory: float = 8.0
    holding_cost: float = 0.01  # cash per unit of inventory per step

    # supplier price process
    supplier_price: float = 1.0
    supplier_price_mode: str = "sine"  # "constant" | "sine" | "random_walk"
    supplier_price_amp: float = 0.35  # relative amplitude / step size
    supplier_price_period: float = 60.0
    supplier_capacity: float = 1e12  # supplier's water is effectively unlimited

    # final buyer
    buyer_need: float = 12.0  # units the buyer wants each step
    buyer_need_noise: float = 0.15  # relative std-dev of need
    buyer_willingness_to_pay: float = 4.0  # max price the buyer will accept

    allow_bankruptcy: bool = True
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# agent-facing observation / action
# --------------------------------------------------------------------------- #
@dataclass
class SupplyObservation:
    """What a trader node sees before it acts."""

    step: int
    max_steps: int
    node_id: int
    layer: int
    position: int
    is_bottom: bool

    inventory: float
    cash: float
    storage_capacity: float
    storage_headroom: float

    last_sell_price: float
    last_sold: float  # units sold last step (demand signal)
    last_bought: float

    supplier_price: float  # current apex price
    parent_prices: Dict[int, float]  # parent id -> its sell price (last step)
    parent_capacities: Dict[int, float]  # parent id -> pipeline max flow
    child_capacities: Dict[int, float]  # child id -> pipeline max flow

    # only meaningful for bottom-layer nodes
    buyer_willingness_to_pay: Optional[float] = None

    @property
    def cheapest_parent_price(self) -> float:
        if not self.parent_prices:
            return self.supplier_price
        return min(self.parent_prices.values())


@dataclass
class SupplyAction:
    """A trader's decision: what to charge, and how much to order upstream."""

    sell_price: float
    orders: Dict[int, float] = field(default_factory=dict)  # parent id -> qty

    @staticmethod
    def hold(sell_price: float) -> "SupplyAction":
        return SupplyAction(sell_price=sell_price, orders={})


class SupplyAgent:
    """Base class for a trader strategy. Subclass and implement ``act``."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def reset(self) -> None:
        """Called at the start of each episode."""

    def act(self, obs: SupplyObservation) -> SupplyAction:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# --------------------------------------------------------------------------- #
# node state
# --------------------------------------------------------------------------- #
@dataclass
class NodeState:
    node_id: int
    inventory: float
    cash: float
    sell_price: float
    alive: bool = True
    # per-step scratch
    incoming: float = 0.0
    sold_this: float = 0.0
    bought_this: float = 0.0
    # previous completed step (demand signal for agents)
    last_sold_v: float = 0.0
    last_bought_v: float = 0.0
    # cumulative stats
    total_sold: float = 0.0
    total_bought: float = 0.0
    total_profit: float = 0.0
    died_at: Optional[int] = None


@dataclass
class PyramidStep:
    """Snapshot of one step, for logging / plotting."""

    step: int
    supplier_price: float
    buyer_need: float
    buyer_filled: float
    buyer_avg_price: Optional[float]
    inventory: Dict[int, float]
    cash: Dict[int, float]
    profit: Dict[int, float]


# --------------------------------------------------------------------------- #
# the simulation
# --------------------------------------------------------------------------- #
class PyramidWorld:
    """A multi-echelon water supply chain shaped like a pyramid."""

    def __init__(
        self,
        agents: Dict[int, SupplyAgent],
        config: Optional[PyramidConfig] = None,
        supplier_price_fn: Optional[Callable[[int], float]] = None,
    ):
        self.config = config or PyramidConfig()
        self.topology = build_pyramid(self.config.n_layers, self.config.pipeline_capacity)
        missing = set(self.topology.trader_ids) - set(agents)
        if missing:
            raise ValueError(f"no agent supplied for trader nodes {sorted(missing)}")
        self.agents = agents
        self._custom_price_fn = supplier_price_fn
        self.states: Dict[int, NodeState] = {}
        self._rng = random.Random(self.config.seed)
        self._sp = self.config.supplier_price
        self.step_count = 0
        self.history: List[PyramidStep] = []
        self.total_unmet = 0.0
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        cfg = self.config
        self._rng = random.Random(cfg.seed)
        self._sp = cfg.supplier_price
        self.step_count = 0
        self.history = []
        self.total_unmet = 0.0
        # volume-weighted price accumulators (for price_cascade)
        self._price_num = {nid: 0.0 for nid in self.topology.trader_ids}
        self._price_vol = {nid: 0.0 for nid in self.topology.trader_ids}
        self._supplier_price_sum = 0.0
        self._buyer_price_num = 0.0
        self._buyer_vol = 0.0
        self.states = {}
        for nid in self.topology.trader_ids:
            self.states[nid] = NodeState(
                node_id=nid,
                inventory=cfg.initial_inventory,
                cash=cfg.initial_cash,
                sell_price=cfg.supplier_price,
            )
        for agent in self.agents.values():
            agent.reset()

    def done(self) -> bool:
        if self.step_count >= self.config.n_steps:
            return True
        return not any(s.alive for s in self.states.values())

    # ------------------------------------------------------------------ #
    def step(self) -> PyramidStep:
        cfg = self.config
        topo = self.topology

        supplier_price = self._supplier_price()
        prev_cash = {nid: s.cash for nid, s in self.states.items()}
        prev_sell = {nid: s.sell_price for nid, s in self.states.items()}

        # reset per-step scratch
        for s in self.states.values():
            s.incoming = 0.0
            s.sold_this = 0.0
            s.bought_this = 0.0

        # 1. every trader decides sell price + upstream orders
        orders: Dict[int, Dict[int, float]] = {}
        for nid in topo.trader_ids:
            s = self.states[nid]
            if not s.alive:
                continue
            obs = self._observe(nid, supplier_price, prev_sell)
            action = self.agents[nid].act(obs)
            s.sell_price = max(0.0, action.sell_price)
            orders[nid] = {p: max(0.0, q) for p, q in action.orders.items()}

        # 2. top-down selling from start-of-step inventory (purchases land next step)
        for L in range(0, topo.n_layers):
            for seller in topo.layers[L]:
                is_supplier = seller == topo.supplier_id
                if not is_supplier and not self.states[seller].alive:
                    continue
                price = supplier_price if is_supplier else self.states[seller].sell_price
                available = cfg.supplier_capacity if is_supplier else self.states[seller].inventory
                self._sell(seller, price, available, is_supplier, orders)

        # 3. final buyer takes the cheapest water up to its need
        buyer_need = max(0.0, self._rng.gauss(cfg.buyer_need, cfg.buyer_need * cfg.buyer_need_noise))
        filled, avg_price = self._buyer_purchase(buyer_need)
        self.total_unmet += buyer_need - filled

        # price-cascade accumulators (volume-weighted)
        self._supplier_price_sum += supplier_price
        if avg_price is not None:
            self._buyer_price_num += avg_price * filled
            self._buyer_vol += filled

        # 4. deliveries arrive; apply holding cost
        for s in self.states.values():
            if not s.alive:
                continue
            s.inventory = min(cfg.storage_capacity, s.inventory + s.incoming)
            s.cash -= cfg.holding_cost * s.inventory
            s.total_sold += s.sold_this
            s.total_bought += s.bought_this
            s.last_sold_v = s.sold_this
            s.last_bought_v = s.bought_this
            self._price_num[s.node_id] += s.sell_price * s.sold_this
            self._price_vol[s.node_id] += s.sold_this

        # 5. bankruptcy
        if cfg.allow_bankruptcy:
            for s in self.states.values():
                if s.alive and s.cash < -_EPS:
                    s.alive = False
                    s.died_at = self.step_count + 1

        # 6. accounting
        profit = {nid: self.states[nid].cash - prev_cash[nid] for nid in self.states}
        for nid, p in profit.items():
            self.states[nid].total_profit += p

        self.step_count += 1
        info = PyramidStep(
            step=self.step_count,
            supplier_price=supplier_price,
            buyer_need=buyer_need,
            buyer_filled=filled,
            buyer_avg_price=avg_price,
            inventory={nid: s.inventory for nid, s in self.states.items()},
            cash={nid: s.cash for nid, s in self.states.items()},
            profit=profit,
        )
        self.history.append(info)
        return info

    def run(self, verbose: bool = False) -> List[PyramidStep]:
        while not self.done():
            info = self.step()
            if verbose:
                ap = f"{info.buyer_avg_price:.2f}" if info.buyer_avg_price is not None else "  -  "
                print(
                    f"step {info.step:4d} | supply {info.supplier_price:.2f} | "
                    f"buyer {info.buyer_filled:5.1f}/{info.buyer_need:5.1f} @ {ap}"
                )
        return self.history

    # ------------------------------------------------------------------ #
    # phases
    # ------------------------------------------------------------------ #
    def _supplier_price(self) -> float:
        cfg = self.config
        if self._custom_price_fn is not None:
            return max(0.0, self._custom_price_fn(self.step_count))
        mode = cfg.supplier_price_mode
        if mode == "constant":
            return cfg.supplier_price
        if mode == "sine":
            phase = 2 * math.pi * self.step_count / max(1.0, cfg.supplier_price_period)
            return max(0.05, cfg.supplier_price * (1.0 + cfg.supplier_price_amp * math.sin(phase)))
        if mode == "random_walk":
            self._sp += self._rng.gauss(0.0, cfg.supplier_price_amp * cfg.supplier_price * 0.15)
            self._sp = max(0.05, self._sp)
            return self._sp
        raise ValueError(f"unknown supplier_price_mode {mode!r}")

    def _observe(self, nid: int, supplier_price: float, prev_sell: Dict[int, float]) -> SupplyObservation:
        cfg = self.config
        topo = self.topology
        s = self.states[nid]
        parent_prices: Dict[int, float] = {}
        parent_caps: Dict[int, float] = {}
        for p in topo.parents[nid]:
            parent_caps[p] = topo.capacity[(p, nid)]
            if p == topo.supplier_id:
                parent_prices[p] = supplier_price
            else:
                parent_prices[p] = prev_sell.get(p, supplier_price)
        child_caps = {c: topo.capacity[(nid, c)] for c in topo.children[nid]}
        is_bottom = nid in topo.bottom_ids
        return SupplyObservation(
            step=self.step_count,
            max_steps=cfg.n_steps,
            node_id=nid,
            layer=topo.layer_of[nid],
            position=topo.pos[nid],
            is_bottom=is_bottom,
            inventory=s.inventory,
            cash=s.cash,
            storage_capacity=cfg.storage_capacity,
            storage_headroom=max(0.0, cfg.storage_capacity - s.inventory),
            last_sell_price=s.sell_price,
            last_sold=s.last_sold_v,
            last_bought=s.last_bought_v,
            supplier_price=supplier_price,
            parent_prices=parent_prices,
            parent_capacities=parent_caps,
            child_capacities=child_caps,
            buyer_willingness_to_pay=cfg.buyer_willingness_to_pay if is_bottom else None,
        )

    def _sell(
        self,
        seller: int,
        price: float,
        available: float,
        is_supplier: bool,
        orders: Dict[int, Dict[int, float]],
    ) -> None:
        cfg = self.config
        topo = self.topology
        # gather this seller's child orders, capped by capacity / storage / cash
        reqs: Dict[int, float] = {}
        for child in topo.children[seller]:
            if child == topo.buyer_id:
                continue  # the buyer is handled separately
            cs = self.states[child]
            if not cs.alive:
                continue
            desired = orders.get(child, {}).get(seller, 0.0)
            if desired <= _EPS:
                continue
            cap = topo.capacity[(seller, child)]
            headroom = cfg.storage_capacity - (cs.inventory + cs.incoming)
            afford = cs.cash / price if price > _EPS else float("inf")
            q = max(0.0, min(desired, cap, headroom, afford))
            if q > _EPS:
                reqs[child] = q

        total = sum(reqs.values())
        if total <= _EPS:
            return
        scale = 1.0 if total <= available + _EPS else available / total
        seller_state = None if is_supplier else self.states[seller]
        for child, q in reqs.items():
            fill = q * scale
            if fill <= _EPS:
                continue
            cs = self.states[child]
            cost = price * fill
            cs.cash -= cost
            cs.incoming += fill
            cs.bought_this += fill
            if seller_state is not None:
                seller_state.inventory -= fill
                seller_state.cash += cost
                seller_state.sold_this += fill

    def _buyer_purchase(self, need: float):
        cfg = self.config
        topo = self.topology
        remaining = need
        spend = 0.0
        volume = 0.0
        sellers = sorted(
            (self.states[nid] for nid in topo.bottom_ids if self.states[nid].alive),
            key=lambda s: s.sell_price,
        )
        for s in sellers:
            if remaining <= _EPS:
                break
            if s.sell_price > cfg.buyer_willingness_to_pay + _EPS:
                break  # everyone else is even pricier -> buyer walks away
            cap = topo.capacity[(s.node_id, topo.buyer_id)]
            qty = min(remaining, s.inventory, cap)
            if qty <= _EPS:
                continue
            s.inventory -= qty
            revenue = s.sell_price * qty
            s.cash += revenue
            s.sold_this += qty
            remaining -= qty
            spend += revenue
            volume += qty
        avg = spend / volume if volume > _EPS else None
        return volume, avg

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def price_cascade(self) -> Dict[str, object]:
        """Volume-weighted average price at each stage of the chain, from the
        supplier down to what the buyer actually pays.

        Returns ``{"supplier": float, "layers": [p1, p2, ...], "buyer": float}``
        where ``layers[k]`` is the mean price at which trader layer ``k+1`` sold,
        weighted by volume. Run the episode first.
        """
        topo = self.topology
        n = max(1, self.step_count)
        supplier = self._supplier_price_sum / n
        layers: List[Optional[float]] = []
        for L in range(1, topo.n_layers + 1):
            num = sum(self._price_num[nid] for nid in topo.layers[L])
            vol = sum(self._price_vol[nid] for nid in topo.layers[L])
            layers.append(num / vol if vol > _EPS else None)
        buyer = self._buyer_price_num / self._buyer_vol if self._buyer_vol > _EPS else None
        return {"supplier": supplier, "layers": layers, "buyer": buyer}

    def price_cascade_str(self) -> str:
        """A human-readable table of :meth:`price_cascade`."""
        c = self.price_cascade()
        sup = c["supplier"]
        lines = [
            "Price cascade (volume-weighted avg along the chain):",
            f"  {'stage':<14}{'price':>8}{'x supplier':>12}{'vs prev':>10}",
            f"  {'Supplier':<14}{sup:>8.3f}{1.0:>12.2f}{'':>10}",
        ]
        prev = sup
        for i, p in enumerate(c["layers"], start=1):
            if p is None:
                lines.append(f"  {'Layer '+str(i):<14}{'  -  ':>8}")
                continue
            step = f"{(p / prev - 1) * 100:+.0f}%" if prev else ""
            lines.append(f"  {'Layer '+str(i):<14}{p:>8.3f}{p / sup:>12.2f}{step:>10}")
            prev = p
        b = c["buyer"]
        if b is not None:
            step = f"{(b / prev - 1) * 100:+.0f}%" if prev else ""
            lines.append(f"  {'Buyer (paid)':<14}{b:>8.3f}{b / sup:>12.2f}{step:>10}")
        return "\n".join(lines)

    def leaderboard(self) -> List[NodeState]:
        return sorted(
            self.states.values(),
            key=lambda s: (s.alive, s.total_profit),
            reverse=True,
        )

    def summary(self) -> str:
        topo = self.topology
        served = 0.0
        needed = 0.0
        for h in self.history:
            served += h.buyer_filled
            needed += h.buyer_need
        fill_rate = 100.0 * served / needed if needed else 0.0
        lines = [
            f"Pyramid Arena -- {self.config.n_layers} trader layers, "
            f"{self.step_count} steps",
            f"buyer fill rate: {fill_rate:.1f}%  "
            f"(served {served:.0f} of {needed:.0f} units of demand)",
            f"{'node':<8}{'agent':<16}{'status':<10}"
            f"{'profit':>10}{'cash':>10}{'inv':>8}{'price':>8}"
            f"{'sold':>9}{'bought':>9}",
        ]
        for s in self.leaderboard():
            agent = self.agents[s.node_id]
            status = "alive" if s.alive else f"died@{s.died_at}"
            lines.append(
                f"{topo.label(s.node_id):<8}{agent.name:<16}{status:<10}"
                f"{s.total_profit:>10.2f}{s.cash:>10.2f}{s.inventory:>8.2f}"
                f"{s.sell_price:>8.2f}{s.total_sold:>9.1f}{s.total_bought:>9.1f}"
            )
        return "\n".join(lines)
