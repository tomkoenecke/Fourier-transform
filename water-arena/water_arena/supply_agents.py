"""Scripted strategies for the pyramid supply chain.

A trader must do two things each step: **price** its water for its children and
**order** replenishment from its parent(s), respecting pipeline capacities, its
cash, and its storage. These strategies illustrate the main archetypes -- a
cost-plus reseller, a cut-throat discounter, a margin-hungry monopolist, and a
speculator that stocks up when water is cheap upstream.

Write your own by subclassing :class:`SupplyAgent` and returning a
:class:`SupplyAction` (a sell price plus a ``{parent_id: quantity}`` order map).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .pyramid import SupplyAction, SupplyAgent, SupplyObservation


def _bottom_price(price: float, cost: float, obs: SupplyObservation) -> float:
    """For a bottom-layer node, don't bother pricing above the buyer's
    willingness-to-pay (it would just walk away) -- but never sell below the
    input cost either. Non-bottom nodes keep their price unchanged."""
    if obs.is_bottom and obs.buyer_willingness_to_pay is not None:
        return max(cost, min(price, obs.buyer_willingness_to_pay))
    return price


def _order_up_to(obs: SupplyObservation, target: float, fraction: float = 1.0) -> Dict[int, float]:
    """Order toward an inventory ``target``, sourcing from the cheapest parents
    first within pipeline limits.

    ``fraction`` < 1 closes only part of the gap each step -- proportional
    ("order-up-to with smoothing") replenishment, which damps the bullwhip
    oscillation that aggressive full-gap ordering produces.
    """
    deficit = max(0.0, target - obs.inventory) * fraction
    if deficit <= 0 or not obs.parent_prices:
        return {}
    orders: Dict[int, float] = {}
    remaining = min(deficit, obs.storage_headroom)
    for pid in sorted(obs.parent_prices, key=lambda p: obs.parent_prices[p]):
        if remaining <= 0:
            break
        cap = obs.parent_capacities.get(pid, 0.0)
        affordable = obs.cash / obs.parent_prices[pid] if obs.parent_prices[pid] > 0 else 0.0
        q = min(remaining, cap, affordable)
        if q > 0:
            orders[pid] = q
            remaining -= q
    return orders


class CostPlusTrader(SupplyAgent):
    """The textbook reseller: mark up input cost by a fixed fraction and keep a
    few steps of demand in stock (a base-stock policy)."""

    def __init__(self, name: Optional[str] = None, markup: float = 0.25, coverage: float = 3.0):
        super().__init__(name)
        self.markup = markup
        self.coverage = coverage

    def act(self, obs: SupplyObservation) -> SupplyAction:
        cost = obs.cheapest_parent_price
        price = cost * (1.0 + self.markup)
        price = _bottom_price(price, cost, obs)
        expected_demand = max(obs.last_sold, 2.0)
        target = self.coverage * expected_demand
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class Discounter(SupplyAgent):
    """Wins the cheapest-first buyer with a thin margin and high volume."""

    def __init__(self, name: Optional[str] = None, markup: float = 0.08, coverage: float = 4.0):
        super().__init__(name)
        self.markup = markup
        self.coverage = coverage

    def act(self, obs: SupplyObservation) -> SupplyAction:
        cost = obs.cheapest_parent_price
        price = cost * (1.0 + self.markup)
        price = _bottom_price(price, cost, obs)
        expected_demand = max(obs.last_sold, 3.0)
        target = self.coverage * expected_demand
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class Monopolist(SupplyAgent):
    """Charges a fat margin and stocks conservatively. Fine when it has pricing
    power; punished when a discounter undercuts it into dead stock."""

    def __init__(self, name: Optional[str] = None, markup: float = 0.9, coverage: float = 2.0):
        super().__init__(name)
        self.markup = markup
        self.coverage = coverage

    def act(self, obs: SupplyObservation) -> SupplyAction:
        cost = obs.cheapest_parent_price
        price = cost * (1.0 + self.markup)
        price = _bottom_price(price, cost, obs)
        expected_demand = max(obs.last_sold, 2.0)
        target = self.coverage * expected_demand
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class Speculator(SupplyAgent):
    """Times the supplier's price cycle: stocks up when upstream water is cheap
    relative to its recent range, coasts on inventory when it's dear."""

    def __init__(self, name: Optional[str] = None, markup: float = 0.3, window: int = 40):
        super().__init__(name)
        self.markup = markup
        self.window = window
        self._seen: List[float] = []

    def reset(self) -> None:
        self._seen = []

    def act(self, obs: SupplyObservation) -> SupplyAction:
        cost = obs.cheapest_parent_price
        self._seen.append(cost)
        self._seen = self._seen[-self.window :]
        lo = min(self._seen)
        hi = max(self._seen)
        cheap = cost <= lo + 0.3 * (hi - lo) if hi > lo else True

        price = cost * (1.0 + self.markup)
        price = _bottom_price(price, cost, obs)

        expected_demand = max(obs.last_sold, 2.0)
        coverage = 6.0 if cheap else 1.0  # load up cheap, ride it out when dear
        target = coverage * expected_demand
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class RandomTrader(SupplyAgent):
    """Random margins and order sizes -- a noisy baseline."""

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name)
        self._rng = random.Random(seed)

    def act(self, obs: SupplyObservation) -> SupplyAction:
        cost = obs.cheapest_parent_price
        price = cost * (1.0 + self._rng.uniform(0.0, 0.8))
        price = _bottom_price(price, cost, obs)
        target = self._rng.uniform(0.0, obs.storage_capacity)
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


# --------------------------------------------------------------------------- #
# Advanced strategies
# --------------------------------------------------------------------------- #
class EWMAReplenisher(SupplyAgent):
    """Forecast-driven base-stock with smoothing.

    Forecasts demand with an exponentially-weighted moving average of recent
    sales, targets a safety-stock covering the lead time plus a buffer, and
    closes the inventory gap only partially each step. That proportional
    ordering is the textbook antidote to the bullwhip effect -- inventories
    ride much flatter than a full order-up-to policy."""

    def __init__(
        self,
        name: Optional[str] = None,
        markup: float = 0.2,
        alpha: float = 0.3,
        safety: float = 2.0,
        smoothing: float = 0.5,
    ):
        super().__init__(name)
        self.markup = markup
        self.alpha = alpha
        self.safety = safety
        self.smoothing = smoothing
        self._forecast: Optional[float] = None

    def reset(self) -> None:
        self._forecast = None

    def act(self, obs: SupplyObservation) -> SupplyAction:
        if self._forecast is None:
            self._forecast = max(obs.last_sold, 2.0)
        else:
            self._forecast = self.alpha * obs.last_sold + (1 - self.alpha) * self._forecast

        cost = obs.cheapest_parent_price
        price = _bottom_price(cost * (1.0 + self.markup), cost, obs)
        # base-stock level: cover lead time (1 step) + a safety buffer of demand
        target = self._forecast * (1.0 + self.safety)
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target, self.smoothing))


class AdaptivePricer(SupplyAgent):
    """A feedback price controller.

    Nudges its margin up when it sells through its stock (demand is strong or it
    is under-pricing) and down when inventory piles up (it is being undercut or
    over-pricing). For a bottom node facing the cheapest-first buyer, this is a
    hands-off way to discover a competitive price without ever seeing rivals."""

    def __init__(
        self,
        name: Optional[str] = None,
        markup: float = 0.3,
        step: float = 0.05,
        lo: float = 0.02,
        hi: float = 1.5,
        coverage: float = 3.0,
    ):
        super().__init__(name)
        self.markup = markup
        self.step = step
        self.lo = lo
        self.hi = hi
        self.coverage = coverage
        self._markup = markup

    def reset(self) -> None:
        self._markup = self.markup

    def act(self, obs: SupplyObservation) -> SupplyAction:
        expected = max(obs.last_sold, 1.0)
        # glut -> cut margin to move stock; scarce -> raise margin
        if obs.inventory > 3.0 * expected:
            self._markup -= self.step
        elif obs.inventory < 1.0 * expected:
            self._markup += self.step
        self._markup = max(self.lo, min(self.hi, self._markup))

        cost = obs.cheapest_parent_price
        price = _bottom_price(cost * (1.0 + self._markup), cost, obs)
        target = self.coverage * expected
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class InventoryAwarePricer(SupplyAgent):
    """Dynamic revenue management: price scales with scarcity.

    Charges a premium when inventory is thin (ration it) and discounts when the
    tank is full (clear it), around a base margin -- a continuous version of
    surge pricing driven by its own stock position."""

    def __init__(
        self,
        name: Optional[str] = None,
        base_markup: float = 0.35,
        sensitivity: float = 0.6,
        coverage: float = 3.0,
    ):
        super().__init__(name)
        self.base_markup = base_markup
        self.sensitivity = sensitivity
        self.coverage = coverage

    def act(self, obs: SupplyObservation) -> SupplyAction:
        expected = max(obs.last_sold, 1.0)
        target = self.coverage * expected
        # scarcity in [-1, 1]: +1 when empty, -1 when at/above target
        scarcity = max(-1.0, min(1.0, (target - obs.inventory) / target)) if target > 0 else 0.0
        markup = max(0.02, self.base_markup + self.sensitivity * scarcity)

        cost = obs.cheapest_parent_price
        price = _bottom_price(cost * (1.0 + markup), cost, obs)
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class JustInTimeTrader(SupplyAgent):
    """Lean operator: holds almost no stock, orders exactly what it expects to
    sell, and runs a thin margin. Minimal holding cost, but a demand spike or a
    supply hiccup leaves it stocked out."""

    def __init__(self, name: Optional[str] = None, markup: float = 0.12, buffer: float = 1.2):
        super().__init__(name)
        self.markup = markup
        self.buffer = buffer

    def act(self, obs: SupplyObservation) -> SupplyAction:
        cost = obs.cheapest_parent_price
        price = _bottom_price(cost * (1.0 + self.markup), cost, obs)
        target = self.buffer * max(obs.last_sold, 1.0)
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


class BanditPricer(SupplyAgent):
    """An online learner: an epsilon-greedy multi-armed bandit over a menu of
    markups, using realised per-step profit (its own change in cash) as the
    reward. It has no model of the market -- it simply drifts toward whatever
    margin has paid best lately. A useful yardstick for RL agents: if your
    trained policy can't beat a bandit, it hasn't learned much."""

    def __init__(
        self,
        name: Optional[str] = None,
        markups: Optional[List[float]] = None,
        epsilon: float = 0.1,
        lr: float = 0.1,
        coverage: float = 3.0,
        seed: Optional[int] = None,
    ):
        super().__init__(name)
        self.markups = markups or [0.05, 0.15, 0.3, 0.5, 0.8]
        self.epsilon = epsilon
        self.lr = lr
        self.coverage = coverage
        self._rng = random.Random(seed)
        self._value = [0.0 for _ in self.markups]
        self._last_arm: Optional[int] = None
        self._prev_cash: Optional[float] = None

    def reset(self) -> None:
        self._value = [0.0 for _ in self.markups]
        self._last_arm = None
        self._prev_cash = None

    def act(self, obs: SupplyObservation) -> SupplyAction:
        # attribute last step's profit (change in cash) to the arm we played
        if self._last_arm is not None and self._prev_cash is not None:
            reward = obs.cash - self._prev_cash
            v = self._value[self._last_arm]
            self._value[self._last_arm] = v + self.lr * (reward - v)

        if self._rng.random() < self.epsilon:
            arm = self._rng.randrange(len(self.markups))
        else:
            best = max(self._value)
            arm = self._value.index(best)

        cost = obs.cheapest_parent_price
        price = _bottom_price(cost * (1.0 + self.markups[arm]), cost, obs)
        target = self.coverage * max(obs.last_sold, 2.0)

        self._last_arm = arm
        self._prev_cash = obs.cash
        return SupplyAction(sell_price=price, orders=_order_up_to(obs, target))


def default_supply_roster(trader_ids) -> Dict[int, SupplyAgent]:
    """Assign a rotating mix of strategies -- basic and advanced -- across the
    trader nodes."""
    factories = [
        lambda i: CostPlusTrader(name=f"CostPlus{i}"),
        lambda i: Discounter(name=f"Discount{i}"),
        lambda i: Monopolist(name=f"Monopol{i}"),
        lambda i: Speculator(name=f"Specul{i}"),
        lambda i: EWMAReplenisher(name=f"EWMA{i}"),
        lambda i: AdaptivePricer(name=f"Adaptive{i}"),
        lambda i: InventoryAwarePricer(name=f"InvAware{i}"),
        lambda i: JustInTimeTrader(name=f"JIT{i}"),
        lambda i: BanditPricer(name=f"Bandit{i}", seed=i),
    ]
    roster: Dict[int, SupplyAgent] = {}
    for i, nid in enumerate(trader_ids):
        roster[nid] = factories[i % len(factories)](i)
    return roster


def advanced_supply_roster(trader_ids) -> Dict[int, SupplyAgent]:
    """A roster of only the advanced strategies, cycled across the nodes."""
    factories = [
        lambda i: EWMAReplenisher(name=f"EWMA{i}"),
        lambda i: AdaptivePricer(name=f"Adaptive{i}"),
        lambda i: InventoryAwarePricer(name=f"InvAware{i}"),
        lambda i: JustInTimeTrader(name=f"JIT{i}"),
        lambda i: BanditPricer(name=f"Bandit{i}", seed=i),
    ]
    roster: Dict[int, SupplyAgent] = {}
    for i, nid in enumerate(trader_ids):
        roster[nid] = factories[i % len(factories)](i)
    return roster
