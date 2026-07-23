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


def _order_up_to(obs: SupplyObservation, target: float) -> Dict[int, float]:
    """Order enough to bring inventory (plus what's already on hand) up to
    ``target``, sourcing from the cheapest parents first within pipeline limits."""
    deficit = max(0.0, target - obs.inventory)
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


def default_supply_roster(trader_ids) -> Dict[int, SupplyAgent]:
    """Assign a rotating mix of strategies across the trader nodes."""
    factories = [
        lambda i: CostPlusTrader(name=f"CostPlus{i}"),
        lambda i: Discounter(name=f"Discount{i}"),
        lambda i: Monopolist(name=f"Monopol{i}"),
        lambda i: Speculator(name=f"Specul{i}"),
    ]
    roster: Dict[int, SupplyAgent] = {}
    for i, nid in enumerate(trader_ids):
        roster[nid] = factories[i % len(factories)](i)
    return roster
