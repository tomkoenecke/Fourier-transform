"""Uniform-price call-auction market clearing.

Every step, agents submit orders. All orders are collected and cleared at a
single market-clearing price -- the price that maximises the traded volume
(the classic call-auction / uniform-price rule). This gives clean price
discovery while keeping the mechanism simple enough to reason about.

The module is pure standard library so the core engine has no dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

_EPS = 1e-9


@dataclass
class Order:
    """A single buy or sell order submitted to the auction."""

    agent_id: int
    side: str  # "buy" or "sell"
    price: float  # limit price: max a buyer pays / min a seller accepts
    quantity: float  # units of water, always positive

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.quantity < 0:
            raise ValueError("order quantity must be non-negative")


@dataclass
class Trade:
    """The fill an agent received from clearing."""

    agent_id: int
    side: str  # "buy" or "sell"
    price: float  # the uniform clearing price
    quantity: float  # units actually transacted


@dataclass
class MarketResult:
    """Outcome of clearing one auction round."""

    clearing_price: Optional[float]
    volume: float
    trades: List[Trade]


def clear_market(orders: List[Order]) -> MarketResult:
    """Clear a batch of orders at a single uniform price.

    The clearing price is chosen to maximise traded volume; ties are broken by
    taking the midpoint of the price range that achieves that volume. Orders on
    the "long" side of the book are filled by price priority (best price first),
    with the marginal order partially filled.

    Returns a :class:`MarketResult`. If no trade is possible the clearing price
    is ``None`` and no trades are returned.
    """
    bids = [o for o in orders if o.side == "buy" and o.quantity > _EPS and o.price > 0]
    asks = [o for o in orders if o.side == "sell" and o.quantity > _EPS and o.price > 0]

    if not bids or not asks:
        return MarketResult(clearing_price=None, volume=0.0, trades=[])

    # Candidate clearing prices are exactly the submitted limit prices: the
    # aggregate demand/supply curves are step functions that only change there.
    candidates = sorted({o.price for o in bids} | {o.price for o in asks})

    best_volume = 0.0
    volume_maximising: List[float] = []
    for price in candidates:
        demand = sum(o.quantity for o in bids if o.price >= price - _EPS)
        supply = sum(o.quantity for o in asks if o.price <= price + _EPS)
        volume = min(demand, supply)
        if volume > best_volume + _EPS:
            best_volume = volume
            volume_maximising = [price]
        elif abs(volume - best_volume) <= _EPS and volume > _EPS:
            volume_maximising.append(price)

    if best_volume <= _EPS:
        return MarketResult(clearing_price=None, volume=0.0, trades=[])

    # Midpoint of the volume-maximising price range is the uniform price.
    clearing_price = (min(volume_maximising) + max(volume_maximising)) / 2.0

    eligible_buys = sorted(
        (o for o in bids if o.price >= clearing_price - _EPS),
        key=lambda o: -o.price,  # highest bid gets priority
    )
    eligible_sells = sorted(
        (o for o in asks if o.price <= clearing_price + _EPS),
        key=lambda o: o.price,  # lowest ask gets priority
    )

    traded = min(
        sum(o.quantity for o in eligible_buys),
        sum(o.quantity for o in eligible_sells),
    )

    trades: List[Trade] = []
    trades.extend(_fill(eligible_buys, traded, "buy", clearing_price))
    trades.extend(_fill(eligible_sells, traded, "sell", clearing_price))

    return MarketResult(clearing_price=clearing_price, volume=traded, trades=trades)


def _fill(
    side_orders: List[Order], total: float, side: str, price: float
) -> List[Trade]:
    """Fill ``total`` units across ``side_orders`` in priority order."""
    trades: List[Trade] = []
    remaining = total
    for order in side_orders:
        if remaining <= _EPS:
            break
        qty = min(order.quantity, remaining)
        if qty <= _EPS:
            continue
        trades.append(Trade(agent_id=order.agent_id, side=side, price=price, quantity=qty))
        remaining -= qty
    return trades
