"""Agent interface and a library of scripted trading strategies.

An agent observes the world (:class:`Observation`) and returns an
:class:`Action`: a limit price and a *signed* quantity (positive = buy,
negative = sell, zero = do nothing). The simulation validates and clips the
action to what the agent can actually afford / hold before it hits the market.

To write your own strategy, subclass :class:`Agent` and implement ``act``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Observation:
    """Everything an agent sees at decision time (after production, before
    consumption and trading)."""

    step: int
    max_steps: int
    n_agents: int
    cash: float
    water: float
    well_capacity: float  # this agent's base production per step
    storage_capacity: float  # max water it can hold
    demand: float  # units it must consume this step
    unit_value: float  # cash earned per unit consumed (water's value to this agent)
    last_price: Optional[float]  # last clearing price, None before first trade
    price_history: List[float] = field(default_factory=list)

    @property
    def storage_headroom(self) -> float:
        return max(0.0, self.storage_capacity - self.water)

    @property
    def reference_price(self) -> float:
        """A sensible fallback price when the market has not cleared yet."""
        return self.last_price if self.last_price is not None else 1.0


@dataclass
class Action:
    """An agent's desired order. ``quantity`` is signed: >0 buy, <0 sell."""

    price: float
    quantity: float

    @staticmethod
    def hold() -> "Action":
        return Action(price=0.0, quantity=0.0)


class Agent:
    """Base class for all strategies."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def reset(self) -> None:
        """Called at the start of each episode. Override to clear state."""

    def act(self, obs: Observation) -> Action:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# --------------------------------------------------------------------------- #
# Scripted strategies
# --------------------------------------------------------------------------- #


class DoNothingAgent(Agent):
    """Never trades. A baseline; slowly dies of thirst if it can't self-supply."""

    def act(self, obs: Observation) -> Action:
        return Action.hold()


class RandomAgent(Agent):
    """Submits random orders around the reference price."""

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name)
        self._rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        ref = obs.reference_price
        price = ref * self._rng.uniform(0.8, 1.2)
        if self._rng.random() < 0.5:  # buy
            qty = self._rng.uniform(0, obs.storage_headroom)
        else:  # sell
            qty = -self._rng.uniform(0, obs.water)
        return Action(price=price, quantity=qty)


class NeedsTrader(Agent):
    """Rational subsistence trader.

    Buys just enough to cover its expected consumption plus a safety buffer,
    and sells water held above that buffer. Trades a small margin off the
    reference price so it actually crosses the book.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        buffer_steps: float = 3.0,
        margin: float = 0.1,
    ):
        super().__init__(name)
        self.buffer_steps = buffer_steps
        self.margin = margin

    def act(self, obs: Observation) -> Action:
        target = obs.demand * self.buffer_steps
        ref = obs.reference_price
        if obs.water < target:
            deficit = target - obs.water
            qty = min(deficit, obs.storage_headroom)
            # bid up, but never above what the water is worth to us
            price = min(ref * (1 + self.margin), obs.unit_value)
            return Action(price=price, quantity=qty)
        surplus = obs.water - target
        if surplus > 0:
            return Action(price=ref * (1 - self.margin), quantity=-surplus)  # offer down
        return Action.hold()


class Hoarder(Agent):
    """Buys aggressively whenever water looks cheap; rarely sells.

    Bets on scarcity: accumulate now, ride price spikes later.
    """

    def __init__(self, name: Optional[str] = None, cheap_below: float = 0.9):
        super().__init__(name)
        self.cheap_below = cheap_below

    def act(self, obs: Observation) -> Action:
        history = obs.price_history
        if not history:
            # No signal yet: cover subsistence conservatively.
            return Action(price=min(obs.reference_price, obs.unit_value), quantity=obs.demand)
        avg = sum(history) / len(history)
        price = obs.last_price if obs.last_price is not None else avg
        if price <= avg * self.cheap_below:
            qty = obs.storage_headroom  # back up the truck
            return Action(price=min(price * 1.05, obs.unit_value), quantity=qty)
        # Only sell to avoid overflow / raise emergency cash.
        if obs.storage_headroom <= 0 and obs.water > obs.demand:
            return Action(price=avg * 1.2, quantity=-(obs.water - obs.demand))
        return Action.hold()


class MeanReversionTrader(Agent):
    """Estimates fair value from a moving average and fades deviations.

    Buys below fair value, sells above it -- classic speculation that also
    supplies liquidity to the market.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        window: int = 10,
        band: float = 0.05,
        aggressiveness: float = 0.5,
    ):
        super().__init__(name)
        self.window = window
        self.band = band
        self.aggressiveness = aggressiveness

    def act(self, obs: Observation) -> Action:
        history = obs.price_history[-self.window :]
        if len(history) < 2 or obs.last_price is None:
            return Action(price=min(obs.reference_price, obs.unit_value), quantity=obs.demand)
        fair = sum(history) / len(history)
        price = obs.last_price
        # Always keep enough for subsistence out of the tradeable pool.
        reserve = obs.demand
        if price < fair * (1 - self.band):
            qty = obs.storage_headroom * self.aggressiveness
            return Action(price=min(fair, obs.unit_value), quantity=qty)  # buy toward fair value
        if price > fair * (1 + self.band):
            sellable = max(0.0, obs.water - reserve) * self.aggressiveness
            return Action(price=fair, quantity=-sellable)  # sell toward fair value
        return Action.hold()


class TrendFollower(Agent):
    """Momentum trader: buys when price rises, sells when it falls."""

    def __init__(self, name: Optional[str] = None, lookback: int = 3):
        super().__init__(name)
        self.lookback = lookback

    def act(self, obs: Observation) -> Action:
        history = obs.price_history
        if len(history) <= self.lookback or obs.last_price is None:
            return Action(price=min(obs.reference_price, obs.unit_value), quantity=obs.demand)
        past = history[-self.lookback - 1]
        now = history[-1]
        if now > past * 1.01:  # uptrend
            return Action(price=min(now * 1.05, obs.unit_value), quantity=obs.storage_headroom * 0.5)
        if now < past * 0.99:  # downtrend
            sellable = max(0.0, obs.water - obs.demand)
            return Action(price=now * 0.95, quantity=-sellable * 0.5)
        return Action.hold()


#: Convenient roster used by the example tournament.
def default_roster() -> List[Agent]:
    return [
        NeedsTrader(name="Needs"),
        Hoarder(name="Hoarder"),
        MeanReversionTrader(name="MeanRev"),
        TrendFollower(name="Trend"),
        RandomAgent(name="Random", seed=0),
        DoNothingAgent(name="Idle"),
    ]
