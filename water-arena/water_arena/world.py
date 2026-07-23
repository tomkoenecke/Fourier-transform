"""The water economy simulation.

Each step:

1. **Production** -- every agent's well produces water (stochastic, with the
   occasional market-wide drought), capped by storage.
2. **Decision** -- agents observe the world and submit orders.
3. **Market** -- all orders clear at a single uniform price; cash and water
   are settled.
4. **Consumption** -- each agent must consume its demand; any shortfall incurs
   a cash penalty (buying emergency water / going thirsty).
5. **Accounting** -- rewards (change in mark-to-market net worth) are recorded
   and bankrupt agents (cash < 0) are removed.

The engine is pure standard library. It is deterministic given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .agents import Action, Agent, Observation
from .market import Order, Trade, clear_market

_EPS = 1e-9


@dataclass
class WorldConfig:
    """Tunable parameters of the economy."""

    n_steps: int = 200
    initial_cash: float = 100.0
    initial_water: float = 10.0
    storage_capacity: float = 40.0
    well_capacity: float = 5.0  # mean production per step (across agents)
    well_capacity_spread: float = 0.6  # relative spread of per-agent wells
    production_noise: float = 0.4  # relative std-dev of production
    drought_prob: float = 0.05  # chance of a market-wide drought each step
    drought_factor: float = 0.2  # production multiplier during a drought
    base_demand: float = 5.0  # units each agent must consume per step
    demand_noise: float = 0.15  # relative std-dev of demand
    consumption_value: float = 2.0  # cash earned per unit consumed (utility revenue)
    unmet_penalty: float = 3.0  # extra cash penalty per unit of unmet demand
    initial_price: float = 1.0  # reference price before the first clear
    allow_bankruptcy: bool = True  # remove agents whose cash goes negative
    seed: Optional[int] = None


@dataclass
class AgentState:
    """Mutable per-agent state carried through an episode."""

    agent_id: int
    name: str
    cash: float
    water: float
    well_capacity: float
    storage_capacity: float
    alive: bool = True
    # cumulative stats
    total_bought: float = 0.0
    total_sold: float = 0.0
    total_consumed: float = 0.0
    total_shortfall: float = 0.0
    died_at: Optional[int] = None

    def net_worth(self, price: float) -> float:
        return self.cash + self.water * price


@dataclass
class StepInfo:
    """A snapshot of one simulation step (useful for logging / plotting)."""

    step: int
    clearing_price: Optional[float]
    volume: float
    reference_price: float
    drought: bool
    rewards: Dict[int, float]
    net_worth: Dict[int, float]


class World:
    """A multi-agent water-trading episode."""

    def __init__(self, agents: List[Agent], config: Optional[WorldConfig] = None):
        if not agents:
            raise ValueError("need at least one agent")
        self.agents = agents
        self.config = config or WorldConfig()
        self._rng = random.Random(self.config.seed)
        self.states: List[AgentState] = []
        self.price_history: List[float] = []
        self.last_price: Optional[float] = None
        self.step_count = 0
        self.history: List[StepInfo] = []
        self.reset()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        cfg = self.config
        self._rng = random.Random(cfg.seed)
        wells = self._draw_well_capacities()
        self.states = [
            AgentState(
                agent_id=i,
                name=agent.name,
                cash=cfg.initial_cash,
                water=cfg.initial_water,
                well_capacity=wells[i],
                storage_capacity=cfg.storage_capacity,
            )
            for i, agent in enumerate(self.agents)
        ]
        for agent in self.agents:
            agent.reset()
        self.price_history = []
        self.last_price = None
        self.step_count = 0
        self.history = []

    def _draw_well_capacities(self) -> List[float]:
        """Assign each agent a fixed well capacity spread around the mean.

        This makes some agents structural *producers* (big wells, chronic
        surplus) and others *consumers* (small wells, chronic deficit), which
        is what keeps trade -- and therefore price -- alive over an episode.
        The draw is deterministic given the config seed.
        """
        cfg = self.config
        spread = cfg.well_capacity_spread
        return [
            max(0.0, cfg.well_capacity * (1.0 + self._rng.uniform(-spread, spread)))
            for _ in self.agents
        ]

    @property
    def reference_price(self) -> float:
        return self.last_price if self.last_price is not None else self.config.initial_price

    def done(self) -> bool:
        if self.step_count >= self.config.n_steps:
            return True
        return not any(s.alive for s in self.states)

    # ------------------------------------------------------------------ #
    # the step
    # ------------------------------------------------------------------ #
    def step(self) -> StepInfo:
        cfg = self.config

        net_worth_before = {
            s.agent_id: s.net_worth(self.reference_price) for s in self.states
        }

        drought = self._rng.random() < cfg.drought_prob
        self._produce(drought)

        demands = {s.agent_id: self._draw_demand() for s in self.states}
        orders = self._collect_orders(demands)

        result = clear_market(orders)
        if result.clearing_price is not None:
            self._settle(result.trades)
            self.last_price = result.clearing_price
            self.price_history.append(result.clearing_price)

        self._consume(demands)
        self._settle_bankruptcies()

        mark = self.reference_price
        rewards = {}
        net_worth_after = {}
        for s in self.states:
            nw = s.net_worth(mark)
            net_worth_after[s.agent_id] = nw
            rewards[s.agent_id] = nw - net_worth_before[s.agent_id]

        self.step_count += 1
        info = StepInfo(
            step=self.step_count,
            clearing_price=result.clearing_price,
            volume=result.volume,
            reference_price=mark,
            drought=drought,
            rewards=rewards,
            net_worth=net_worth_after,
        )
        self.history.append(info)
        return info

    def run(self, verbose: bool = False) -> List[StepInfo]:
        """Run the episode to completion and return per-step info."""
        while not self.done():
            info = self.step()
            if verbose:
                price = (
                    f"{info.clearing_price:.3f}"
                    if info.clearing_price is not None
                    else "  -  "
                )
                drought = " DROUGHT" if info.drought else ""
                print(
                    f"step {info.step:4d} | price {price} | "
                    f"vol {info.volume:6.2f}{drought}"
                )
        return self.history

    # ------------------------------------------------------------------ #
    # phases
    # ------------------------------------------------------------------ #
    def _produce(self, drought: bool) -> None:
        cfg = self.config
        for s in self.states:
            if not s.alive:
                continue
            mult = max(0.0, self._rng.gauss(1.0, cfg.production_noise))
            if drought:
                mult *= cfg.drought_factor
            produced = s.well_capacity * mult
            s.water = min(s.storage_capacity, s.water + produced)

    def _draw_demand(self) -> float:
        cfg = self.config
        return max(0.0, self._rng.gauss(cfg.base_demand, cfg.base_demand * cfg.demand_noise))

    def _collect_orders(self, demands: Dict[int, float]) -> List[Order]:
        orders: List[Order] = []
        for agent, s in zip(self.agents, self.states):
            if not s.alive:
                continue
            obs = Observation(
                step=self.step_count,
                max_steps=self.config.n_steps,
                n_agents=len(self.states),
                cash=s.cash,
                water=s.water,
                well_capacity=s.well_capacity,
                storage_capacity=s.storage_capacity,
                demand=demands[s.agent_id],
                unit_value=self.config.consumption_value,
                last_price=self.last_price,
                price_history=list(self.price_history),
            )
            action = agent.act(obs)
            order = self._validate(action, s)
            if order is not None:
                orders.append(order)
        return orders

    def _validate(self, action: Action, s: AgentState) -> Optional[Order]:
        """Clip an action to what the agent can actually afford / hold."""
        if action is None or abs(action.quantity) <= _EPS or action.price <= 0:
            return None
        if action.quantity > 0:  # buy
            affordable = s.cash / action.price if action.price > 0 else 0.0
            qty = min(action.quantity, affordable, s.storage_capacity - s.water)
            if qty <= _EPS:
                return None
            return Order(agent_id=s.agent_id, side="buy", price=action.price, quantity=qty)
        # sell
        qty = min(-action.quantity, s.water)
        if qty <= _EPS:
            return None
        return Order(agent_id=s.agent_id, side="sell", price=action.price, quantity=qty)

    def _settle(self, trades: List[Trade]) -> None:
        by_id = {s.agent_id: s for s in self.states}
        for t in trades:
            s = by_id[t.agent_id]
            cost = t.price * t.quantity
            if t.side == "buy":
                s.cash -= cost
                s.water = min(s.storage_capacity, s.water + t.quantity)
                s.total_bought += t.quantity
            else:  # sell
                s.cash += cost
                s.water = max(0.0, s.water - t.quantity)
                s.total_sold += t.quantity

    def _consume(self, demands: Dict[int, float]) -> None:
        cfg = self.config
        for s in self.states:
            if not s.alive:
                continue
            need = demands[s.agent_id]
            consumed = min(s.water, need)
            s.water -= consumed
            s.total_consumed += consumed
            s.cash += cfg.consumption_value * consumed  # utility revenue
            shortfall = need - consumed
            if shortfall > _EPS:
                s.total_shortfall += shortfall
                s.cash -= cfg.unmet_penalty * shortfall

    def _settle_bankruptcies(self) -> None:
        if not self.config.allow_bankruptcy:
            return
        for s in self.states:
            if s.alive and s.cash < -_EPS:
                s.alive = False
                s.died_at = self.step_count + 1

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def leaderboard(self) -> List[AgentState]:
        """Agent states sorted by final net worth (living agents first)."""
        price = self.reference_price
        return sorted(
            self.states,
            key=lambda s: (s.alive, s.net_worth(price)),
            reverse=True,
        )

    def summary(self) -> str:
        price = self.reference_price
        lines = [
            f"Water Arena -- {self.step_count} steps, "
            f"final reference price {price:.3f}",
            f"{'rank':>4}  {'agent':<12}{'status':<10}"
            f"{'net worth':>12}{'cash':>10}{'water':>8}"
            f"{'bought':>9}{'sold':>8}{'thirst':>9}",
        ]
        for rank, s in enumerate(self.leaderboard(), start=1):
            status = "alive" if s.alive else f"died@{s.died_at}"
            lines.append(
                f"{rank:>4}  {s.name:<12}{status:<10}"
                f"{s.net_worth(price):>12.2f}{s.cash:>10.2f}{s.water:>8.2f}"
                f"{s.total_bought:>9.1f}{s.total_sold:>8.1f}{s.total_shortfall:>9.2f}"
            )
        return "\n".join(lines)
