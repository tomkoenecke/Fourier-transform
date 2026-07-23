"""Tests for the uniform-price call-auction clearing."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.market import Order, clear_market


def test_no_cross_no_trade():
    # highest bid below lowest ask -> nothing trades
    orders = [
        Order(0, "buy", price=1.0, quantity=5),
        Order(1, "sell", price=2.0, quantity=5),
    ]
    result = clear_market(orders)
    assert result.clearing_price is None
    assert result.volume == 0
    assert result.trades == []


def test_empty_book():
    assert clear_market([]).clearing_price is None
    assert clear_market([Order(0, "buy", 1.0, 5)]).clearing_price is None


def test_simple_cross():
    orders = [
        Order(0, "buy", price=2.0, quantity=5),
        Order(1, "sell", price=1.0, quantity=5),
    ]
    result = clear_market(orders)
    assert result.clearing_price is not None
    assert 1.0 <= result.clearing_price <= 2.0
    assert result.volume == 5


def test_quantity_conservation():
    orders = [
        Order(0, "buy", 3.0, 10),
        Order(1, "buy", 2.5, 4),
        Order(2, "sell", 2.0, 6),
        Order(3, "sell", 1.5, 5),
    ]
    result = clear_market(orders)
    bought = sum(t.quantity for t in result.trades if t.side == "buy")
    sold = sum(t.quantity for t in result.trades if t.side == "sell")
    assert abs(bought - sold) < 1e-9
    assert abs(bought - result.volume) < 1e-9


def test_price_priority_fill():
    # demand (14) exceeds supply (5): only the highest bid should fill
    orders = [
        Order(0, "buy", 3.0, 10),
        Order(1, "buy", 2.9, 4),
        Order(2, "sell", 1.0, 5),
    ]
    result = clear_market(orders)
    assert abs(result.volume - 5) < 1e-9
    buy_fills = {t.agent_id: t.quantity for t in result.trades if t.side == "buy"}
    assert abs(buy_fills.get(0, 0) - 5) < 1e-9  # best bid filled first
    assert buy_fills.get(1, 0) < 1e-9  # worse bid gets nothing


def test_partial_marginal_fill():
    orders = [
        Order(0, "buy", 3.0, 7),
        Order(1, "sell", 1.0, 4),
        Order(2, "sell", 1.2, 4),
    ]
    result = clear_market(orders)
    # supply within the crossing region is 8, demand 7 -> 7 trades
    assert abs(result.volume - 7) < 1e-9
    sold = sum(t.quantity for t in result.trades if t.side == "sell")
    assert abs(sold - 7) < 1e-9
