"""Drive the pyramid RL env with a random policy, and compare to a baseline.

Shows the Gymnasium interface without needing an RL library: swap the random
policy for a trained one (SB3/RLlib/CleanRL) and it plugs straight in. We also
run the same node under a scripted CostPlus strategy on the same seed so you can
see what the seat earns without learning.

    python examples/rl_pyramid_rollout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig, PyramidWorld, build_pyramid
from water_arena.pyramid_env import PyramidTradingEnv
from water_arena.supply_agents import CostPlusTrader, default_supply_roster


def random_rollout(seed: int) -> float:
    import random

    rng = random.Random(seed)
    env = PyramidTradingEnv(config=PyramidConfig(n_layers=3, n_steps=300, seed=seed))
    env.reset(seed=seed)
    total = 0.0
    while True:
        action = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total, env.learner_node


def scripted_baseline(seed: int, node: int) -> float:
    cfg = PyramidConfig(n_layers=3, n_steps=300, seed=seed)
    topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
    roster = default_supply_roster(topo.trader_ids)
    roster[node] = CostPlusTrader(name="Baseline")  # same seat, scripted
    world = PyramidWorld(roster, cfg)
    world.run()
    return world.states[node].total_profit


def main() -> None:
    seed = 0
    total, node = random_rollout(seed)
    baseline = scripted_baseline(seed, node)
    print(f"controlled node: {node}")
    print(f"random policy   total reward (profit): {total:8.2f}")
    print(f"cost-plus baseline in the same seat:   {baseline:8.2f}")
    print("\n(a trained policy should beat the random rollout, and ideally the baseline)")


if __name__ == "__main__":
    main()
