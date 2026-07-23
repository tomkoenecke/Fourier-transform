"""Random rollout of the multi-agent pyramid env (every node acts at once).

Demonstrates the PettingZoo ParallelEnv API without needing an RL library.

    python examples/marl_rollout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig
from water_arena.pyramid_marl import PyramidParallelEnv


def main() -> None:
    import random

    rng = random.Random(0)
    env = PyramidParallelEnv(config=PyramidConfig(n_layers=3, n_steps=300, seed=0))
    observations, _ = env.reset(seed=0)

    totals = {a: 0.0 for a in env.possible_agents}
    while env.agents:
        actions = {a: [rng.uniform(-1, 1), rng.uniform(-1, 1)] for a in env.agents}
        observations, rewards, terms, truncs, _ = env.step(actions)
        for a, r in rewards.items():
            totals[a] += r

    print("Multi-agent random rollout — total profit per node:")
    for a in sorted(totals, key=totals.get, reverse=True):
        print(f"  {a:<7} {totals[a]:8.2f}")
    print(f"\n  system total: {sum(totals.values()):.2f}")


if __name__ == "__main__":
    main()
