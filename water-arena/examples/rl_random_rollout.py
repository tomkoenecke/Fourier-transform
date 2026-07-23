"""Drive the Gymnasium-style env with random actions.

This shows the RL interface without needing an RL library installed -- swap the
random policy for a trained one (e.g. Stable-Baselines3) and it plugs straight in.

    python examples/rl_random_rollout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.env import WaterTradingEnv
from water_arena.world import WorldConfig


def main() -> None:
    import random

    env = WaterTradingEnv(config=WorldConfig(n_steps=200, seed=0))
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    steps = 0
    while True:
        action = [random.uniform(-1, 1), random.uniform(-1, 1)]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    print(f"episode finished after {steps} steps")
    print(f"learner total reward (net-worth change): {total_reward:.2f}")
    env.render()


if __name__ == "__main__":
    main()
