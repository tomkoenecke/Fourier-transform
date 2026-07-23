"""Train a PPO policy on the pyramid RL env with Stable-Baselines3.

Trains an agent to run one trader node, then compares it against a random
policy and a scripted cost-plus baseline in the *same* seat on the *same*
evaluation worlds. Writes a learning curve to fig/sb3_learning_curve.png.

    pip install stable-baselines3
    python examples/rl_train_sb3.py --timesteps 60000

Everything is pure-CPU and small; 60k steps trains in a couple of minutes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig, PyramidWorld, build_pyramid
from water_arena.supply_agents import CostPlusTrader, default_supply_roster

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
N_STEPS = 300


def _make_env(seed=0):
    from stable_baselines3.common.monitor import Monitor

    from water_arena.pyramid_env import PyramidTradingEnv

    env = PyramidTradingEnv(config=PyramidConfig(n_layers=3, n_steps=N_STEPS, seed=seed))
    return Monitor(env)


def evaluate_policy_mean(model, learner_node, eval_seeds):
    """Average total episode reward of a trained policy over fixed worlds."""
    from water_arena.pyramid_env import PyramidTradingEnv

    env = PyramidTradingEnv(config=PyramidConfig(n_layers=3, n_steps=N_STEPS))
    totals = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        totals.append(total)
    return sum(totals) / len(totals)


def evaluate_random_mean(learner_node, eval_seeds):
    import random

    from water_arena.pyramid_env import PyramidTradingEnv

    env = PyramidTradingEnv(config=PyramidConfig(n_layers=3, n_steps=N_STEPS))
    rng = random.Random(123)
    totals = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        done = False
        total = 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step([rng.uniform(-1, 1), rng.uniform(-1, 1)])
            total += r
            done = terminated or truncated
        totals.append(total)
    return sum(totals) / len(totals)


def evaluate_costplus_mean(learner_node, eval_seeds):
    """Cost-plus strategy in the same seat, on the same evaluation worlds."""
    totals = []
    for s in eval_seeds:
        cfg = PyramidConfig(n_layers=3, n_steps=N_STEPS, seed=s)
        topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
        roster = default_supply_roster(topo.trader_ids)
        roster[learner_node] = CostPlusTrader(name="Baseline")
        world = PyramidWorld(roster, cfg)
        world.run()
        totals.append(world.states[learner_node].total_profit)
    return sum(totals) / len(totals)


class LearningCurve:
    """Records the rolling mean episode reward across training."""

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback

        curve = self

        class _CB(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                buf = self.model.ep_info_buffer
                if buf:
                    curve.x.append(self.num_timesteps)
                    curve.y.append(sum(e["r"] for e in buf) / len(buf))

        self.x, self.y = [], []
        self.callback = _CB()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=60000)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("this example needs stable-baselines3:  pip install stable-baselines3")
        return

    env = _make_env(seed=0)
    learner_node = env.env.learner_node  # unwrap Monitor
    print(f"training PPO to control node {learner_node} for {args.timesteps} steps...")

    curve = LearningCurve()
    model = PPO("MlpPolicy", env, seed=0, verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=curve.callback)

    eval_seeds = list(range(1000, 1030))  # held-out worlds, same for all policies
    trained = evaluate_policy_mean(model, learner_node, eval_seeds)
    rnd = evaluate_random_mean(learner_node, eval_seeds)
    base = evaluate_costplus_mean(learner_node, eval_seeds)

    print("\nMean episode profit over 30 held-out worlds:")
    print(f"  random policy     {rnd:8.2f}")
    print(f"  cost-plus baseline{base:8.2f}")
    print(f"  trained PPO       {trained:8.2f}")

    _plot(curve)


def _plot(curve) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib for the learning-curve figure)")
        return
    if not curve.x:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve.x, curve.y, color="tab:blue")
    ax.set_xlabel("training steps")
    ax.set_ylabel("mean episode profit")
    ax.set_title("PPO learning curve — pyramid trader")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "sb3_learning_curve.png")
    fig.savefig(out, dpi=110)
    print(f"\nlearning curve written to {out}")


if __name__ == "__main__":
    main()
