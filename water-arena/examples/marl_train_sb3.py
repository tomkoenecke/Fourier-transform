"""Train ALL pyramid nodes at once with parameter-sharing PPO.

Uses SuperSuit to flatten the PettingZoo ParallelEnv into a vectorised env that
Stable-Baselines3 can train: one shared policy controls every trader. We then
compare the learned policy against the scripted roster on held-out worlds.

Multi-agent training here is *non-stationary* -- every agent's environment keeps
shifting as the others learn -- so a single shared PPO tends to peak and then
collapse. We evaluate on held-out worlds during training and keep the **best**
checkpoint (early stopping), which is both more honest and better practice.

    pip install stable-baselines3 supersuit pettingzoo
    python examples/marl_train_sb3.py --timesteps 500000

Bankruptcy is disabled during training so the agent set is constant (what
parameter-sharing expects).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from water_arena.pyramid import PyramidConfig, PyramidWorld, build_pyramid
from water_arena.pyramid_marl import PyramidParallelEnv
from water_arena.supply_agents import default_supply_roster

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "fig")
MODELS_DIR = os.path.join(ROOT, "models")
N_STEPS = 300


def _cfg(seed=0):
    return PyramidConfig(n_layers=3, n_steps=N_STEPS, seed=seed, allow_bankruptcy=False)


def _make_vec_env(n_copies=2):
    import supersuit as ss

    env = PyramidParallelEnv(config=_cfg())
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, n_copies, num_cpus=1, base_class="stable_baselines3")
    return env


def evaluate_shared_policy(model, eval_seeds):
    """Run the shared policy on every node; report total profit and fill rate."""
    profits, fills = [], []
    for s in eval_seeds:
        env = PyramidParallelEnv(config=_cfg(seed=s))
        obs, _ = env.reset(seed=s)
        totals = {a: 0.0 for a in env.possible_agents}
        while env.agents:
            actions = {a: model.predict(obs[a], deterministic=True)[0] for a in env.agents}
            obs, rewards, terms, truncs, _ = env.step(actions)
            for a, r in rewards.items():
                totals[a] += r
        profits.append(sum(totals.values()))
        served = sum(h.buyer_filled for h in env.world.history)
        need = sum(h.buyer_need for h in env.world.history)
        fills.append(100.0 * served / need if need else 0.0)
    return sum(profits) / len(profits), sum(fills) / len(fills)


def evaluate_scripted(eval_seeds):
    profits, fills = [], []
    for s in eval_seeds:
        cfg = _cfg(seed=s)
        topo = build_pyramid(cfg.n_layers, cfg.pipeline_capacity)
        world = PyramidWorld(default_supply_roster(topo.trader_ids), cfg)
        world.run()
        profits.append(sum(world.states[n].total_profit for n in topo.trader_ids))
        served = sum(h.buyer_filled for h in world.history)
        need = sum(h.buyer_need for h in world.history)
        fills.append(100.0 * served / need if need else 0.0)
    return sum(profits) / len(profits), sum(fills) / len(fills)


def _best_checkpoint_callback(eval_every, eval_seeds, best_path, curve):
    from stable_baselines3.common.callbacks import BaseCallback

    class _CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = float("-inf")
            self._last = 0

        def _on_step(self):
            if self.num_timesteps - self._last >= eval_every:
                self._last = self.num_timesteps
                profit, fill = evaluate_shared_policy(self.model, eval_seeds)
                curve.x.append(self.num_timesteps)
                curve.y.append(profit)
                if profit > self.best:
                    self.best = profit
                    self.model.save(best_path)
            return True

    return _CB()


class Curve:
    def __init__(self):
        self.x, self.y = [], []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    try:
        import supersuit  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
    except ImportError:
        print("needs stable-baselines3, supersuit, pettingzoo:")
        print("  pip install stable-baselines3 supersuit pettingzoo")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_path = os.path.join(MODELS_DIR, "marl_ppo_best")

    # Seed globally rather than PPO(seed=...): SB3's env-seeding path calls
    # VecEnv.seed(), which SuperSuit's ConcatVecEnv no longer exposes.
    set_random_seed(0)
    env = _make_vec_env()  # 2 copies * 9 agents = 18 sub-envs
    print(f"training a shared PPO policy over all 9 nodes for {args.timesteps} steps...")

    curve = Curve()
    eval_seeds = list(range(2000, 2005))  # held-out worlds for checkpoint selection
    cb = _best_checkpoint_callback(max(10000, args.timesteps // 25), eval_seeds, best_path, curve)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=256, verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=cb)

    # reload the best checkpoint (early stopping) for the final comparison
    best = PPO.load(best_path)
    test_seeds = list(range(1000, 1020))
    tp, tf = evaluate_shared_policy(best, test_seeds)
    sp, sf = evaluate_scripted(test_seeds)
    print("\nOver 20 held-out worlds (all nodes controlled):")
    print(f"  {'':<22}{'total profit':>14}{'buyer fill %':>14}")
    print(f"  {'scripted roster':<22}{sp:>14.1f}{sf:>14.1f}")
    print(f"  {'shared PPO (best ckpt)':<22}{tp:>14.1f}{tf:>14.1f}")

    _plot(curve)


def _plot(curve) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not curve.x:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve.x, curve.y, color="tab:green")
    best_i = max(range(len(curve.y)), key=lambda i: curve.y[i])
    ax.scatter([curve.x[best_i]], [curve.y[best_i]], color="tab:red", zorder=5, label="best (kept)")
    ax.set_xlabel("training steps (summed over all agents)")
    ax.set_ylabel("held-out total profit")
    ax.set_title("Parameter-sharing PPO — peaks then destabilises (early stopping keeps the peak)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "marl_learning_curve.png")
    fig.savefig(out, dpi=110)
    print(f"\nlearning curve written to {out}")


if __name__ == "__main__":
    main()
