"""Train ALL pyramid nodes at once with parameter-sharing PPO.

Uses SuperSuit to flatten the PettingZoo ParallelEnv into a vectorised env that
Stable-Baselines3 can train: one shared policy controls every trader. We then
compare the learned policy against the scripted roster on held-out worlds, on
two metrics -- total trader profit and buyer fill rate.

    pip install stable-baselines3 supersuit pettingzoo
    python examples/marl_train_sb3.py --timesteps 200000

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

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
N_STEPS = 300


def _cfg(seed=0):
    return PyramidConfig(n_layers=3, n_steps=N_STEPS, seed=seed, allow_bankruptcy=False)


def _make_vec_env(n_copies=2):
    import supersuit as ss

    env = PyramidParallelEnv(config=_cfg())
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # n_copies * 9 agents concatenated sub-envs; kept small so PPO gets many
    # gradient updates within a modest step budget.
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


class LearningCurve:
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
    parser.add_argument("--timesteps", type=int, default=200000)
    args = parser.parse_args()

    try:
        import supersuit  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import VecMonitor
    except ImportError:
        print("needs stable-baselines3, supersuit, pettingzoo:")
        print("  pip install stable-baselines3 supersuit pettingzoo")
        return

    # Seed globally rather than via PPO(seed=...): SB3's env-seeding path calls
    # VecEnv.seed(), which SuperSuit's ConcatVecEnv no longer exposes.
    set_random_seed(0)
    env = VecMonitor(_make_vec_env())  # 2 copies * 9 agents = 18 sub-envs
    print(f"training a shared PPO policy over all 9 nodes for {args.timesteps} steps...")
    curve = LearningCurve()
    # n_steps=512 over 18 sub-envs -> ~9k-sample rollouts, so many updates fit
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=256, verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=curve.callback)

    eval_seeds = list(range(1000, 1020))
    tp, tf = evaluate_shared_policy(model, eval_seeds)
    sp, sf = evaluate_scripted(eval_seeds)
    print("\nOver 20 held-out worlds (all nodes controlled):")
    print(f"  {'':<18}{'total profit':>14}{'buyer fill %':>14}")
    print(f"  {'scripted roster':<18}{sp:>14.1f}{sf:>14.1f}")
    print(f"  {'shared PPO policy':<18}{tp:>14.1f}{tf:>14.1f}")

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
    ax.set_xlabel("training steps (summed over all agents)")
    ax.set_ylabel("mean per-agent episode profit")
    ax.set_title("Parameter-sharing PPO — all pyramid nodes learning at once")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "marl_learning_curve.png")
    fig.savefig(out, dpi=110)
    print(f"\nlearning curve written to {out}")


if __name__ == "__main__":
    main()
