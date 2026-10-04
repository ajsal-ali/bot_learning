#!/usr/bin/env python3
"""Train GO-BDX to walk: MJX physics + Brax PPO, everything on the GPU.

    python -m bdx_mjx.train --preset gpu         # Linux + NVIDIA, >= 8 GB VRAM
    python -m bdx_mjx.train --preset gpu_small   # Linux + NVIDIA, 4-6 GB VRAM
    python -m bdx_mjx.train --preset cpu         # no GPU: ~30 min learning check
    python -m bdx_mjx.train --preset cpu_test    # smoke test anywhere, minutes

    # resume from a checkpoint (step counter restarts, weights + normalizer kept)
    python -m bdx_mjx.train --preset gpu --resume runs/<run>/checkpoints/latest.pkl

Each run writes runs/<name>/:
    config.json        env + PPO config (play.py rebuilds everything from it)
    progress.csv/png   eval reward, episode length, reward terms vs. env steps
    checkpoints/       latest.pkl (+ one per eval) - Brax params, for --resume
    policy.npz         NumPy-only policy for play.py / the real robot
"""

import argparse
import csv
import datetime
import functools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRESETS = {
    "gpu": dict(
        num_timesteps=200_000_000, num_envs=8192, batch_size=256,
        num_minibatches=32, num_evals=21, num_eval_envs=128),
    "gpu_small": dict(
        num_timesteps=150_000_000, num_envs=2048, batch_size=256,
        num_minibatches=8, num_evals=16, num_eval_envs=64),
    "cpu": dict(  # learning sanity check without a GPU (~1k steps/s)
        num_timesteps=2_000_000, num_envs=128, batch_size=128,
        num_minibatches=4, num_evals=9, num_eval_envs=16, episode_length=500),
    "cpu_test": dict(
        num_timesteps=20_000, num_envs=16, batch_size=16, num_minibatches=4,
        num_evals=3, num_eval_envs=8, episode_length=200),
}

PPO_DEFAULTS = dict(
    episode_length=1000,
    unroll_length=20,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=5e-3,
    clipping_epsilon=0.2,
    max_grad_norm=1.0,
    reward_scaling=1.0,
    normalize_observations=True,
    num_resets_per_eval=1,
    action_repeat=1,
)

NETWORK = dict(
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--preset", choices=PRESETS, default="gpu")
  p.add_argument("--num-timesteps", type=int, help="override preset")
  p.add_argument("--num-envs", type=int, help="override preset")
  p.add_argument("--impl", choices=["jax", "warp"], default="jax",
                 help="MJX backend. 'warp' (MuJoCo Warp) can be faster on recent NVIDIA GPUs.")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default=None)
  p.add_argument("--runs-dir", default="runs")
  p.add_argument("--resume", default=None, help="path to a checkpoints/*.pkl")
  return p.parse_args()


def main():
  args = parse_args()

  # Must be set before JAX is imported.
  os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
  os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

  import jax
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from brax.training.agents.ppo import networks as ppo_networks
  from brax.training.agents.ppo import train as ppo
  from mujoco_playground._src import wrapper

  from bdx_mjx import joystick
  from bdx_mjx import policy as policy_lib
  from bdx_mjx.randomize import domain_randomize

  backend = jax.default_backend()
  print(f"JAX {jax.__version__} backend={backend} devices={jax.devices()}")
  if backend == "cpu" and not args.preset.startswith("cpu"):
    print("\n!! JAX is running on CPU. Native Windows JAX has no GPU support -\n"
          "!! train on Linux or WSL2 with jax[cuda12]. This will be very slow.\n")

  ppo_cfg = dict(PPO_DEFAULTS)
  ppo_cfg.update(PRESETS[args.preset])
  if args.num_timesteps:
    ppo_cfg["num_timesteps"] = args.num_timesteps
  if args.num_envs:
    ppo_cfg["num_envs"] = args.num_envs
  assert (ppo_cfg["batch_size"] * ppo_cfg["num_minibatches"]) % ppo_cfg["num_envs"] == 0, \
      "batch_size * num_minibatches must be a multiple of num_envs"

  env_cfg = joystick.default_config()
  env_cfg.impl = args.impl
  env_cfg.episode_length = ppo_cfg["episode_length"]
  env_cfg.naconmax = 8 * ppo_cfg["num_envs"]  # 2 feet x 4 box-plane contacts

  run_name = args.run_name or f"{args.preset}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
  run_dir = os.path.join(args.runs_dir, run_name)
  ckpt_dir = os.path.join(run_dir, "checkpoints")
  os.makedirs(ckpt_dir, exist_ok=True)
  with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump({"env": env_cfg.to_dict(), "ppo": ppo_cfg, "network": NETWORK,
               "preset": args.preset, "seed": args.seed, "resume": args.resume}, f, indent=2)
  print(f"run dir: {run_dir}")
  print("ppo:", json.dumps(ppo_cfg))

  env = joystick.Joystick(config=env_cfg)
  eval_env = joystick.Joystick(config=env_cfg)

  restore_params = None
  if args.resume:
    with open(args.resume, "rb") as f:
      restore_params = pickle.load(f)
    print(f"resuming from {args.resume}")

  # ---------------------------------------------------------------- logging
  csv_path = os.path.join(run_dir, "progress.csv")
  history = []
  t_start = time.time()
  last = {"t": t_start, "step": 0}

  def progress(step, metrics):
    now = time.time()
    sps = (step - last["step"]) / max(now - last["t"], 1e-9)
    last.update(t=now, step=step)
    row = {
        "step": step,
        "minutes": (now - t_start) / 60,
        "steps_per_s": sps,
        "reward": float(metrics.get("eval/episode_reward", float("nan"))),
        "reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
        "episode_length": float(metrics.get("eval/avg_episode_length", float("nan"))),
    }
    prefix = "eval/episode_reward/"
    terms = {k[len(prefix):]: float(v) for k, v in metrics.items()
             if k.startswith(prefix) and not k.endswith("_std")}
    row.update(terms)
    history.append(row)

    print(f"[{step:>12,} steps | {row['minutes']:6.1f} min | {sps:>9,.0f} steps/s] "
          f"reward {row['reward']:8.2f} +- {row['reward_std']:6.2f}   "
          f"episode length {row['episode_length']:6.1f}")
    if terms:
      top = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:8]
      print("    " + "  ".join(f"{k}={v:.2f}" for k, v in top))

    with open(csv_path, "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=list(history[-1].keys()))
      w.writeheader()
      for r in history:
        w.writerow({k: r.get(k, "") for k in history[-1].keys()})

    if len(history) >= 2:
      steps = [r["step"] for r in history]
      fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
      ax[0].plot(steps, [r["reward"] for r in history], "o-")
      ax[0].set_title("eval episode reward")
      ax[1].plot(steps, [r["episode_length"] for r in history], "o-", color="tab:purple")
      ax[1].axhline(env_cfg.episode_length, ls="--", c="gray")
      ax[1].set_title("eval episode length (max = no falls)")
      for k in terms:
        ax[2].plot(steps, [r.get(k, float("nan")) for r in history], label=k)
      ax[2].set_title("reward terms (per episode)")
      ax[2].legend(fontsize=6, ncol=2)
      for a in ax:
        a.set_xlabel("env steps")
        a.grid(alpha=0.3)
      fig.tight_layout()
      fig.savefig(os.path.join(run_dir, "progress.png"), dpi=110)
      plt.close(fig)

  def save_params(step, make_policy, params):
    del make_policy
    params = jax.device_get(params)
    for name in (f"params_{step:012d}.pkl", "latest.pkl"):
      with open(os.path.join(ckpt_dir, name), "wb") as f:
        pickle.dump(params, f)
    policy_lib.export_npz(params, env, os.path.join(run_dir, "policy.npz"))

  network_factory = functools.partial(ppo_networks.make_ppo_networks, **NETWORK)

  train_fn = functools.partial(
      ppo.train,
      **ppo_cfg,
      network_factory=network_factory,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      randomization_fn=domain_randomize,
      progress_fn=progress,
      policy_params_fn=save_params,
      restore_params=restore_params,
      seed=args.seed,
  )

  print("compiling (first iteration takes a few minutes)...")
  try:
    train_fn(environment=env, eval_env=eval_env)
  except KeyboardInterrupt:
    print("\ninterrupted - last checkpoint is in", ckpt_dir)

  print(f"\ndone in {(time.time() - t_start) / 60:.1f} min. Watch it:")
  print(f"  python -m bdx_mjx.play {run_dir}")


if __name__ == "__main__":
  main()
