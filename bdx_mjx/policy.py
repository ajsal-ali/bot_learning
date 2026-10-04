"""NumPy-only walking policy: what runs in play.py and on the real robot.

train.py exports the actor network + observation normalizer to policy.npz.
WalkController rebuilds the exact actor observation the env uses
(joystick.Joystick._get_obs "state") from raw sensor readings, so the same
class can drive the simulator or the hardware. No JAX needed at runtime.
"""

import os
from typing import Mapping

import numpy as np


def _dense_layers(tree):
  """[(name, kernel, bias)] from a flax params tree, in layer order."""
  layers = []

  def walk(node, name):
    if isinstance(node, Mapping):
      if "kernel" in node:
        layers.append((name, np.asarray(node["kernel"]), np.asarray(node["bias"])))
        return
      for k in node:
        walk(node[k], k)

  walk(tree, "")
  layers.sort(key=lambda t: int(t[0].rsplit("_", 1)[-1]))  # hidden_0, hidden_1, ...
  return layers


def export_npz(params, env, path):
  """Saves Brax PPO params (normalizer, policy, value) as a plain .npz."""
  normalizer, policy_params = params[0], params[1]
  layers = _dense_layers(policy_params)
  cfg = env._config  # pylint: disable=protected-access
  arrays = {
      "obs_mean": np.asarray(normalizer.mean["state"], np.float32),
      "obs_std": np.asarray(normalizer.std["state"], np.float32),
      "num_layers": np.array(len(layers)),
      "action_size": np.array(env.action_size),
      "default_pose": np.asarray(env._default_pose, np.float32),  # pylint: disable=protected-access
      "joint_lower": np.asarray(env._lowers, np.float32),  # pylint: disable=protected-access
      "joint_upper": np.asarray(env._uppers, np.float32),  # pylint: disable=protected-access
      "action_scale": np.array(cfg.action_scale),
      "ctrl_dt": np.array(cfg.ctrl_dt),
      "gait_freq": np.array(np.mean(cfg.gait_config.freq_range)),
  }
  for i, (_, w, b) in enumerate(layers):
    arrays[f"w{i}"] = w.astype(np.float32)
    arrays[f"b{i}"] = b.astype(np.float32)
  tmp = path + ".tmp.npz"
  np.savez(tmp, **arrays)
  os.replace(tmp, path)  # atomic: play.py may be reading it


class NumpyPolicy:
  """Deterministic actor: normalize -> MLP (swish) -> tanh(mean)."""

  def __init__(self, npz_path: str):
    z = np.load(npz_path)
    self.meta = {k: z[k] for k in z.files if not k.startswith(("w", "b"))}
    self.mean, self.std = z["obs_mean"], z["obs_std"]
    self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(int(z["num_layers"]))]
    self.action_size = int(z["action_size"])

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = (np.asarray(obs, np.float32) - self.mean) / self.std
    for i, (w, b) in enumerate(self.layers):
      x = x @ w + b
      if i < len(self.layers) - 1:
        x = x / (1.0 + np.exp(-x))  # swish
    return np.tanh(x[..., :self.action_size])


class WalkController:
  """Runs the policy at ctrl_dt. Call step() once per control tick."""

  def __init__(self, npz_path: str):
    self.policy = NumpyPolicy(npz_path)
    m = self.policy.meta
    self.default_pose = m["default_pose"]
    self.lower, self.upper = m["joint_lower"], m["joint_upper"]
    self.action_scale = float(m["action_scale"])
    self.dt = float(m["ctrl_dt"])
    self.phase_dt = 2 * np.pi * self.dt * float(m["gait_freq"])
    self.command = np.zeros(3, np.float32)  # vx [m/s], vy [m/s], yaw rate [rad/s]
    self.reset()

  def reset(self):
    self.last_act = np.zeros(self.policy.action_size, np.float32)
    self.phase = np.array([0.0, np.pi])  # left, right

  def observation(self, gyro, gravity, joint_pos, joint_vel) -> np.ndarray:
    """gyro: IMU angular velocity (IMU frame: x fwd, y left, z up) [rad/s].
    gravity: unit gravity direction in the IMU frame ((0,0,-1) when upright).
    joint_pos / joint_vel: the 10 leg joints in constants.LEG_JOINTS order."""
    return np.concatenate([
        gyro, gravity, self.command,
        np.asarray(joint_pos) - self.default_pose, joint_vel,
        self.last_act, np.cos(self.phase), np.sin(self.phase),
    ]).astype(np.float32)

  def step(self, gyro, gravity, joint_pos, joint_vel) -> np.ndarray:
    """Returns joint position targets for the 10 leg motors."""
    action = self.policy(self.observation(gyro, gravity, joint_pos, joint_vel))
    self.last_act = action.astype(np.float32)
    self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
    targets = self.default_pose + action * self.action_scale
    return np.clip(targets, self.lower, self.upper)
