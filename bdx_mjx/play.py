#!/usr/bin/env python3
"""Watch or record a trained GO-BDX policy in plain MuJoCo, at real-time (1x).

Uses the NumPy policy (policy.npz) and the same sensor -> observation code as
the real robot would (policy.WalkController), so no JAX/GPU is needed.

    python -m bdx_mjx.play runs/<run>                      # interactive viewer
    python -m bdx_mjx.play runs/<run> --cmd 0.3 0 0        # start walking forward
    python -m bdx_mjx.play runs/<run> --video walk.gif --seconds 10 --cmd 0.3 0 0
    python -m bdx_mjx.play runs/<run> --push-test          # push-recovery survival table

Viewer keys:  Up/Down = forward speed   Left/Right = turn rate
              PageUp/PageDown = sideways speed   Home = stop   End = reset robot
              Insert = push the torso (--push-force N for 0.15 s, random direction)
              (or MuJoCo's own: double-click the body, then Ctrl + right-drag)
"""

import argparse
import os
import shutil
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdx_mjx import constants as consts  # pylint: disable=wrong-import-position
from bdx_mjx.policy import WalkController  # pylint: disable=wrong-import-position

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PAGE_UP, KEY_PAGE_DOWN, KEY_HOME, KEY_END = 266, 267, 268, 269
KEY_INSERT = 260
CMD_LIMITS = np.array([[-0.3, 0.5], [-0.2, 0.2], [-0.8, 0.8]])  # match training


class Sim:
  """Plain-MuJoCo robot + controller, one control tick at a time."""

  def __init__(self, policy_path):
    self.model = mujoco.MjModel.from_xml_path(consts.VIEW_XML)
    self.data = mujoco.MjData(self.model)
    self.ctrl = WalkController(policy_path)
    self.n_substeps = int(round(self.ctrl.dt / self.model.opt.timestep))
    self.imu = self.model.site("imu").id
    self.gyro = self.model.sensor(consts.GYRO_SENSOR)
    self.up = self.model.sensor(consts.UPVECTOR_SENSOR)
    self.linvel = self.model.sensor(consts.LOCAL_LINVEL_SENSOR)
    self.base = self.model.body(consts.ROOT_BODY).id
    self.falls = 0
    self.reset()

  def reset(self):
    mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.keyframe("home").id)
    mujoco.mj_forward(self.model, self.data)
    self.ctrl.reset()
    self.push_force = np.zeros(2)
    self.push_ticks = 0

  def push(self, force_xy, seconds):
    """Horizontal force [N] on the torso CoM for `seconds` (same as training)."""
    self.push_force = np.asarray(force_xy, float)
    self.push_ticks = int(round(seconds / self.ctrl.dt))

  def tick(self):
    d = self.data
    gravity = d.site_xmat[self.imu].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    d.ctrl[:] = self.ctrl.step(d.sensordata[self.gyro.adr[0]:self.gyro.adr[0] + 3],
                               gravity, d.qpos[7:], d.qvel[6:])
    d.xfrc_applied[self.base, :2] = self.push_force if self.push_ticks > 0 else 0.0
    self.push_ticks = max(self.push_ticks - 1, 0)
    for _ in range(self.n_substeps):
      mujoco.mj_step(self.model, d)
    if self.fallen():
      self.falls += 1
      self.reset()
      return False
    return True

  def fallen(self):
    """Same termination test as training."""
    return self.data.sensordata[self.up.adr[0] + 2] < 0.6 or self.data.qpos[2] < 0.18

  def local_velocity(self):
    a = self.linvel.adr[0]
    return self.data.sensordata[a:a + 3].copy()


def run_viewer(sim, args):
  import mujoco.viewer

  def on_key(key):
    c = sim.ctrl.command
    if key == KEY_UP: c[0] += 0.05
    elif key == KEY_DOWN: c[0] -= 0.05
    elif key == KEY_LEFT: c[2] += 0.1
    elif key == KEY_RIGHT: c[2] -= 0.1
    elif key == KEY_PAGE_UP: c[1] += 0.05
    elif key == KEY_PAGE_DOWN: c[1] -= 0.05
    elif key == KEY_HOME: c[:] = 0
    elif key == KEY_END: sim.reset()
    elif key == KEY_INSERT:
      a = np.random.uniform(0, 2 * np.pi)
      sim.push(args.push_force * np.array([np.cos(a), np.sin(a)]), 0.15)
      print(f"push {args.push_force:.0f} N towards {np.degrees(a):.0f} deg")
      return
    else: return
    c[:] = np.clip(c, CMD_LIMITS[:, 0], CMD_LIMITS[:, 1])
    print(f"command vx={c[0]:+.2f} m/s  vy={c[1]:+.2f} m/s  yaw={c[2]:+.2f} rad/s")

  with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = sim.model.body(consts.ROOT_BODY).id
    v.cam.distance, v.cam.elevation = 1.4, -15
    next_t = time.perf_counter()
    while v.is_running():
      with v.lock():
        sim.tick()
      v.sync()
      next_t += sim.ctrl.dt / args.speed  # real-time pacing (speed=1 -> 1x)
      time.sleep(max(0.0, next_t - time.perf_counter()))


def run_video(sim, args):
  r = mujoco.Renderer(sim.model, height=480, width=640)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = sim.model.body(consts.ROOT_BODY).id
  cam.distance, cam.elevation, cam.azimuth = 1.4, -15, 120
  frames, vels = [], []
  steps = int(args.seconds / sim.ctrl.dt)
  for i in range(steps):
    sim.tick()
    vels.append(sim.local_velocity())
    if i % 2 == 0:  # 25 fps
      r.update_scene(sim.data, cam)
      frames.append(r.render())
  vels = np.array(vels[steps // 4:])  # skip the start-up transient
  print(f"command {sim.ctrl.command}, measured mean vx={vels[:, 0].mean():+.3f} "
        f"vy={vels[:, 1].mean():+.3f} m/s, falls={sim.falls}")
  out = args.video
  if out.endswith(".mp4") and shutil.which("ffmpeg"):
    import mediapy
    mediapy.write_video(out, frames, fps=25)
  else:
    if out.endswith(".mp4"):
      out = out[:-4] + ".gif"
      print("ffmpeg not found - writing a GIF instead")
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=40, loop=0)
  print(f"wrote {out}")


def run_push_test(sim, args):
  """Survival rate vs push force, standing and walking, 8 directions each.

  Each trial: 2 s to settle into the gait, push for 0.2 s at a heading-relative
  direction, then 3 s to recover. A trial fails if the robot falls (same test
  as training termination) at any point.
  """
  forces = [20, 30, 40, 50, 60, 70]
  commands = {"standing": [0.0, 0.0, 0.0], "walking 0.3 m/s": [0.3, 0.0, 0.0]}
  dirs = np.radians(np.arange(0, 360, 45))  # 0 = forward, 90 = left
  dt = sim.ctrl.dt
  print(f"push duration 0.2 s, 8 directions per cell. Training max: 55 N.")
  print(f"{'':18s}" + "".join(f"{f:>7d} N" for f in forces))
  for label, cmd in commands.items():
    row = []
    for force in forces:
      ok = 0
      for ang in dirs:
        sim.reset()
        sim.falls = 0
        sim.ctrl.command[:] = cmd
        survived = all(sim.tick() for _ in range(int(2.0 / dt)))
        if survived:
          # push direction is relative to the robot's current heading
          fwd = sim.data.site_xmat[sim.imu].reshape(3, 3)[:2, 0]
          yaw = np.arctan2(fwd[1], fwd[0]) + ang
          sim.push(force * np.array([np.cos(yaw), np.sin(yaw)]), 0.2)
          survived = all(sim.tick() for _ in range(int(3.0 / dt)))
        ok += survived
      row.append(ok / len(dirs))
    print(f"{label:18s}" + "".join(f"{r:>8.0%} " for r in row))
  print("(fraction of pushes survived)")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("run", help="run directory (uses its policy.npz) or a .npz file")
  p.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
  p.add_argument("--video", help="record to .mp4/.gif instead of opening the viewer")
  p.add_argument("--seconds", type=float, default=10.0)
  p.add_argument("--speed", type=float, default=1.0, help="viewer playback speed, 1 = real time")
  p.add_argument("--push-force", type=float, default=40.0, help="viewer Insert-key push [N]")
  p.add_argument("--push-test", action="store_true", help="print push-recovery survival table")
  args = p.parse_args()

  path = args.run if args.run.endswith(".npz") else os.path.join(args.run, "policy.npz")
  sim = Sim(path)
  sim.ctrl.command[:] = args.cmd
  if args.push_test:
    run_push_test(sim, args)
  elif args.video:
    run_video(sim, args)
  else:
    run_viewer(sim, args)


if __name__ == "__main__":
  main()
