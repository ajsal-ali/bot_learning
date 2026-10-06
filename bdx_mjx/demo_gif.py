#!/usr/bin/env python3
"""Render the README demo: a scripted sequence of velocity commands and a push,
with the active command and the measured velocity overlaid on every frame.

    python -m bdx_mjx.demo_gif pretrained/go_bdx_walk.npz
    python -m bdx_mjx.demo_gif runs/<run> --out docs/media/walk_demo.gif

Plain MuJoCo + the NumPy policy, exactly like play.py (no JAX needed).
"""

import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bdx_mjx.play import Sim  # pylint: disable=wrong-import-position

# (seconds, label, [vx m/s, vy m/s, yaw rad/s], push [N] or 0)
SCRIPT = [
    (2.0, "stand still", [0.0, 0.0, 0.0], 0),
    (4.0, "forward  0.30 m/s", [0.3, 0.0, 0.0], 0),
    (3.0, "turn left  0.60 rad/s", [0.0, 0.0, 0.6], 0),
    (3.0, "backward  0.20 m/s", [-0.2, 0.0, 0.0], 0),
    (2.5, "sidestep left  0.15 m/s", [0.0, 0.15, 0.0], 0),
    (3.5, "stand still + 40 N side push", [0.0, 0.0, 0.0], 40),
]
PUSH_AT, PUSH_FOR = 0.8, 0.2  # seconds into the segment


def font(size):
  for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
    try:
      return ImageFont.truetype(name, size)
    except OSError:
      pass
  return ImageFont.load_default(size=size)


def add_arrow(scene, start, end, rgba, width):
  if scene.ngeom >= scene.maxgeom:
    return
  g = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                      np.zeros(9), np.asarray(rgba, np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start, float),
                       np.asarray(end, float))
  scene.ngeom += 1


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("policy", help="a run directory or a policy .npz")
  p.add_argument("--out", default="docs/media/walk_demo.gif")
  p.add_argument("--width", type=int, default=400)
  p.add_argument("--height", type=int, default=300)
  p.add_argument("--colors", type=int, default=64, help="GIF palette size (smaller file)")
  p.add_argument("--fps", type=int, default=25, choices=[10, 25, 50])
  args = p.parse_args()

  path = args.policy if args.policy.endswith(".npz") else os.path.join(args.policy, "policy.npz")
  sim = Sim(path)
  m, d = sim.model, sim.data
  renderer = mujoco.Renderer(m, height=args.height, width=args.width)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = sim.base
  cam.distance, cam.elevation, cam.azimuth = 1.35, -18, 130
  big, small = font(max(12, args.width // 27)), font(max(10, args.width // 36))
  every = 50 // args.fps
  frames, t = [], 0.0

  for seconds, label, cmd, push_n in SCRIPT:
    sim.ctrl.command[:] = cmd
    for i in range(int(seconds / sim.ctrl.dt)):
      ts = i * sim.ctrl.dt
      if push_n and abs(ts - PUSH_AT) < 1e-6:
        side = d.site_xmat[sim.imu].reshape(3, 3)[:2, 1]  # robot's left in world
        sim.push(-push_n * side / np.linalg.norm(side), PUSH_FOR)  # push from the left
      sim.tick()
      t += sim.ctrl.dt
      if i % every:
        continue

      renderer.update_scene(d, cam)
      base = d.xpos[sim.base]
      # green: commanded planar velocity (drawn on the floor, in the robot's frame)
      rot = d.site_xmat[sim.imu].reshape(3, 3)
      v_world = rot[:, 0] * cmd[0] + rot[:, 1] * cmd[1]
      v_world[2] = 0
      if np.linalg.norm(v_world) > 1e-3:
        s = np.array([base[0], base[1], 0.01])
        add_arrow(renderer.scene, s, s + v_world / np.linalg.norm(v_world) * (0.15 + np.linalg.norm(v_world)),
                  [0.2, 0.85, 0.3, 0.9], 0.012)
      # red: push force on the torso
      if sim.push_ticks > 0:
        f = np.array([*sim.push_force, 0.0]); f /= np.linalg.norm(f)
        add_arrow(renderer.scene, base - f * 0.45, base - f * 0.08, [0.95, 0.15, 0.1, 1], 0.02)

      img = Image.fromarray(renderer.render())
      draw = ImageDraw.Draw(img, "RGBA")
      v = sim.local_velocity()
      yaw = d.sensordata[sim.gyro.adr[0] + 2]
      line1 = f"cmd: {label}"
      line2 = f"measured  vx {v[0]:+.2f}  vy {v[1]:+.2f} m/s  yaw {yaw:+.2f} rad/s"
      h1, h2 = big.size + 6, small.size + 6
      box_w = 16 + max(draw.textlength(line1, font=big), draw.textlength(line2, font=small))
      draw.rectangle([6, 6, 6 + box_w, 10 + h1 + h2], fill=(10, 12, 16, 175))
      draw.text((14, 9), line1, font=big, fill=(120, 230, 140))
      draw.text((14, 9 + h1), line2, font=small, fill=(230, 230, 230))
      if sim.push_ticks > 0:
        draw.text((14, 16 + h1 + h2), f"PUSH {push_n} N", font=big, fill=(255, 90, 70))
      draw.text((args.width - 6 - draw.textlength(f"t {t:4.1f}s", font=small), args.height - small.size - 8),
                f"t {t:4.1f}s", font=small, fill=(230, 230, 230))
      frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=args.colors))

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=1000 // args.fps,
                 loop=0, optimize=True)
  print(f"wrote {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.1f} MB, "
        f"falls={sim.falls}")


if __name__ == "__main__":
  main()
