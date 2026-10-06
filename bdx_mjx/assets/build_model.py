#!/usr/bin/env python3
"""Build MJX-ready GO-BDX models from the CAD export (robot/go_bdx.xml).

The CAD export has placeholder physics (fake 1e-4 inertias, a 28 kg base whose
mass came from the visual mesh at water density, mesh collisions, unlimited
torque). This script keeps its kinematics and visuals and replaces the physics:

  * Body inertias are computed from each link's mesh volume, scaled to the
    masses in MASSES (estimates - edit them to match your real robot and rerun).
  * Collision is only two flat boxes under the feet, fitted to the sole mesh.
  * Head/neck/antenna joints are frozen (not used for walking).
  * Leg motors are torque-limited position actuators (PD), modelled on the
    Unitree GO-M8010-6 (23.7 Nm peak).
  * An IMU site whose frame is x=forward, y=left, z=up, plus the sensors the
    env reads.

Outputs (in ./xmls):
  go_bdx_train.xml  - no meshes, fast to load, used for MJX training
  go_bdx_view.xml   - identical physics + visual meshes, used for viewing

Usage:
    python bdx_mjx/assets/build_model.py
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_XML = os.path.join(REPO, "robot", "go_bdx.xml")
OUT_DIR = os.path.join(HERE, "xmls")

# Link masses in kg. ESTIMATES (~11.6 kg total) - replace with weighed values.
MASSES = {
    "floating_base": 4.0,
    "hip_roll_left_link": 0.6,
    "hip_pitch_left_link": 0.6,
    "left_upper_leg_link": 1.0,
    "left_lower_leg_link": 0.45,
    "left_foot_link": 0.3,
    "hip_roll_right_link": 0.6,
    "hip_pitch_right_link": 0.6,
    "right_upper_leg_link": 1.0,
    "right_lower_leg_link": 0.45,
    "right_foot_link": 0.3,
    "neck_pitch_link": 0.4,
    "head_pitch_link": 0.25,
    "head_link": 1.0,
    "left_antenna_link": 0.03,
    "right_antenna_link": 0.03,
}

# Leg joints in action order. NOTE the CAD names do not match the real axes:
#   *_hip_roll  -> rotates about vertical axis   (actually hip YAW)
#   *_hip_pitch -> rotates about forward axis    (actually hip ROLL)
#   *_hip_yaw   -> rotates about lateral axis    (actually hip PITCH)
#   *_shin      -> knee pitch,  *_foot -> ankle pitch
LEG_JOINTS = [
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw", "left_shin", "left_foot",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw", "right_shin", "right_foot",
]

# Per-joint actuator class (gains are in emit_xml). kp must beat m*g*h ~ 34
# Nm/rad at the ankle/knee/hip-pitch or the zero-action stance topples.
ACTUATOR_CLASS = {
    "hip_roll": "hip_yaw_act",     # vertical axis
    "hip_pitch": "hip_roll_act",   # forward axis
    "hip_yaw": "hip_pitch_act",    # lateral axis
    "shin": "knee_act",
    "foot": "ankle_act",
}
JOINT_RANGE = 0.7854   # +-45 deg, from the URDF
TORQUE_LIMIT = 23.7    # GO-M8010-6 peak torque [Nm]

SOLE_HALF_THICKNESS = 0.005
IMU_POS_IN_BASE = (0.0, 0.10, 0.0)
# Base frame is x=left, y=backward, z=up. Rotate -90 deg about z so the IMU
# frame is x=forward, y=left, z=up.
IMU_QUAT_IN_BASE = (0.7071068, 0.0, 0.0, -0.7071068)
# Spawn the base rotated +90 deg about z so the robot faces world +x.
HOME_BASE_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)


def _fmt(v, prec=6):
  return " ".join(f"{float(x):.{prec}g}" for x in np.atleast_1d(v))


def compute_physics():
  """Compiles the CAD model with geom-derived inertia; returns inertials and soles."""
  spec = mujoco.MjSpec.from_file(SRC_XML)
  spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
  for mesh in spec.meshes:
    # LEGACY copes with the mirrored (scale -1) right-side meshes.
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY
  for body in spec.bodies:
    if body.name not in MASSES:
      continue
    body.explicitinertial = False
    geoms = body.geoms
    assert len(geoms) == 1, body.name
    geoms[0].mass = MASSES[body.name]
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[2] = 0.3
  mujoco.mj_forward(model, data)

  inertial = {}
  for name in MASSES:
    b = model.body(name).id
    inertial[name] = dict(
        mass=model.body_mass[b],
        pos=model.body_ipos[b].copy(),
        quat=model.body_iquat[b].copy(),
        diag=model.body_inertia[b].copy(),
    )

  # Fit a flat box to the lowest 6 mm of each foot mesh (sole is flat at q=0).
  soles = {}
  sole_bottom = None
  for name in ["left_foot_link", "right_foot_link"]:
    b = model.body(name).id
    g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == b][0]
    mid = model.geom_dataid[g]
    adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[adr:adr + n]
    w = (data.geom_xmat[g].reshape(3, 3) @ v.T).T + data.geom_xpos[g]
    zmin = w[:, 2].min()
    low = w[w[:, 2] < zmin + 0.006]
    lo, hi = low[:, :2].min(0), low[:, :2].max(0)
    center_w = np.array([*(lo + hi) / 2, zmin + SOLE_HALF_THICKNESS])
    half = np.array([*(hi - lo) / 2, SOLE_HALF_THICKNESS])
    # Express in the foot body frame, box axes aligned with the world at q=0.
    rot_b = data.xmat[b].reshape(3, 3)
    pos_b = rot_b.T @ (center_w - data.xpos[b])
    quat_b = np.zeros(4)
    mujoco.mju_negQuat(quat_b, data.xquat[b])
    soles[name] = dict(pos=pos_b, quat=quat_b, size=half)
    sole_bottom = zmin

  home_z = 0.3 - sole_bottom + 0.001
  total = sum(v["mass"] for v in inertial.values())
  return spec, inertial, soles, home_z, total


def emit_xml(spec, inertial, soles, home_z, with_visuals):
  """Writes the MJCF text for the robot + scene."""
  src = mujoco.MjSpec.from_file(SRC_XML)
  mesh_files = {m.name: os.path.basename(m.file) for m in src.meshes}
  mesh_scales = {m.name: m.scale for m in src.meshes}
  meshdir = os.path.relpath(os.path.join(REPO, "robot", "meshes"), OUT_DIR).replace("\\", "/")

  out = []
  w = out.append
  w(f'<mujoco model="go_bdx_{"view" if with_visuals else "train"}">')
  w("  <!-- GENERATED by bdx_mjx/assets/build_model.py - edit that, not this. -->")
  w(f'  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>')
  w('  <option timestep="0.004" iterations="2" ls_iterations="6" integrator="Euler">')
  w('    <flag eulerdamp="disable"/>')
  w("  </option>")
  w('  <statistic center="0 0 0.3" extent="0.8"/>')
  w("  <visual>")
  w('    <headlight diffuse=".7 .7 .7" ambient=".3 .3 .3" specular="0 0 0"/>')
  w('    <global azimuth="160" elevation="-20" offwidth="1920" offheight="1080"/>')
  w('    <quality shadowsize="4096"/>')
  w("  </visual>")
  w("  <default>")
  w('    <joint damping="0.05" armature="0.01" frictionloss="0.05"/>')
  w('    <position forcerange="-{0} {0}" inheritrange="1"/>'.format(TORQUE_LIMIT))
  w('    <default class="hip_yaw_act"><position kp="40" kv="1.5"/></default>')
  w('    <default class="hip_roll_act"><position kp="60" kv="1.5"/></default>')
  w('    <default class="hip_pitch_act"><position kp="60" kv="1.5"/></default>')
  w('    <default class="knee_act"><position kp="60" kv="1.5"/></default>')
  w('    <default class="ankle_act"><position kp="50" kv="1.5"/></default>')
  w('    <default class="visual"><geom type="mesh" contype="0" conaffinity="0" density="0" group="2" rgba="0.75 0.75 0.78 1"/></default>')
  w('    <default class="sole"><geom type="box" contype="0" conaffinity="1" condim="3" friction="0.8 0.02 0.01" group="3" rgba="0.9 0.3 0.2 0.6"/></default>')
  w('    <site group="4" size="0.01"/>')
  w("  </default>")

  w("  <asset>")
  w('    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" rgb2="0.05 0.05 0.08" width="512" height="512"/>')
  w('    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.25 0.27 0.3" rgb2="0.2 0.22 0.25" markrgb="0.5 0.5 0.5" width="300" height="300"/>')
  w('    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.1"/>')
  if with_visuals:
    for name, f in mesh_files.items():
      w(f'    <mesh name="{name}" file="{f}" scale="{_fmt(mesh_scales[name])}"/>')
  w("  </asset>")

  w("  <worldbody>")
  w('    <light pos="0 0 3" dir="0 0 -1" directional="true"/>')
  w('    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="0" priority="1" friction="0.8 0.02 0.01" condim="3"/>')

  def emit_body(body, indent):
    pad = " " * indent
    attrs = f'name="{body.name}" pos="{_fmt(body.pos)}"'
    if not np.allclose(body.quat, [1, 0, 0, 0]):
      attrs += f' quat="{_fmt(body.quat)}"'
    w(f"{pad}<body {attrs}>")
    if body.name == "floating_base":
      w(f'{pad}  <freejoint name="root"/>')
      w(f'{pad}  <site name="imu" pos="{_fmt(IMU_POS_IN_BASE)}" quat="{_fmt(IMU_QUAT_IN_BASE)}"/>')
    ine = inertial[body.name]
    w(f'{pad}  <inertial pos="{_fmt(ine["pos"])}" quat="{_fmt(ine["quat"])}" '
      f'mass="{ine["mass"]:.4g}" diaginertia="{_fmt(ine["diag"])}"/>')
    for j in body.joints:
      if j.name in LEG_JOINTS:
        w(f'{pad}  <joint name="{j.name}" axis="{_fmt(j.axis)}" range="-{JOINT_RANGE} {JOINT_RANGE}"/>')
      # other joints (head/neck/antennas) are dropped -> welded at zero
    if with_visuals:
      for g in body.geoms:
        gattr = f'class="visual" mesh="{g.meshname}"'
        if not np.allclose(g.pos, 0):
          gattr += f' pos="{_fmt(g.pos)}"'
        if not np.allclose(g.quat, [1, 0, 0, 0]):
          gattr += f' quat="{_fmt(g.quat)}"'
        w(f"{pad}  <geom {gattr}/>")
    if body.name in soles:
      side = body.name.split("_")[0]
      s = soles[body.name]
      common = f'pos="{_fmt(s["pos"])}" quat="{_fmt(s["quat"])}"'
      w(f'{pad}  <geom name="{side}_sole" class="sole" {common} size="{_fmt(s["size"])}"/>')
      w(f'{pad}  <site name="{side}_foot" {common}/>')
    for child in body.bodies:
      emit_body(child, indent + 2)
    w(f"{pad}</body>")

  emit_body(src.body("floating_base"), 4)
  w("  </worldbody>")

  w("  <actuator>")
  for j in LEG_JOINTS:
    cls = ACTUATOR_CLASS[j.split("_", 1)[1]]
    w(f'    <position name="{j}" joint="{j}" class="{cls}"/>')
  w("  </actuator>")

  w("  <sensor>")
  w('    <gyro name="gyro" site="imu"/>')
  w('    <accelerometer name="accelerometer" site="imu"/>')
  w('    <velocimeter name="local_linvel" site="imu"/>')
  w('    <framezaxis name="upvector" objtype="site" objname="imu"/>')
  w('    <framelinvel name="global_linvel" objtype="site" objname="imu"/>')
  w('    <frameangvel name="global_angvel" objtype="site" objname="imu"/>')
  for side in ["left", "right"]:
    w(f'    <framepos name="{side}_foot_pos" objtype="site" objname="{side}_foot"/>')
    w(f'    <framelinvel name="{side}_foot_global_linvel" objtype="site" objname="{side}_foot"/>')
    w(f'    <contact name="{side}_sole_floor_found" geom1="{side}_sole" geom2="floor" reduce="mindist" num="1" data="found"/>')
  w("  </sensor>")

  zeros = " ".join(["0"] * len(LEG_JOINTS))
  w("  <keyframe>")
  w(f'    <key name="home" qpos="0 0 {home_z:.4f} {_fmt(HOME_BASE_QUAT)} {zeros}" ctrl="{zeros}"/>')
  w("  </keyframe>")
  w("</mujoco>")
  return "\n".join(out) + "\n"


def main():
  spec, inertial, soles, home_z, total = compute_physics()
  os.makedirs(OUT_DIR, exist_ok=True)
  for fname, vis in [("go_bdx_train.xml", False), ("go_bdx_view.xml", True)]:
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
      f.write(emit_xml(spec, inertial, soles, home_z, vis))
    m = mujoco.MjModel.from_xml_path(path)  # sanity: must compile
    print(f"wrote {path}: nq={m.nq} nu={m.nu} mass={m.body_subtreemass[1]:.2f} kg")
  print(f"total mass {total:.2f} kg, home base height {home_z:.4f} m")


if __name__ == "__main__":
  main()
