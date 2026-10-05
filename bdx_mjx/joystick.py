"""Joystick walking task for GO-BDX in MJX.

The policy gets a velocity command (vx, vy, yaw rate) and must track it with a
periodic gait. Standing is just the zero command, so there is no separate
standing / balance / stepping curriculum - one reward function for the whole run.

Actor observations only use what a real robot can measure (IMU gyro + gravity,
joint encoders, last action, command, gait clock). The critic also gets
privileged sim state (true base velocity, contacts, foot velocities, ...).
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from bdx_mjx import mjx_env

from bdx_mjx import constants as consts


def swing_height_profile(phase: jax.Array, swing_height: float) -> jax.Array:
  """Desired foot height for a gait phase in [-pi, pi).

  First half of the cycle is swing (smooth lift to swing_height and back down),
  second half is stance (foot on the floor). Legs are pi apart, so exactly one
  foot swings at a time.
  """
  x = (phase + jp.pi) / (2 * jp.pi)  # [0, 1)
  bezier = lambda t: t**3 + 3 * t**2 * (1 - t)  # smooth 0 -> 1
  up = swing_height * bezier(jp.clip(4 * x, 0.0, 1.0))
  down = swing_height * (1 - bezier(jp.clip(4 * x - 1, 0.0, 1.0)))
  return jp.where(x < 0.25, up, jp.where(x < 0.5, down, 0.0))


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,   # 50 Hz policy, same as the real robot should run
      sim_dt=0.004,   # 5 physics substeps per action
      episode_length=1000,  # 20 s
      action_repeat=1,
      # motor_target = default_pose + action * action_scale  (action in [-1, 1])
      action_scale=0.4,
      soft_joint_pos_limit_factor=0.95,
      impl="jax",       # "jax" works everywhere; "warp" needs Linux + NVIDIA
      naconmax=8 * 8192,  # warp only: total contacts over the whole batch
      njmax=40,           # warp only: constraint rows per env
      noise_config=config_dict.create(
          level=1.0,  # 0.0 disables observation noise
          scales=config_dict.create(
              joint_pos=0.03,   # rad
              joint_vel=1.5,    # rad/s
              gravity=0.05,
              gyro=0.2,         # rad/s
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.5,
              tracking_ang_vel=0.8,
              # Base.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.15,
              orientation=-2.0,
              base_height=0.0,
              # Energy / smoothness.
              torques=-1e-4,
              action_rate=-0.02,
              energy=0.0,
              # Feet.
              feet_air_time=2.0,
              feet_phase=1.0,
              feet_slip=-0.5,
              feet_height=0.0,
              feet_distance=-5.0,
              # Pose / limits.
              pose=-0.5,
              dof_pos_limits=-1.0,
              # Episode.
              termination=-1.0,
              alive=0.0,
          ),
          # exp(-err^2 / sigma). Commands are <= 0.5 m/s, so 0.1 makes a 0.3 m/s
          # error worth 0.41 of the max - standing still clearly loses to walking.
          tracking_sigma=0.1,
          ang_tracking_sigma=0.2,
          base_height_target=0.28,
          feet_phase_sigma=0.002,
          min_feet_distance=0.14,  # lateral, metres (default stance ~0.20)
          air_time_range=[0.15, 0.4],  # seconds
      ),
      gait_config=config_dict.create(
          freq_range=[1.4, 1.8],  # steps per second per foot
          swing_height=0.05,      # metres
      ),
      # Pushes: a horizontal force on the torso CoM, random direction, held for
      # duration_range seconds, every interval_range seconds. Measured on this
      # model: with zero action (stiff stance) a 0.2 s push topples it from 30 N
      # forward / 50 N backward or sideways. The top of the range (55 N x 0.2 s =
      # 11 N s on 11.6 kg, ~0.95 m/s) is beyond that, so it must step to recover;
      # capture point v*sqrt(h/g) ~ 0.16 m is one big or two normal steps. Much
      # bigger pushes are not recoverable and only teach that falling is
      # unavoidable. train.py ramps force_range up in phases (CURRICULUM).
      push_config=config_dict.create(
          enable=True,
          interval_range=[4.0, 8.0],    # seconds between pushes
          force_range=[10.0, 55.0],     # newtons (final curriculum phase)
          duration_range=[0.1, 0.2],    # seconds
      ),
      command_config=config_dict.create(
          lin_vel_x=[-0.3, 0.5],
          lin_vel_y=[-0.2, 0.2],
          ang_vel_yaw=[-0.8, 0.8],
          zero_prob=0.2,  # standing still (and taking pushes while standing)
          resample_steps=500,  # 10 s
      ),
      termination_config=config_dict.create(
          min_up_z=0.6,         # cos(53 deg)
          min_base_height=0.18,
      ),
      reset_config=config_dict.create(
          joint_noise=0.1,  # rad
          base_vel_noise=0.3,
      ),
  )


class Joystick(mjx_env.MjxEnv):
  """Track a joystick velocity command with a periodic gait."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    self._xml_path = consts.TRAIN_XML
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    m = self._mj_model
    home = m.keyframe("home")
    self._init_q = jp.array(home.qpos)
    self._default_pose = jp.array(home.qpos[7:])

    self._lowers, self._uppers = m.jnt_range[1:].T  # joint 0 is the freejoint
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = jp.array(c - 0.5 * r * f)
    self._soft_uppers = jp.array(c + 0.5 * r * f)
    self._lowers, self._uppers = jp.array(self._lowers), jp.array(self._uppers)

    weights = np.zeros(consts.NUM_JOINTS)
    weights[list(consts.HIP_YAW_IDS)] = 1.0
    weights[list(consts.HIP_ROLL_IDS)] = 1.0
    weights[list(consts.HIP_PITCH_IDS)] = 0.01
    weights[list(consts.KNEE_IDS)] = 0.01
    weights[list(consts.ANKLE_IDS)] = 0.5
    self._pose_weights = jp.array(weights)

    self._imu_site_id = m.site("imu").id
    self._torso_body_id = m.body(consts.ROOT_BODY).id
    self._feet_site_id = np.array([m.site(s).id for s in consts.FEET_SITES])
    sole = m.geom(consts.FEET_GEOMS[0]).id
    self._sole_half_thickness = float(m.geom_size[sole][2])

    # Precomputed (address, dim) for every sensor -> static slices under jit.
    self._sensors = {
        m.sensor(i).name: (int(m.sensor_adr[i]), int(m.sensor_dim[i]))
        for i in range(m.nsensor)
    }

  # ---------------------------------------------------------------- sensors

  def _sensor(self, data: mjx.Data, name: str) -> jax.Array:
    adr, dim = self._sensors[name]
    return data.sensordata[adr:adr + dim]

  def _feet_contact(self, data: mjx.Data) -> jax.Array:
    return jp.array([
        self._sensor(data, n)[0] > 0 for n in consts.FEET_CONTACT_SENSORS
    ])

  def _feet_linvel(self, data: mjx.Data) -> jax.Array:
    return jp.stack([self._sensor(data, n) for n in consts.FEET_LINVEL_SENSORS])

  def _feet_height(self, data: mjx.Data) -> jax.Array:
    """Height of each sole's bottom above the floor."""
    return data.site_xpos[self._feet_site_id][:, 2] - self._sole_half_thickness

  # ------------------------------------------------------------------ reset

  def reset(self, rng: jax.Array) -> mjx_env.State:
    cfg = self._config
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Random position / heading.
    rng, k_xy, k_yaw, k_q, k_v = jax.random.split(rng, 5)
    qpos = qpos.at[0:2].add(jax.random.uniform(k_xy, (2,), minval=-0.5, maxval=0.5))
    yaw = jax.random.uniform(k_yaw, (), minval=-jp.pi, maxval=jp.pi)
    yaw_quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(yaw_quat, qpos[3:7]))

    # Perturbed joints and base velocity.
    dq = cfg.reset_config.joint_noise * jax.random.uniform(
        k_q, (consts.NUM_JOINTS,), minval=-1.0, maxval=1.0)
    qpos = qpos.at[7:].set(jp.clip(qpos[7:] + dq, self._lowers, self._uppers))
    qvel = qvel.at[0:6].set(cfg.reset_config.base_vel_noise * jax.random.uniform(
        k_v, (6,), minval=-1.0, maxval=1.0))

    data = self._make_data(qpos, qvel, ctrl=qpos[7:])
    data = mjx.forward(self.mjx_model, data)

    rng, k_freq, k_cmd, k_push = jax.random.split(rng, 4)
    gait_freq = jax.random.uniform(
        k_freq, minval=cfg.gait_config.freq_range[0],
        maxval=cfg.gait_config.freq_range[1])
    push_interval = jax.random.uniform(
        k_push, minval=cfg.push_config.interval_range[0],
        maxval=cfg.push_config.interval_range[1])

    info = {
        "rng": rng,
        "step": 0,
        "command": self.sample_command(k_cmd),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": self._default_pose,
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "phase_dt": 2 * jp.pi * self.dt * gait_freq,
        "phase": jp.array([0.0, jp.pi]),  # left, right: anti-phase
        "push": jp.zeros(2),            # force currently applied [N]
        "push_force": jp.zeros(2),      # force of the current/last push [N]
        "push_remaining": 0,            # control steps left in the current push
        "push_step": 0,
        "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
    }

    metrics = {f"reward/{k}": jp.zeros(()) for k in cfg.reward_config.scales}
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info, self._feet_contact(data))
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def _make_data(self, qpos, qvel, ctrl) -> mjx.Data:
    kwargs = {}
    if self._config.impl == "warp":
      kwargs = dict(naconmax=self._config.naconmax, njmax=self._config.njmax)
    return mjx_env.make_data(
        self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl,
        impl=self.mjx_model.impl.value, **kwargs)

  # ------------------------------------------------------------------- step

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    cfg = self._config
    info = state.info

    # Random push: every push_interval_steps start a horizontal force on the
    # torso, held for a sampled duration (see push_config).
    pc = cfg.push_config
    info["rng"], k_theta, k_mag, k_dur = jax.random.split(info["rng"], 4)
    start = (jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0) & pc.enable
    theta = jax.random.uniform(k_theta, maxval=2 * jp.pi)
    magnitude = jax.random.uniform(
        k_mag, minval=pc.force_range[0], maxval=pc.force_range[1])
    duration = jax.random.uniform(
        k_dur, minval=pc.duration_range[0], maxval=pc.duration_range[1])
    info["push_force"] = jp.where(
        start, jp.array([jp.cos(theta), jp.sin(theta)]) * magnitude, info["push_force"])
    info["push_remaining"] = jp.where(
        start, jp.round(duration / self.dt).astype(jp.int32), info["push_remaining"])
    push = info["push_force"] * (info["push_remaining"] > 0)
    xfrc = state.data.xfrc_applied.at[self._torso_body_id, :2].set(push)
    data = state.data.replace(xfrc_applied=xfrc)

    motor_targets = self._default_pose + action * cfg.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
    info["motor_targets"] = motor_targets

    contact = self._feet_contact(data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += self.dt
    info["swing_peak"] = jp.maximum(info["swing_peak"], self._feet_height(data))

    done = self._get_termination(data)

    # Rewards see this step's phase/command and the previous action.
    rewards = self._get_reward(data, action, info, done, first_contact, contact)
    rewards = {k: v * cfg.reward_config.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Book-keeping, then the observation for the next action (so it holds the
    # action just applied and the next phase - same order as deploy code).
    info["push"] = push
    info["push_remaining"] = jp.where(done, 0, jp.maximum(info["push_remaining"] - 1, 0))
    info["step"] += 1
    info["push_step"] += 1
    phase = info["phase"] + info["phase_dt"]
    info["phase"] = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi
    info["last_last_act"] = info["last_act"]
    info["last_act"] = action
    info["rng"], k_cmd = jax.random.split(info["rng"])
    resample = info["step"] > cfg.command_config.resample_steps
    info["command"] = jp.where(resample, self.sample_command(k_cmd), info["command"])
    info["step"] = jp.where(done | resample, 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    obs = self._get_obs(data, info, contact)
    # On a fall the auto-reset wrapper swaps in the cached first data/obs but
    # keeps info, so put the per-episode info back to what that obs assumes.
    info["phase"] = jp.where(done, jp.array([0.0, jp.pi]), info["phase"])
    info["last_act"] = jp.where(done, 0.0, info["last_act"])
    info["feet_air_time"] = jp.where(done, 0.0, info["feet_air_time"])
    info["swing_peak"] = jp.where(done, 0.0, info["swing_peak"])
    info["last_contact"] = jp.where(done, False, info["last_contact"])

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(info["swing_peak"])

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    tcfg = self._config.termination_config
    up = self._sensor(data, consts.UPVECTOR_SENSOR)
    fallen = (up[2] < tcfg.min_up_z) | (data.qpos[2] < tcfg.min_base_height)
    return fallen | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

  # ------------------------------------------------------------ observation

  def _noisy(self, info, x, scale):
    info["rng"], key = jax.random.split(info["rng"])
    u = 2 * jax.random.uniform(key, shape=x.shape) - 1
    return x + u * self._config.noise_config.level * scale

  def _get_obs(self, data: mjx.Data, info: dict, contact: jax.Array):
    scales = self._config.noise_config.scales
    gyro = self._sensor(data, consts.GYRO_SENSOR)
    # Gravity direction in the IMU frame (what an IMU's attitude filter gives).
    gravity = data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])
    joint_pos = data.qpos[7:]
    joint_vel = data.qvel[6:]
    phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])

    state = jp.hstack([
        self._noisy(info, gyro, scales.gyro),                     # 3
        self._noisy(info, gravity, scales.gravity),               # 3
        info["command"],                                          # 3
        self._noisy(info, joint_pos, scales.joint_pos) - self._default_pose,  # 10
        self._noisy(info, joint_vel, scales.joint_vel),           # 10
        info["last_act"],                                         # 10
        phase,                                                    # 4
    ])

    privileged_state = jp.hstack([
        state,
        gyro,                                                     # 3
        self._sensor(data, consts.ACCELEROMETER_SENSOR),          # 3
        gravity,                                                  # 3
        self._sensor(data, consts.LOCAL_LINVEL_SENSOR),           # 3
        self._sensor(data, consts.GLOBAL_ANGVEL_SENSOR),          # 3
        joint_pos - self._default_pose,                           # 10
        joint_vel,                                                # 10
        data.qpos[2],                                             # 1
        data.actuator_force,                                      # 10
        contact,                                                  # 2
        self._feet_linvel(data).ravel(),                          # 6
        self._feet_height(data),                                  # 2
        info["feet_air_time"],                                    # 2
        info["push"],                                             # 2
    ])
    return {"state": state, "privileged_state": privileged_state}

  # ---------------------------------------------------------------- rewards

  def _get_reward(self, data, action, info, done, first_contact, contact):
    rcfg = self._config.reward_config
    cmd = info["command"]
    local_linvel = self._sensor(data, consts.LOCAL_LINVEL_SENSOR)
    gyro = self._sensor(data, consts.GYRO_SENSOR)
    global_linvel = self._sensor(data, consts.GLOBAL_LINVEL_SENSOR)
    global_angvel = self._sensor(data, consts.GLOBAL_ANGVEL_SENSOR)
    up = self._sensor(data, consts.UPVECTOR_SENSOR)
    qpos = data.qpos[7:]
    torques = data.actuator_force
    feet_vel = self._feet_linvel(data)
    feet_z = self._feet_height(data)

    # Tracking.
    lin_err = jp.sum(jp.square(cmd[:2] - local_linvel[:2]))
    ang_err = jp.square(cmd[2] - gyro[2])

    # Feet air time: reward steps that land after a long enough swing.
    lo, hi = rcfg.air_time_range
    air = jp.clip((info["feet_air_time"] - lo) * first_contact, max=hi - lo)
    air_time = jp.sum(air) * (jp.linalg.norm(cmd) > 0.05)

    # Feet should follow the gait clock's swing-height profile.
    rz = swing_height_profile(info["phase"], self._config.gait_config.swing_height)
    phase_err = jp.sum(jp.square(feet_z - rz))

    # Feet should not slide while in contact.
    slip = jp.sum(jp.sum(jp.square(feet_vel[:, :2]), axis=-1) * contact)

    # Swing apex vs target height, scored at touchdown.
    height_err = info["swing_peak"] / self._config.gait_config.swing_height - 1.0
    feet_height = jp.sum(jp.square(height_err) * first_contact)

    # Keep feet apart sideways (there is no foot-foot collision).
    feet_pos = data.site_xpos[self._feet_site_id]
    left_dir = data.site_xmat[self._imu_site_id][:, 1]  # IMU y-axis in world
    left_dir = left_dir.at[2].set(0.0)
    left_dir = left_dir / (jp.linalg.norm(left_dir) + 1e-6)
    lateral = jp.dot(feet_pos[0] - feet_pos[1], left_dir)
    feet_distance = jp.clip(rcfg.min_feet_distance - lateral, 0.0, None)

    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)

    return {
        "tracking_lin_vel": jp.exp(-lin_err / rcfg.tracking_sigma),
        "tracking_ang_vel": jp.exp(-ang_err / rcfg.ang_tracking_sigma),
        "lin_vel_z": jp.square(global_linvel[2]),
        "ang_vel_xy": jp.sum(jp.square(global_angvel[:2])),
        "orientation": jp.sum(jp.square(up[:2])),
        "base_height": jp.square(data.qpos[2] - rcfg.base_height_target),
        "torques": jp.sum(jp.square(torques)),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "energy": jp.sum(jp.abs(data.qvel[6:] * torques)),
        "feet_air_time": air_time,
        "feet_phase": jp.exp(-phase_err / rcfg.feet_phase_sigma),
        "feet_slip": slip,
        "feet_height": feet_height,
        "feet_distance": feet_distance,
        "pose": jp.sum(jp.square(qpos - self._default_pose) * self._pose_weights),
        "dof_pos_limits": jp.sum(out_of_limits),
        "termination": done,
        "alive": jp.array(1.0),
    }

  def sample_command(self, rng: jax.Array) -> jax.Array:
    c = self._config.command_config
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    cmd = jp.hstack([
        jax.random.uniform(k1, minval=c.lin_vel_x[0], maxval=c.lin_vel_x[1]),
        jax.random.uniform(k2, minval=c.lin_vel_y[0], maxval=c.lin_vel_y[1]),
        jax.random.uniform(k3, minval=c.ang_vel_yaw[0], maxval=c.ang_vel_yaw[1]),
    ])
    return jp.where(jax.random.bernoulli(k4, p=c.zero_prob), jp.zeros(3), cmd)

  # -------------------------------------------------------------- accessors

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
