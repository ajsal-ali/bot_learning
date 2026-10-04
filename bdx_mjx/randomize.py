"""Domain randomization for GO-BDX.

Every parallel env gets its own physics, so the policy can't overfit to one
(guessed) set of masses, gains and frictions. This matters a lot here because
the link masses in the model are estimates.
"""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1
NUM_JOINTS = 10


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    keys = jax.random.split(rng, 9)
    u = lambda k, lo, hi, shape=(): jax.random.uniform(k, shape, minval=lo, maxval=hi)

    # Floor friction.
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(u(keys[0], 0.4, 1.0))

    # Joint friction, damping and armature (motor/gearbox variation).
    dof_frictionloss = model.dof_frictionloss.at[6:].multiply(
        u(keys[1], 0.5, 2.0, (NUM_JOINTS,)))
    dof_damping = model.dof_damping.at[6:].multiply(u(keys[2], 0.5, 2.0, (NUM_JOINTS,)))
    dof_armature = model.dof_armature.at[6:].multiply(u(keys[3], 1.0, 1.5, (NUM_JOINTS,)))

    # Link masses +-10%, extra payload on the torso, torso CoM offset.
    body_mass = model.body_mass * u(keys[4], 0.9, 1.1, (model.nbody,))
    body_mass = body_mass.at[TORSO_BODY_ID].add(u(keys[5], -0.5, 1.0))
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].add(u(keys[6], -0.015, 0.015, (3,)))

    # Motor PD gains (kp +-15%, kd +-20%).
    kp = model.actuator_gainprm[:, 0] * u(keys[7], 0.85, 1.15, (model.nu,))
    kd = -model.actuator_biasprm[:, 2] * u(keys[8], 0.8, 1.2, (model.nu,))
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)

    return (geom_friction, dof_frictionloss, dof_damping, dof_armature,
            body_mass, body_ipos, actuator_gainprm, actuator_biasprm)

  (geom_friction, dof_frictionloss, dof_damping, dof_armature, body_mass,
   body_ipos, actuator_gainprm, actuator_biasprm) = rand_dynamics(rng)

  fields = {
      "geom_friction": geom_friction,
      "dof_frictionloss": dof_frictionloss,
      "dof_damping": dof_damping,
      "dof_armature": dof_armature,
      "body_mass": body_mass,
      "body_ipos": body_ipos,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  }
  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({k: 0 for k in fields})
  model = model.tree_replace(fields)
  return model, in_axes
