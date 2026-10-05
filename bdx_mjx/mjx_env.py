# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Env base class, trimmed from MuJoCo Playground 0.2.0's mjx_env.py.

Vendored so playground does not need to be installed (the conda-forge build is too old).
"""

import abc
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from flax import struct
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]


def make_data(
    model: mujoco.MjModel,
    qpos: Optional[jax.Array] = None,
    qvel: Optional[jax.Array] = None,
    ctrl: Optional[jax.Array] = None,
    impl: Optional[str] = None,
    naconmax: Optional[int] = None,
    njmax: Optional[int] = None,
) -> mjx.Data:
  """Initialize MJX Data."""
  kwargs = {}
  if naconmax is not None:
    kwargs["naconmax"] = naconmax
  if njmax is not None:
    kwargs["njmax"] = njmax
  data = mjx.make_data(model, impl=impl, **kwargs)
  if qpos is not None:
    data = data.replace(qpos=qpos)
  if qvel is not None:
    data = data.replace(qvel=qvel)
  if ctrl is not None:
    data = data.replace(ctrl=ctrl)
  return data


def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
  def single_step(data, _):
    data = data.replace(ctrl=action)
    data = mjx.step(model, data)
    return data, None

  return jax.lax.scan(single_step, data, (), n_substeps)[0]


@struct.dataclass
class State:
  """Environment state for training and inference."""

  data: mjx.Data
  obs: Observation
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array]
  info: Dict[str, Any]

  def tree_replace(
      self, params: Dict[str, Optional[jax.typing.ArrayLike]]
  ) -> "State":
    new = self
    for k, v in params.items():
      new = _tree_replace(new, k.split("."), v)
    return new


def _tree_replace(base: Any, attr: Sequence[str], val: Any) -> Any:
  """Sets attributes in a struct.dataclass with values."""
  if not attr:
    return base
  if len(attr) == 1:
    return base.replace(**{attr[0]: val})
  return base.replace(
      **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
  )


class MjxEnv(abc.ABC):
  """Base class for MJX environments."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    self._config = config.lock()
    if config_overrides:
      self._config.update_from_flattened_dict(config_overrides)
    self._ctrl_dt = config.ctrl_dt
    self._sim_dt = config.sim_dt

  @abc.abstractmethod
  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""

  @property
  @abc.abstractmethod
  def xml_path(self) -> str:
    """Path to the xml file for the environment."""

  @property
  @abc.abstractmethod
  def action_size(self) -> int:
    """Size of the action space."""

  @property
  @abc.abstractmethod
  def mj_model(self) -> mujoco.MjModel:
    """Mujoco model for the environment."""

  @property
  @abc.abstractmethod
  def mjx_model(self) -> mjx.Model:
    """Mjx model for the environment."""

  @property
  def dt(self) -> float:
    """Control timestep for the environment."""
    return self._ctrl_dt

  @property
  def sim_dt(self) -> float:
    """Simulation timestep for the environment."""
    return self._sim_dt

  @property
  def n_substeps(self) -> int:
    """Number of sim steps per control step."""
    return int(round(self.dt / self.sim_dt))

  @property
  def observation_size(self) -> ObservationSize:
    abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
    obs = abstract_state.obs
    if isinstance(obs, Mapping):
      return jax.tree_util.tree_map(lambda x: x.shape, obs)
    return obs.shape[-1]

  @property
  def unwrapped(self) -> "MjxEnv":
    return self
