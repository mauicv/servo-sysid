import functools
import math
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from sysid.config import CONTROL_HZ

XML = Path(__file__).parent / "desc" / "robot.xml"
PHYSICS_DT = 0.002


@functools.lru_cache(maxsize=1)
def _load_model():
    # Parse the XML and load meshes ONCE per process (re-parsing 13 STLs per
    # rollout is wasteful; the structure never changes between rollouts).
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = PHYSICS_DT
    return model


class Env:
    """Forward rollout of the bench rig driven by the learned actuator net.

    The rig model has no actuator and no passive friction (it is the inverse-
    dynamics model), so torque is applied directly via `data.qfrc_applied`. Each
    control step we build the (pos_err, velocity) history the net was trained on,
    query it for a torque (N.m), and hold that torque across the physics substeps.

    All positions/velocities are in the normalized policy convention (1.0 == pi rad).
    `actuator` is a callable: features (2*history,) float array -> torque (N.m).
    """

    def __init__(self, actuator, history=3, initial_states=None, initial_velocities=None):
        self.actuator = actuator
        self.history = history
        self.model = _load_model()
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))
        self._pos_err = deque(maxlen=history)
        self._vel = deque(maxlen=history)
        self.reset(initial_states, initial_velocities)

    def reset(self, initial_states=None, initial_velocities=None):
        if initial_states is not None:
            self.initial_states = initial_states
            self.initial_velocities = initial_velocities
        mujoco.mj_resetData(self.model, self.data)
        if getattr(self, "initial_states", None) is not None:
            self.data.qpos[:] = np.asarray(self.initial_states) * math.pi
            self.data.qvel[:] = np.asarray(self.initial_velocities) * math.pi
        self._pos_err.clear()
        self._vel.clear()

    def _observe(self):
        # normalized position/velocity (1.0 == pi rad)
        return self.data.qpos[0] / math.pi, self.data.qvel[0] / math.pi

    def step(self, action):
        """Advance one control step. `action` is the target position (normalized).

        Returns the position at the instant the action was applied (matches how
        the real rig logs sensor_data: read at the start of the control step).
        """
        pos, vel = self._observe()
        pos_err = action - pos

        if not self._pos_err:  # warm-start the history with the first sample
            for _ in range(self.history):
                self._pos_err.append(pos_err)
                self._vel.append(vel)
        else:
            self._pos_err.append(pos_err)
            self._vel.append(vel)

        # feature order matches training: pos_err (oldest..newest) then velocity
        features = np.array(list(self._pos_err) + list(self._vel), dtype=np.float32)
        torque = self.actuator(features)

        self.data.qfrc_applied[0] = torque
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return pos
