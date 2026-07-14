import sys
import functools
from pathlib import Path

import mujoco
import mujoco.viewer
import math

from sysid.config import CONTROL_HZ
XML = Path(__file__).parent / "desc" / "robot.xml"

PHYSICS_DT    = 0.002


@functools.lru_cache(maxsize=1)
def _load_model():
    # Parse the XML and load meshes ONCE per process. The structure never
    # changes between rollouts — only the (runtime) actuator/dof params do — so
    # reloading per rollout just re-parses 13 STLs for nothing (~38ms -> ~3ms).
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = PHYSICS_DT
    return model


class Env:
    def __init__(self, params, initial_states=None, initial_velocities=None):
        self.params = params
        self.initial_states = initial_states
        self.initial_velocities = initial_velocities
        self.model = _load_model()
        self._apply_params(self.params)
        self.data = mujoco.MjData(self.model)
        if initial_states is not None:
            self.data.qpos[:] = self.initial_states
            self.data.qvel[:] = self.initial_velocities
        # self.viewer = mujoco.viewer.launch(self.model, self.data)
        self.n_substeps = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))

    def _apply_params(self, params):
        # Plain-MuJoCo equivalent of the MJX tree_replace: MjModel fields are
        # mutable numpy arrays, so assign in place (no recompile needed — these
        # are read live by mj_step).
        kp = params["kp"]
        kv = params["kv"]
        m = self.model
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -kv
        m.actuator_dynprm[:, 0]  = params["tau"]
        m.dof_damping[:]         = params["damping"]
        m.dof_frictionloss[:]    = params["frictionloss"]
        m.dof_armature[:]        = params["armature"]
        m.actuator_forcelimited[:] = True
        m.actuator_forcerange[:, 0] = -params["force_limit"]
        m.actuator_forcerange[:, 1] = params["force_limit"]


    def step(self, action):
        # mujoco control range is [-1.963, 1.963] ~ radians but action is in [-1, 1]
        self.data.ctrl[:] = action * math.pi
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.data.sensordata / math.pi

    def reset(self):
        self._apply_params(self.params)
        mujoco.mj_resetData(self.model, self.data)
        if self.initial_states is not None:
            self.data.qpos[:] = self.initial_states
            self.data.qvel[:] = self.initial_velocities

