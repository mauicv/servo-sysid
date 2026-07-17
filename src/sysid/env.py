import sys
import functools
from pathlib import Path

import mujoco
import mujoco.viewer
import math
import numpy as np

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


# Position-servo law: force = gainprm[0]*act + biasprm[1]*qpos + biasprm[2]*qvel,
# i.e. kp*(ctrl - qpos) - kv*qvel, with a first-order filter of time constant
# tau (dynprm[0]) on the control. So kp lives in BOTH gainprm[0] and biasprm[1]
# (as -kp), kv in biasprm[2] (as -kv), and tau in dynprm[0].
def set_kp(m, p):
    m.actuator_gainprm[:, 0] = p
    m.actuator_biasprm[:, 1] = -p
def set_kv(m, p): m.actuator_biasprm[:, 2] = -p
def set_tau(m, p): m.actuator_dynprm[:, 0] = p
def set_damping(m, p): m.dof_damping[:] = p
def set_frictionloss(m, p): m.dof_frictionloss[:] = p
def set_armature(m, p): m.dof_armature[:] = p
def set_force_limit(m, p): m.actuator_forcerange[:, 0:2] = np.array([-p, p])

attr_map = {
    'kp': set_kp,
    'kv': set_kv,
    'tau': set_tau,
    'damping': set_damping,
    'frictionloss': set_frictionloss,
    'armature': set_armature,
    'force_limit': set_force_limit,
}

class Env:
    def __init__(self, params, initial_states=None, initial_velocities=None):
        self.params = params
        self.initial_states = initial_states
        self.initial_velocities = initial_velocities
        self.model = _load_model()
        # Bit for opt.disableactuator that switches the servo off (zero torque).
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "servo")
        self._servo_disable_mask = 1 << int(self.model.actuator_group[aid])
        self._set_servo_powered(True)
        self._apply_params(self.params)
        self.data = mujoco.MjData(self.model)
        if initial_states is not None:
            self.data.qpos[:] = self.initial_states
            self.data.qvel[:] = self.initial_velocities
        # self.viewer = mujoco.viewer.launch(self.model, self.data)
        self.n_substeps = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))

    def _set_servo_powered(self, powered):
        # disableactuator is a group bitfield: setting the servo's bit makes the
        # actuator produce zero force (unlike ctrl=0, which still applies the
        # affine position/velocity bias, i.e. the servo keeps holding).
        if powered:
            self.model.opt.disableactuator &= ~self._servo_disable_mask
        else:
            self.model.opt.disableactuator |= self._servo_disable_mask

    def _apply_params(self, params):
        m = self.model

        for key, value in params.items():
            attr_map[key](m, value)


    def step(self, action):
        # action=None -> cut servo torque (free swing / depowered). Otherwise
        # power the servo and command it (ctrl in [-1.963, 1.963] ~ radians;
        # action is in [-1, 1]).
        if action is None:
            self._set_servo_powered(False)
        else:
            self._set_servo_powered(True)
            self.data.ctrl[:] = action * math.pi
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.data.sensordata / math.pi

    def reset(self, initial_states=None, initial_velocities=None):
        # Passing new initial states lets a single Env be reused across rollouts
        # (avoids reallocating MjData every rollout).
        if initial_states is not None:
            self.initial_states = initial_states
            self.initial_velocities = initial_velocities
        self._apply_params(self.params)
        self._set_servo_powered(True)
        mujoco.mj_resetData(self.model, self.data)
        if self.initial_states is not None:
            self.data.qpos[:] = self.initial_states
            self.data.qvel[:] = self.initial_velocities

