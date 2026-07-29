import math
from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).parent.parent / "desc" / "robot.xml"
POS_UNIT_TO_RAD = math.pi


class RigModel:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)

    def torque(self, qpos, qvel, qacc):
        self.data.qpos[:] = np.asarray(qpos) * POS_UNIT_TO_RAD
        self.data.qvel[:] = np.asarray(qvel) * POS_UNIT_TO_RAD
        self.data.qacc[:] = np.asarray(qacc) * POS_UNIT_TO_RAD
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def torque_batch(self, qpos, qvel, qacc):
        qpos = np.asarray(qpos)
        qvel = np.asarray(qvel)
        qacc = np.asarray(qacc)
        return np.stack(
            [self.torque(p, v, a) for p, v, a in zip(qpos, qvel, qacc)]
        )


if __name__ == "__main__":
    # Smoke test on one logged rollout: compute v/a then torque labels.
    from sysid.data_pipeline.data_interface import SysidDSInterface
    from sysid.data_pipeline.process_data import compute_velocities_and_accelerations

    ds = SysidDSInterface(dataset_name="responses")
    rig = RigModel()
    rollout = next(ds.iter_rollouts())
    v, a = compute_velocities_and_accelerations(rollout)
    tau = rig.torque_batch(rollout["sensor_data"], v, a)
    print(f"rollout type={rollout['type']} n={len(tau)}")
    print(f"torque N.m: min={tau.min():.4f} max={tau.max():.4f} mean={tau.mean():.4f}")
