from sysid.config import CONTROL_HZ
from sysid.data_pipeline.data_interface import SysidDSInterface
from sysid.data_pipeline.inverse_dynamics import RigModel
from tqdm import tqdm
import json
import os
try:
    import numpy as np
    from scipy.signal import savgol_filter
except ImportError:
    print('numpy and scipy are not installed. Please install them to use this module.')

# Single-pass Savitzky-Golay: compute BOTH derivatives directly from raw
# positions (cascaded filters give inconsistent estimates). Acceleration needs a
# wider window / higher polyorder than velocity.
VEL_WINDOW, VEL_POLYORDER = 11, 3
ACCEL_WINDOW, ACCEL_POLYORDER = 21, 4

# savgol's derivative estimate is a polynomial fit; within ~window/2 of each end
# it extrapolates and blows up (huge spurious q̈ on near-flat/noisy boundaries,
# which then produce non-physical torque labels). Drop those boundary samples.
EDGE_PAD = ACCEL_WINDOW // 2

# The servo can't produce more than ~2.7 N.m (25 kg.cm stall). A label above this
# is a differentiation artifact (step/square settling corners, where savgol rings
# on the sharp stop). Truncate each rollout at its first such sample, keeping the
# clean launch/slew part and discarding the corrupted tail.
MAX_TORQUE = 2.7
MIN_VALID = 5  # drop a rollout if fewer than this many samples survive truncation


def compute_velocities_and_accelerations(rollout):
    states = np.array(rollout['sensor_data'])
    dt = 1 / CONTROL_HZ
    v = savgol_filter(states, window_length=VEL_WINDOW, polyorder=VEL_POLYORDER,
                      deriv=1, delta=dt, axis=0)
    a = savgol_filter(states, window_length=ACCEL_WINDOW, polyorder=ACCEL_POLYORDER,
                      deriv=2, delta=dt, axis=0)
    return v, a


def process_data():
    ds = SysidDSInterface(dataset_name='responses')
    rig = RigModel()
    kept, dropped = [], 0
    pbar = tqdm(total=len(ds))
    for rollout in ds.iter_rollouts():
        pbar.update(1)
        states = np.asarray(rollout['sensor_data'])
        actions = np.asarray(rollout['actions'])
        velocities, accelerations = compute_velocities_and_accelerations(rollout)
        tau = rig.torque_batch(states, velocities, accelerations)[:, 0]

        # 1) Trim the unreliable savgol boundary samples from every aligned array
        #    so positions/actions/velocities/accelerations/torque stay in lock-step.
        sl = slice(EDGE_PAD, len(states) - EDGE_PAD)
        states, actions = states[sl], actions[sl]
        velocities, accelerations, tau = velocities[sl], accelerations[sl], tau[sl]

        # 2) Truncate at the first non-physical torque (differentiation artifact).
        bad = np.nonzero(np.abs(tau) > MAX_TORQUE)[0]
        end = int(bad[0]) if len(bad) else len(tau)
        if end < MIN_VALID:
            dropped += 1
            continue

        rollout['sensor_data'] = states[:end].tolist()
        rollout['actions'] = actions[:end].tolist()
        rollout['velocities'] = velocities[:end].tolist()
        rollout['accelerations'] = accelerations[:end].tolist()
        rollout['torque'] = tau[:end].tolist()
        kept.append(rollout)
    pbar.close()
    ds.data['data'] = kept
    print(f'kept {len(kept)} rollouts, dropped {dropped} (too short after truncation)')

    filename = os.path.dirname(__file__) + '/../dataset/dataset.json'
    print('saving dataset to', filename)
    with open(filename, 'w') as f:
        json.dump(ds.data, f)
