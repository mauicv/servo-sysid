from sysid.config import CONTROL_HZ
from sysid.data_pipeline.data_interface import SysidDSInterface
from tqdm import tqdm
import json
try:
    import numpy as np
    from scipy.signal import savgol_filter
except ImportError:
    print('numpy and scipy are not installed. Please install them to use this module.')


def compute_velocities_and_accelerations(rollout):
    states = np.array(rollout['sensor_data'])
    dt = 1 / CONTROL_HZ
    v = savgol_filter(states, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
    a = savgol_filter(states, window_length=15, polyorder=4, deriv=2, delta=dt, axis=0)
    return v, a


def compute_va():
    ds = SysidDSInterface(dataset_name='responses')
    print(f'Computing velocities for {len(ds)} rollouts...')
    pbar = tqdm(total=len(ds))
    for rollout in ds.iter_rollouts():
        velocities, accelerations = compute_velocities_and_accelerations(rollout)
        rollout['velocities'] = velocities.tolist()
        rollout['accelerations'] = accelerations.tolist()
        pbar.update(1)
    pbar.close()
    with open('../dataset/dataset.json', 'w') as f:
        json.dump(ds.data, f)