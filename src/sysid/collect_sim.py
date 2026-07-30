from sysid.data_interface import DSInterface
from sysid.config import CONTROL_HZ
from sysid.env import Env
import time
import json
from tqdm import tqdm

params = {'kp': 56.121566743762834, 'kv': 2.8935209569975355, 'damping': 0.17889452739994438, 'frictionloss': 0.0072408653310164755, 'armature': 0.014643797951585229, 'force_limit': 2.746855360183254}


if __name__ == '__main__':
    ds = DSInterface('actions-dataset')
    env = Env(params=params)

    data = {
        'dataset': 'simulation',
        'config': ds.config,
        'data': [],
    }

    print(f'Collecting {len(ds)} rollouts...')
    pbar = tqdm(total=len(ds))

    for rollout in ds.iter_rollouts():
        pbar.update(1)
        env.reset()

        rollout_data = {
            'type': rollout['type'],
            'actions': [],
            'sensor_data': [],
        }
        for action in rollout['actions']:
            sensor_data = env.step(action)
            rollout_data['actions'].append(action)
            rollout_data['sensor_data'].append(sensor_data.tolist())

        data['data'].append(rollout_data)

    pbar.close()
    with open('src/sysid/dataset/sim-action-state-dataset.json', 'w') as f:
        json.dump(data, f)
