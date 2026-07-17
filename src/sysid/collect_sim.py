from sysid.data_interface import DSInterface
from sysid.config import CONTROL_HZ
from sysid.env import Env
import time
import json
from tqdm import tqdm

best_params_9 = {"kp": 55.16425524424094, "kv": 2.8441638016834414, "tau": 0.013807696318891036, "damping": 1.5439016340657283, "frictionloss": 0.06249036222116202, "armature": 0.12637940307607023, "force_limit": 8.40629855285546}
# initial_params = {'tau': 0.075}4
if __name__ == '__main__':
    ds = DSInterface('actions-dataset')
    env = Env(params=best_params_9)

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
