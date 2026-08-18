from sysid.data_interface import DSInterface
from sysid.config import CONTROL_HZ
from sysid.env import Env
import time
import json
from tqdm import tqdm

# best_params_9 = {"kp": 55.16425524424094, "kv": 2.8441638016834414, "damping": 1.5439016340657283, "frictionloss": 0.06249036222116202, "armature": 0.12637940307607023, "force_limit": 8.40629855285546}
# best_params_9 = {"kp": 55, "kv": 3, "damping": 1.5439016340657283, "frictionloss": 0.06249036222116202, "armature": 0.12637940307607023, "force_limit": 2.746855360183254}
# params = {"kp": 49.80626029233553, "kv": 2.5679157997056907, "damping": 0.1587636966237435, "frictionloss": 0.006426057652042094, "armature": 0.012995945315907826, "force_limit": 2.4377543428059765}
# params = {"kp": 77.99970388177452, "kv": 4.021515986037546, "damping": 0.2486338313927485, "frictionloss": 0.010063606282514398, "armature": 0.020352459316455393, "force_limit": 3.817675042441136}
params = {"kp": 36.81949096939071, "kv": 1.8983427392955468, "damping": 0.11736674184720024, "frictionloss": 0.004750490606992202, "armature": 0.00960730816546369, "force_limit": 2.4}
params = {'kp': 42.17808137088663, 'kv': 2.1746214415186613, 'damping': 0.13444792031433667, 'frictionloss': 0.005441861744908997, 'armature': 0.011005524924149064, 'force_limit': 2.7492882879418965}


if __name__ == '__main__':
    ds = DSInterface('actions')
    env = Env(params=params, action_delay=2)

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
            'actions': [],
            'sensor_data': [],
        }
        for action in rollout['env_actions']:
            sensor_data = env.step(action)
            rollout_data['actions'].append(action)
            rollout_data['sensor_data'].append(sensor_data.tolist())

        data['data'].append(rollout_data)

    pbar.close()
    with open('src/sysid/dataset/sim-action-state-dataset.json', 'w') as f:
        json.dump(data, f)
