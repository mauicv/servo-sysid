from sysid.data_interface import DSInterface
from sysid.config import CONTROL_HZ
from sysid.env import Env
import time
import json
from tqdm import tqdm

best_params_1 = {"kp": 21050.358986742154, "kv": 1186.7635371497274, "tau": 0.0027956143765842474, "damping": 0.5184443265724499, "frictionloss": 0.00936883713610245, "armature": 0.039743636806239574}
best_params_2 = {"kp": 1980.7417515281231, "kv": 155.21882122292646, "tau": 0.0011978782664407155, "damping": 0.44993017530483037, "frictionloss": 0.12802355826237755, "armature": 0.03528190323217326}
best_params_3 = {"kp": 110.66124336854722, "kv": 10.285738506060557, "tau": 0.0037387600559570366, "damping": 0.2930799544673452, "frictionloss": 0.42908244869066225, "armature": 0.034961515804636425, "force_limit": 2.27009311297252}
best_params_4 = {"kp": 35.36364343998168, "kv": 1.5052295049348057, "tau": 0.0027038565728527268, "damping": 0.9275739346202393, "armature": 0.09043794646817424, "force_limit": 5.990346584052404}
best_params_5 = {"kp": 21.896475185763162, "kv": 5.157220084016723, "tau": 0.04284430444317586, "damping": 0.07344569500032361, "armature": 0.008076347360570517, "force_limit": 2.1560559485816437}
best_params_6 = {"kp": 158.62945692582923, "kv": 5.743503956633736, "tau": 0.015841356699969715, "damping": 2.76676801635874, "armature": 0.15837818048685476, "force_limit": 13.710588904525258}
best_params_7 = {"kp": 36.555711631716456, "kv": 1.3439550356858605, "tau": 0.08348584090882719, "damping": 1.6840637165921486, "frictionloss": 0.10659998508713124, "armature": 0.16877042263255881, "force_limit": 7.766416504559106}
best_params_8 ={"kp": 39.575307690601825, "kv": 4.014456708313679, "tau": 0.01557856952574512, "damping": 1.1601280782347234, "frictionloss": 0.07321278356886803, "armature": 0.1284883142837152, "force_limit": 14.030026386125037}
best_params_9 = {"kp": 55.16425524424094, "kv": 2.8441638016834414, "tau": 0.013807696318891036, "damping": 1.5439016340657283, "frictionloss": 0.06249036222116202, "armature": 0.12637940307607023, "force_limit": 8.40629855285546}
initial_params = {'kp': 500, 'kv': 40, 'tau': 0.1, 'damping': 0.1, 'frictionloss': 0.1, 'armature': 0.005}
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
