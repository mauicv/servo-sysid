from sysid.data_interface import DSInterface
from sysid.hardware import Controller
from sysid.config import CONTROL_HZ
import time
import json
from tqdm import tqdm


if __name__ == '__main__':
    controller = Controller()
    ds = DSInterface(dataset_name='actions-dataset')
    dt = 1.0 / CONTROL_HZ

    data = {
        'dataset': 'real',
        'config': ds.config,
        'data': [],
    }

    print(f'Collecting {len(ds)} rollouts...')
    pbar = tqdm(total=len(ds))

    for rollout in ds.iter_rollouts():
        pbar.update(1)
        controller.center()

        rollout_data = {
            'type': rollout['type'],
            'targets': rollout['targets'],
            'actions': [],
            'sensor_data': [],
        }
        for action in rollout['actions']:
            start_time = time.time()
            controller.send_action([action] * 16)
            sensor_data = controller.get_sensor_data()
            rollout_data['actions'].append(action)
            rollout_data['sensor_data'].append(sensor_data)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < dt:
                time.sleep(dt - elapsed_time)

        data['data'].append(rollout_data)

    pbar.close()
    with open('src/sysid/dataset/response.json', 'w') as f:
        json.dump(data, f)

