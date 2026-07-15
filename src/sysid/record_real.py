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
    controller.center()

    rollout_data = {
        'type': 'drop',
        'targets': ['frictionloss'],
        'actions': [],
        'sensor_data': [],
    }

    previous_sensor_data = controller.get_sensor_data()

    action = 0.0
    while True:
        start_time = time.time()
        controller.send_action([action] * 16)
        sensor_data = controller.get_sensor_data()
        rollout_data['actions'].append(action)
        rollout_data['sensor_data'].append(sensor_data)
        diff = sensor_data - previous_sensor_data
        previous_sensor_data = sensor_data

        if diff > 0.01:
            action = None
            count_down = 100

        if action is None:
            count_down -= 1
            if count_down <= 0:
                break

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < dt:
            time.sleep(dt - elapsed_time)

    data['data'].append(rollout_data)

    with open('src/sysid/dataset/drop-response.json', 'w') as f:
        json.dump(data, f)

