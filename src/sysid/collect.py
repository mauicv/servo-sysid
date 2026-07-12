from sysid.data_interface import ActionDSInterface
from sysid.hardware import Controller
from sysid.config import CONTROL_HZ
import time
import json


if __name__ == '__main__':
    controller = Controller()
    ds = ActionDSInterface()
    dt = 1.0 / CONTROL_HZ

    data = {
        'config': ds.config,
        'data': [],
    }

    for rollout in ds.iter_rollouts():
        controller.center()

        rollout_data = {
            'type': rollout['type'],
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

    with open('data.json', 'w') as f:
        json.dump(data, f)

# -0.653 - low
# 0.940 - high


# -0.6608058608058598