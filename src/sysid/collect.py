from sysid.data_interface import ActionDSInterface
from sysid.hardware import Controller
from sysid.config import CONTROL_HZ
import smbus
import time



if __name__ == '__main__':
    controller = Controller()
    ds = ActionDSInterface()

    dt = 1.0 / CONTROL_HZ

    for rollout in ds.iter_rollouts():
        for action in rollout['actions']:
            start_time = time.time()
            controller.send_action(action)
            sensor_data = controller.get_sensor_data()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
            print(f"Sensor data: {sensor_data}")
            print(f"Action: {action}")
            if elapsed_time < dt:
                time.sleep(dt - elapsed_time)
