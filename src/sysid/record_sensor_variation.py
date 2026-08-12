from sysid.hardware import Controller
from sysid.config import CONTROL_HZ
import time
from tqdm import tqdm


if __name__ == '__main__':
    controller = Controller()
    dt = 1.0 / CONTROL_HZ
    controller.center()
    previous_sensor_data = controller.get_sensor_data()

    action = 0.0
    recording = []
    for _ in tqdm(range(200)):
        start_time = time.time()
        sensor_data = controller.get_sensor_data()
        recording.append(sensor_data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < dt:
            time.sleep(dt - elapsed_time)

    print('range:', max(recording) - min(recording))


# above measure range: 0.021001221001220927 slack. 