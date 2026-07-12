from sysid.data_interface import ActionDSInterface
from sysid.hardware import Controller
from sysid.config import CONTROL_HZ
import smbus2 as smbus
import time

def display(elapsed, rot_enc_data):
    rot_line = f"  ROT  | " + " ".join(f"{rot_enc_data:5.3f}")
    hz_line = f"  TIME | {elapsed:.1f}ms ({1000/elapsed:.0f} Hz)"

    print(f"\033[4A{rot_line:<80}\n{hz_line:<80}", flush=True)

def convert_to_pwm(action):
    pass


if __name__ == '__main__':
    controller = Controller()
    controller.center()
    # ds = ActionDSInterface()

    dt = 1.0 / CONTROL_HZ

    while True:
        start_time = time.time()
        sensor_data = controller.get_sensor_data()

        controller.send_action([0]*16)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(sensor_data)
        # display(elapsed_time, sensor_data)
        if elapsed_time < dt:
            time.sleep(dt - elapsed_time)

    # for rollout in ds.iter_rollouts():
    #     for action in rollout['actions']:
    #         start_time = time.time()
    #         controller.send_action(action)
    #         sensor_data = controller.get_sensor_data()
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time
    #         print(f"Elapsed time: {elapsed_time} seconds")
    #         print(f"Sensor data: {sensor_data}")
    #         print(f"Action: {action}")
    #         if elapsed_time < dt:
    #             time.sleep(dt - elapsed_time)

# -0.653 - low
# 0.940 - high


# -0.6608058608058598