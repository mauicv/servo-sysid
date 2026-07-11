import smbus2 as smbus
import time
import struct
import math
import threading as th

I2C_MUX_ADDR_1 = 0x70
I2C_MUX_ADDR_2 = 0x71
ROT_ENC_ADDR = 0x36
PWM_MUX_ADDR = 0x40
PCA9685_LED0 = 0x06
NUM_SERVOS = 1

bus_lock = th.Lock()

device_map = {
    I2C_MUX_ADDR_1: {
        "rot_encs": [0,1,2,3,4,5,6,7],
    },
    I2C_MUX_ADDR_2: {
        "rot_encs": [0,1,2,3,4,5,6,7],
    },
}

def init_pca9685(bus, addr=PWM_MUX_ADDR, freq=50):
    bus.write_byte_data(addr, 0x00, 0x10)  # sleep
    prescale = round(25_000_000 / (4096 * freq)) - 1
    bus.write_byte_data(addr, 0xFE, prescale)
    bus.write_byte_data(addr, 0x00, 0x20)  # wake + auto-increment
    time.sleep(0.005)  # oscillator settle

def _read(bus, addr, reg, data_len):
    try:
        return bus.read_i2c_block_data(addr, reg, data_len)
    except IOError as e:
        # print(f"Error reading device from {addr}: {e}")
        return None

def read_sensor_data(bus):
    rot_enc_data = []
    for mux_addr in [0x70, 0x71]:
        for i in device_map[mux_addr]["rot_encs"]:
            bus.write_byte(mux_addr, 1 << i)
            dev_data = _read(bus, ROT_ENC_ADDR, 0x0C, 2)
            rot_enc_data.append(dev_data)

    return rot_enc_data

def write_servos(bus, data, freq):
    period_us = 1_000_000 / freq
    ticks_per_us = 4096 / period_us
    byte_data = []
    for pwm_us in data:
        on = 0
        off = min(4095, round(pwm_us * ticks_per_us))
        byte_data += [on & 0xFF, on >> 8, off & 0xFF, off >> 8]
    chunk = 32  # 8 channels * 4 bytes
    with bus_lock:
        for i in range(0, len(byte_data), chunk):
            reg = PCA9685_LED0 + (i // 4) * 4
            bus.write_i2c_block_data(PWM_MUX_ADDR, reg, byte_data[i:i + chunk])

def decode_angle(raw):
    if raw is None:
        return None
    return (((raw[0] & 0x0F) << 8) | raw[1]) / 4095 * 2 - 1  # -1 to 1

def test_rot_encs(bus):
    for mux_addr in [I2C_MUX_ADDR_1, I2C_MUX_ADDR_2]:
        for channel in device_map[mux_addr]["rot_encs"]:
            try:
                bus.write_byte(mux_addr, 1 << channel)
                status = bus.read_i2c_block_data(ROT_ENC_ADDR, 0x0B, 1)[0]
                magnet_high = (status >> 3) & 1
                magnet_low = (status >> 4) & 1
                magnet_detected = (status >> 5) & 1
                print(f"{mux_addr:02x}:{channel:02x}: detected: {magnet_detected}, too strong: {magnet_high}, too weak: {magnet_low}")
            except IOError as e:
                print(f"{mux_addr:02x}:{channel:02x}: error: {e}")
        bus.write_byte(mux_addr, 0)


class Controller:
    def __init__(self):
        self.bus = smbus.SMBus(1)
        init_pca9685(self.bus, PWM_MUX_ADDR, freq=100)
        
    def get_sensor_data(self):
        sensor_data = read_sensor_data(self.bus)
        sensor_data = [decode_angle(item) for item in sensor_data]
        return sensor_data
    
    def send_action(self, action):
        write_servos(self.bus, action, 100)


if __name__ == "__main__":
    import os
    bus = smbus.SMBus(1)
    test_rot_encs(bus)    
    init_pca9685(bus, PWM_MUX_ADDR, freq=100)

    def display(elapsed, rot_enc_data):
        rot_line = f"  ROT  | " + " ".join(f"{v:5.3f}" if v is not None else "  N/A" for v in rot_enc_data)
        hz_line = f"  TIME | {elapsed:.1f}ms ({1000/elapsed:.0f} Hz)"

        print(f"\033[4A{rot_line:<80}\n{hz_line:<80}", flush=True)

    # Print initial blank lines for the cursor to overwrite
    print("\n\n\n\n")

    while True:
        start = time.perf_counter()
        rot_enc_data = read_sensor_data()
        elapsed = (time.perf_counter() - start) * 1000
        rot_enc_data = [decode_angle(item) for item in rot_enc_data]
        display(elapsed, rot_enc_data)
        time.sleep(0.1)
