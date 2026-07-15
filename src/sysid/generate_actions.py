import numpy as np
import json
from tqdm import tqdm
import os
from sysid.config import CONTROL_HZ


def generate_chirp(config, seconds=5, freq_low=0.5, freq_high=3):
    joint_limits = config['joint_limits']
    joint_range = joint_limits[1] - joint_limits[0]
    amplitude = joint_range / 2
    action_hz = config['action_hz']
    t = np.linspace(0, seconds, int(seconds * action_hz), endpoint=False)
    k = (freq_high - freq_low) / seconds
    phase = 2 * np.pi * (freq_low * t + 0.5 * k * t**2)
    action = np.zeros((len(t),))
    amplitude_signal = np.ones((len(t),)) * amplitude
    action = np.sin(phase)
    action *= amplitude_signal
    return action


def generate_prbs(config, seconds=3, amplitude=0.2, min_hold=0.05, seed=0):
    action_hz = config['action_hz']
    rng = np.random.default_rng(seed)
    n = int(seconds * action_hz)
    hold_steps = int(min_hold * action_hz)

    signal = np.zeros(n)
    i = 0
    val = amplitude
    while i < n:
        val = amplitude if rng.random() > 0.5 else -amplitude
        run_len = hold_steps + rng.integers(0, hold_steps)  # jittered hold
        signal[i:i+run_len] = val
        i += run_len

    action = np.zeros((n,))
    action[:] = signal[:n]
    return action


def generate_step(config, seconds=1, amplitude=0.625):
    action_hz = config['action_hz']
    t = np.linspace(0, seconds, int(seconds * action_hz), endpoint=False)
    action = np.zeros((len(t),))
    action[:] = amplitude
    return action


def generate_ramp(config, seconds=2, amplitude=0.625):
    action_hz = config['action_hz']
    n = int(seconds * action_hz)
    ramp = np.linspace(0, amplitude, n)
    action = np.zeros((n, ))
    action[:] = ramp
    return action


def generate_triangle(config, seconds=2, amplitude=0.2, period=0.5):
    action_hz = config['action_hz']
    n = int(seconds * action_hz)
    t = np.arange(n) / action_hz

    phase = (t % period) / period  # 0 -> 1 sawtooth phase
    triangle = amplitude * (2 * np.abs(2 * phase - 1) - 1)

    action = np.zeros((n,))
    action[:] = triangle
    return action


def generate_square(config, seconds=2, amplitude=0.2, period=0.5):
    action_hz = config['action_hz']
    n = int(seconds * action_hz)
    t = np.arange(n) / action_hz

    phase = (t % period) / period
    square = np.where(phase < 0.5, amplitude, -amplitude)

    action = np.zeros((n,))
    action[:] = square
    return action


if __name__ == "__main__":
    
    config = {
        'action_hz': CONTROL_HZ,
        'joint_limits': [-0.625, 0.625],
    }

    dataset={
        'config': config,
        'data': [],
    }
    
    actions_hardware = generate_chirp(config, 3)
    rollout = {
        'type': 'chirp',
        'targets': [],
        'actions': actions_hardware.tolist(),
    }
    dataset['data'].append(rollout)

    for amplitude in [-0.15, -0.1, -0.05, 0.05, 0.1, 0.15]:
        actions_hardware = generate_step(config, 1, amplitude)
        rollout = {
            'type': 'step',
            'targets': ['kp', 'tau'],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude in [-0.6, -0.55, 0.55, 0.6]:
        actions_hardware = generate_step(config, 1, amplitude)
        rollout = {
            'type': 'step',
            'targets': ['force_limit'],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude in [0.15, 0.1, 0.05]:
        actions_hardware = generate_prbs(
            config,
            seed=np.random.randint(0, 1000000),
            amplitude=amplitude
        )
        rollout = {
            'type': 'prbs',
            'targets': ['kp', 'tau'],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude in [0.6, 0.6, 0.55, 0.55]:
        actions_hardware = generate_prbs(
            config,
            seed=np.random.randint(0, 1000000),
            amplitude=amplitude
        )
        rollout = {
            'type': 'prbs',
            'targets': ['force_limit'],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude in [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6]:
        actions_hardware = generate_ramp(
            config,
            amplitude=amplitude
        )
        rollout = {
            'type': 'ramp',
            'targets': [],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude, period in [(0.1, 1), (0.1, 0.75), (0.3, 1), (0.3, 0.75), (0.6, 1), (0.6, 0.75)]:
        actions_hardware = generate_triangle(
            config,
            amplitude=amplitude,
            period=period
        )
        rollout = {
            'type': 'triangle',
            'targets': [],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude, period in [(0.1, 1), (0.1, 0.75)]:
        actions_hardware = generate_square(
            config,
            amplitude=amplitude,
            period=period
        )
        rollout = {
            'type': 'square',
            'targets': ['kp', 'tau'],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    for amplitude, period in [(0.6, 1), (0.6, 0.75)]:
        actions_hardware = generate_square(
            config,
            amplitude=amplitude,
            period=period
        )
        rollout = {
            'type': 'square',
            'targets': ['force_limit'],
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)


    filename = os.path.dirname(__file__) + '/dataset/actions-dataset.json'
    print('saving dataset to', filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f)

