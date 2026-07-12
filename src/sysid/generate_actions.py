import numpy as np
import json
from tqdm import tqdm
import os
from sysid.config import CONTROL_HZ


def generate_chirp(config, seconds=5, freq_low=0.5, freq_high=10):
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


def generate_step(config, seconds=1, amplitude=1.0):
    action_hz = config['action_hz']
    t = np.linspace(0, seconds, int(seconds * action_hz), endpoint=False)
    action = np.zeros((len(t),))
    action[:] = amplitude
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


def generate_ramp(config, seconds=2, amplitude=1.0):
    action_hz = config['action_hz']
    n = int(seconds * action_hz)
    ramp = np.linspace(0, amplitude, n)
    action = np.zeros((n, ))
    action[:] = ramp
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
    print("Generating chirp rollouts...")
    
    actions_hardware = generate_chirp(config, 3)
    rollout = {
        'type': 'chirp',
        'actions': actions_hardware.tolist(),
    }
    dataset['data'].append(rollout)

    print("Generating step rollouts...")
    for amplitude in [-1, -0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 1]:
        actions_hardware = generate_step(config, 1, amplitude)
        rollout = {
            'type': 'step',
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    print("Generating prbs rollouts...")
    for _ in range(3):
        actions_hardware = generate_prbs(
            config,
            seed=np.random.randint(0, 1000000)
        )
        rollout = {
            'type': 'prbs',
            'actions': actions_hardware.tolist(),
        }
        dataset['data'].append(rollout)

    print("Generating ramp rollout...")
    actions_hardware = generate_ramp(
        config,
        amplitude=0.4
    )
    rollout = {
        'type': 'ramp',
        'actions': actions_hardware.tolist(),
    }
    dataset['data'].append(rollout)

    filename = os.path.dirname(__file__) + '/dataset/actions-dataset.json'
    print('saving dataset to', filename)
    with open(filename, 'w') as f:
        json.dump(dataset, f)

