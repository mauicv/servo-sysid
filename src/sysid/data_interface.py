import json
import os
from sysid.config import CONTROL_HZ
from typing import Literal
import numpy as np
from scipy.signal import savgol_filter


def _validate_config_settings(config: dict):
    if config['action_hz'] != CONTROL_HZ:
        raise ValueError(f"Control Hz mismatch: {config['control_hz']} != {CONTROL_HZ}")
    # print('control hz:', config['action_hz'], 'Hz')
    return config

class DSInterface:
    DATA_DIR = os.path.dirname(__file__) + '/dataset/'

    def __init__(self, dataset_name: Literal['actions-dataset', 'real-action-state-dataset', 'sim-action-state-dataset']):
        self.dataset_name = dataset_name
        with open(self.DATA_DIR + self.dataset_name + '.json', 'r') as f:
            self.data = json.load(f)
        self.config = _validate_config_settings(self.data['config'])
        self.num_rollouts = len(self.data['data'])

    def iter_rollouts(self):
        for rollout in self.data['data']:
            yield rollout

    def __len__(self):
        return len(self.data['data'])


def compute_velocities(rollout):
    # times = np.array(rollout['times'])
    states = np.array(rollout['sensor_data'])
    # dt = np.mean(times[1:] - times[:-1])
    dt = 1 / CONTROL_HZ
    v = savgol_filter(states, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
    return v


class SysidDSInterface:
    DATA_DIR = os.path.dirname(__file__) + '/dataset/'

    def __init__(self, compute_velocities=True):
        dataset_name = 'real-action-state-dataset'

        with open(self.DATA_DIR + dataset_name + '.json', 'r') as f:
            self.data = json.load(f)
        self.config = _validate_config_settings(self.data['config'])
        self.num_rollouts = len(self.data['data'])
        self.index_weights = None
        self.compute_weights()
        if compute_velocities:
            self.esimate_velocities()

    def esimate_velocities(self):
        velocities = []
        for i, rollout in enumerate(self.data['data']):
            v = compute_velocities(rollout)
            self.data['data'][i]['velocities'] = v

    def compute_weights(self):
        type_weights = {'chirp': 0, 'step': 0, 'ramp': 0, 'prbs': 0, 'square': 0, 'triangle': 0}
        for rollout in self.data['data']:
            type_weights[rollout['type']] += len(rollout['sensor_data'])
        for rtype in type_weights:
            type_weights[rtype] = 1 / type_weights[rtype]

        index_weights = [
            type_weights[rollout['type']] * len(rollout['sensor_data'])
            for rollout in self.data['data']
        ]
        total = sum(index_weights)
        self.index_weights = [w / total for w in index_weights]

    def get_rollout(self, index):
        return self.data['data'][index]

    def sample_index(self, count):
        return np.random.choice(self.num_rollouts, p=self.index_weights, size=count)

    def sample_subset(self, rollout, length=50):
        states = np.array(rollout['sensor_data'])
        actions = np.array(rollout['actions'])
        velocities = np.array(rollout['velocities'])
        
        if len(states) == length:
            return states, actions, velocities
        elif len(states) < length:
            raise ValueError(f"Rollout length {len(states)} is greater than requested length {length}")
        
        start = np.random.randint(0, len(states) - length)
        end = start + length
        return states[start:end], actions[start:end], velocities[start:end]

    def sample(self, count, length):
        indices = self.sample_index(count)
        # initial state and velocity conditions need to be mapped back to correct q0, qd0 values 
        for index in indices:
            rollout = self.get_rollout(index)
            s, a, v = self.sample_subset(rollout, length)

            yield {
                'initial_states': s[0],
                'initial_velocities': v[0],
                'states': s,
                'velocities': v,
                'actions': a,
                'types': rollout['type'],
            }


if __name__ == '__main__':
    ds = SysidDSInterface()
    for data in ds.sample(10, 50):
        print('initial_states       ', data['initial_states'].shape)
        print('initial_velocities   ', data['initial_velocities'].shape)
        print('states               ', data['states'].shape)
        print('velocities           ', data['velocities'].shape)
        print('actions              ', data['actions'].shape)
        print('types                ', data['types'])
        break

