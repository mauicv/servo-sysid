import json
import os
from sysid.config import CONTROL_HZ
from typing import Literal
try:
    import numpy as np
except ImportError:
    print('numpy and scipy are not installed. Please install them to use this module.')


def _validate_config_settings(config: dict):
    if config['action_hz'] != CONTROL_HZ:
        raise ValueError(f"Control Hz mismatch: {config['control_hz']} != {CONTROL_HZ}")
    return config


class SysidDSInterface:
    DATA_DIR = os.path.dirname(__file__) + '/../dataset/'

    def __init__(self, dataset_name: Literal['actions', 'responses', 'dataset'], filter_for=None, filter_short=False):

        with open(self.DATA_DIR + dataset_name + '.json', 'r') as f:
            self.data = json.load(f)
        self.filter_for = filter_for
        if self.filter_for is not None:
            self.data['data'] = [
                rollout for rollout in self.data['data'] 
                if set(rollout['targets']).intersection(set(filter_for)) != set()
            ]
        if filter_short:
            self.data['data'] = [
                rollout for rollout in self.data['data'] 
                if len(rollout['sensor_data']) >= 50
            ]
        self.config = _validate_config_settings(self.data['config'])
        self.num_rollouts = len(self.data['data'])
        self.index_weights = None
        self.compute_weights()

    def __len__(self):
        return len(self.data['data'])

    def iter_rollouts(self):
        for rollout in self.data['data']:
            yield rollout

    def compute_weights(self):
        type_weights = {'chirp': 0, 'step': 0, 'ramp': 0, 'prbs': 0, 'square': 0, 'triangle': 0, 'drop': 0}
        for rollout in self.data['data']:
            type_weights[rollout['type']] += len(rollout['sensor_data'])
        for rtype in type_weights:
            if type_weights[rtype] != 0:
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
    ds = SysidDSInterface(filter_for=['kp', 'tau'])
    count = 0
    for data in ds.sample(10, 12):
        print('--------------------------------')
        print('initial_states       ', data['initial_states'].shape)
        print('initial_velocities   ', data['initial_velocities'].shape)
        print('states               ', data['states'].shape)
        print('velocities           ', data['velocities'].shape)
        print('actions              ', data['actions'].shape)
        print('types                ', data['types'])
        count += 1
        if count > 10:
            break

