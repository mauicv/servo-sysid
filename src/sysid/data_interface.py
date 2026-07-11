import json
import os
from sysid.config import CONTROL_HZ, FREQ


def _validate_config_settings(config: dict):
    if config['action_hz'] != CONTROL_HZ:
        raise ValueError(f"Control Hz mismatch: {config['control_hz']} != {CONTROL_HZ}")
    print('control hz:', config['action_hz'], 'Hz')
    return config


class ActionDSInterface:
    DATA_DIR = os.path.dirname(__file__) + '/dataset/actions-dataset.json'

    def __init__(self):
        with open(self.DATA_DIR, 'r') as f:
            self.data = json.load(f)
        self.config = _validate_config_settings(self.data['config'])
        self.num_rollouts = len(self.data['data'])

    def iter_rollouts(self):
        for rollout in self.data['data']:
            yield rollout

    def __len__(self):
        return len(self.data['data'])


if __name__ == '__main__':
    ds = ActionDSInterface()
    for rollout in ds.iter_rollouts():
        print(rollout['type'], len(rollout['actions']))
