import json
import os
import numpy as np
import math


path = os.getcwd().split('/')
dirname = '/src/sysid/dataset'
response_filename = '/'.join(path) + dirname + '/env-actions.json'

with open(response_filename, 'r') as f:
    response_data = json.load(f)

env_actions = np.array(response_data['actions'])

response_filename = '/'.join(path) + dirname + '/real-actions.json'

with open(response_filename, 'r') as f:
    response_data = json.load(f)

real_actions = np.array(response_data['actions'])

SERVO_UNIT_RAD = 0.75 * math.pi
ACTION_SCALE = (0.25 / SERVO_UNIT_RAD)

scaled_env_actions = (env_actions / ACTION_SCALE) * 0.25 
scaled_real_actions = (real_actions / ACTION_SCALE) * 0.25 

data = []
for i in range(10):
    data.append({'env_actions': scaled_env_actions[:, i].tolist(), 'real_actions': env_actions[:, i].tolist(), 'type': 'recorded'})

for i in range(10):
    data.append({'env_actions': scaled_real_actions[:, i].tolist(), 'real_actions': real_actions[:, i].tolist(), 'type': 'live'})

for level in range(-50, 55, 5):
    env_actions = (level/100) * 0.75 * math.pi
    real_actions = (level/100)
    data.append({'env_actions': [env_actions] * 100, 'real_actions': [real_actions] * 100, 'type': 'step'})


filename = '/'.join(path) + dirname + '/actions.json'

with open(filename, 'w') as f:
    json.dump({
        'data': data,
        'config': {'action_hz': 50,}
    }, f)
