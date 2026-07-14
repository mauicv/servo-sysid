from sysid.batched_job import BatchJob
from sysid.env import Env
from sysid.data_interface import SysidDSInterface
from cmaes import CMA
from sysid.nes import NES
import numpy as np
import time
import os
import json


keys = [
    'kp',
    'kv',
    'tau',
    'damping',
    'frictionloss',
    'armature',
    'force_limit'
]
# initial_params = np.log(np.array([
#     25,
#     5,
#     0.1,
#     0.1,
#     # 0.1,
#     0.005,
#     2.70
# ]))

initial_params = np.log(np.array([
    25,
    5,
    1,
    1,
    0.1,
    1,
    5
]))
# keys = ['tau']
# initial_params = np.log(np.array([1.0]))
# keys = ['damping', 'frictionloss', 'armature']
# initial_params = np.log(np.array([0.1, 0.1, 0.005]))
batch_job = BatchJob(num_processes=os.cpu_count() - 1)


def from_log_to_params(log_params):
    return {key: np.exp(value) for key, value in zip(keys, log_params)}

def from_params_to_log(params):
    return np.log(np.array([params[key] for key in keys]))


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


@batch_job
def job(args):
    # Every candidate is scored on the SAME rollouts (sampled once per generation
    # in the parent) so CMA-ES sees a consistent objective within a generation.
    params, rollouts = args
    losses = []

    for rollout in rollouts:
        actions = rollout['actions']
        real_states = rollout['states']

        env = Env(
            params=params,
            initial_states=rollout['initial_states'],
            initial_velocities=rollout['initial_velocities']
        )
        sim_states = []
        for action in actions:
            sim_states.append(env.step(action))

        sim_states = np.array(sim_states)[:, 0]
        real_states = np.array(real_states)

        loss = np.mean(np.square(sim_states - real_states))
        losses.append(loss)

    params_array = from_params_to_log(params)
    fitness = np.mean(losses)
    return (params_array, fitness)


if __name__ == "__main__":
    # optimizer = CMA(mean=initial_params, sigma=np.sqrt(0.1), population_size=20)
    optimizer = NES(mean=initial_params, sigma=np.sqrt(0.3), population_size=30, alpha=0.5)
    logs = []
    best_fitness = np.inf
    best_params = None

    ds = SysidDSInterface()

    for generation in range(100):
        start_time = time.perf_counter()
        # Sample one evaluation set for the whole generation; all candidates
        # are scored on these same rollouts.
        eval_rollouts = list(ds.sample(20, 35))
        param_list = []
        for _ in range(optimizer.population_size):
            params = from_log_to_params(optimizer.ask())
            param_list.append((params, eval_rollouts))
        results = job(param_list)
        optimizer.tell(results)
        end_time = time.perf_counter()

        print(f'Generation {generation}, time taken: {end_time - start_time} seconds, fitness: {np.mean([r[1] for r in results])}')

        logs.append({
            'generation': generation,
            'time_taken': end_time - start_time,
            'fitness': np.mean([r[1] for r in results]),
            'results': [(from_log_to_params(r[0]), r[1]) for r in results]
        })

        for params_array, fitness in results:
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = from_log_to_params(params_array)
                print(f'new best fitness: {best_fitness}')

    with open('logs/history.json', 'w') as f:        
        json.dump(logs, f, cls=NumpyEncoder)

    with open('logs/best_params.json', 'w') as f:
        json.dump(best_params, f, cls=NumpyEncoder)
        
