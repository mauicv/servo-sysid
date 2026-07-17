from sysid.batched_job import BatchJob
from sysid.env import Env
from sysid.data_interface import SysidDSInterface
from cmaes import CMA
from sysid.nes import NES
import numpy as np
import time
import os
import json
from sysid.util import NumpyEncoder
import click
from datetime import datetime


# 4 workers is the throughput sweet spot on this machine (~4 performance cores);
# more workers just add fork/contention overhead without speeding up the sims.
NUM_WORKERS = int(os.environ.get("SYSID_WORKERS", 4))


@BatchJob(num_processes=NUM_WORKERS)
def job(args):
    # Every candidate is scored on the SAME rollouts (sampled once per generation
    # in the parent) so CMA-ES sees a consistent objective within a generation.
    params, keys, rollouts = args
    losses = []

    # Reuse a single Env across all rollouts (reset() reuses the MjData instead
    # of reallocating it per rollout).
    env = Env(params=params)
    for rollout in rollouts:
        actions = rollout['actions']
        real_states = rollout['states']

        env.reset(
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

    params_array = np.log(np.array([params[key] for key in keys]))
    fitness = np.mean(losses)
    return (params_array, fitness)



class ParameterSet:
    default_params = {"kp": 44.87924130639299, "kv": 2.313888096658114, "tau": 0.011233341812325085, "damping": 0.14305820613013226, "frictionloss": 0.00579036832562935, "armature": 0.011710338467774278, "force_limit": 2.19660269119488}
    def __init__(self, keys):
        self.keys = keys
        self.params = np.log(np.array([[self.default_params[key] for key in self.keys]]))

    def from_log_to_params(self, log_params):
        return {
            **self.default_params,
            **{key: np.exp(value) for key, value in zip(self.keys, log_params)}
        }

    def from_params_to_log(self, params):
        return np.log(np.array([params[key] for key in self.keys]))

    


class FrictionLossParameterSet(ParameterSet):
    def __init__(self):
        super().__init__(['frictionloss', 'damping', 'armature'])


class KpTauParameterSet(ParameterSet):
    def __init__(self):
        super().__init__(['kp', 'tau'])


class ForceLimitParameterSet(ParameterSet):
    def __init__(self):
        super().__init__(['force_limit'])


class AllParameterSet(ParameterSet):
    def __init__(self):
        super().__init__(['kp', 'kv', 'tau', 'damping', 'frictionloss', 'armature', 'force_limit'])


class FileSystemLogger:
    def __init__(self, experiment_name, param_set_name):
        self.experiment_name = experiment_name
        self.date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.param_set_name = param_set_name
        self.logs_dir = os.path.join('logs', experiment_name, self.date_time + '_' + self.param_set_name)
        os.makedirs(self.logs_dir, exist_ok=True)

    def log(self, logs):
        with open(os.path.join(self.logs_dir, 'history.json'), 'w') as f:
            json.dump(logs, f, cls=NumpyEncoder)

    def log_best_params(self, best_params):
        with open(os.path.join(self.logs_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, cls=NumpyEncoder)


def train(param_set_name, num_generations=100, population_size=30, alpha=0.5, name='experiment'):
    logger = FileSystemLogger(name, param_set_name)

    if param_set_name == 'frictionloss':
        param_set = FrictionLossParameterSet()
    elif param_set_name == 'kp_tau':
        param_set = KpTauParameterSet()
    elif param_set_name == 'force_limit':
        param_set = ForceLimitParameterSet()
    elif param_set_name == 'all':
        param_set = AllParameterSet()

    optimizer = NES(
        mean=param_set.params,
        sigma=np.sqrt(0.3),
        population_size=population_size,
        alpha=alpha
    )
    logs = []
    best_fitness = np.inf
    best_params = None

    ds = SysidDSInterface(filter_for=param_set.keys, filter_short=True)

    for generation in range(num_generations):
        start_time = time.perf_counter()
        # Sample one evaluation set for the whole generation; all candidates
        # are scored on these same rollouts.
        eval_rollouts = list(ds.sample(20, 50))
        param_list = []
        for _ in range(optimizer.population_size):
            params = param_set.from_log_to_params(optimizer.ask()[0].tolist())
            param_list.append((params, param_set.keys, eval_rollouts))
        results = job(param_list)
        optimizer.tell(results)
        end_time = time.perf_counter()

        print(f'Generation {generation}, time taken: {end_time - start_time} seconds, fitness: {np.mean([r[1] for r in results])}')

        logs.append({
            'generation': generation,
            'time_taken': end_time - start_time,
            'fitness': np.mean([r[1] for r in results]),
            'results': [(param_set.from_log_to_params(r[0]), r[1]) for r in results]
        })

        for params_array, fitness in results:
            if fitness < best_fitness:
                best_fitness = fitness
                best_params = param_set.from_log_to_params(params_array)
                print(f'new best fitness: {best_fitness}')

    logger.log(logs)
    logger.log_best_params(best_params)
        
