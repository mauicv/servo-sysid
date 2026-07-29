"""Roll out a logged rollout in sim with the learned actuator net and plot it
against the real measured response.

Run:  python -m sysid.rollout [rollout_index]
"""

import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from sysid.config import CONTROL_HZ
from sysid.env import Env
from sysid.training import HISTORY, load_actuator_net
from sysid.data_pipeline.data_interface import SysidDSInterface


def make_actuator(net):
    """Wrap a torch ActuatorModel as a features->torque(N.m) callable for Env."""
    def actuator(features):
        with torch.no_grad():
            x = torch.tensor([features.tolist()], dtype=torch.float32)
            return float(net(x)[0, 0])
    return actuator


def simulate(env, actions):
    return np.array([env.step(a) for a in actions])


def test_rollout(index=0, dataset_name="dataset", net=None, save_path=None):
    """Simulate rollout `index` with the actuator net and plot sim vs real."""
    ds = SysidDSInterface(dataset_name=dataset_name)
    rollout = ds.get_rollout(index)
    real = np.asarray(rollout["sensor_data"])
    actions = np.asarray(rollout["actions"])
    velocities = np.asarray(rollout["velocities"])

    if net is None:
        net = load_actuator_net()
    env = Env(
        make_actuator(net),
        history=HISTORY,
        initial_states=[real[0]],
        initial_velocities=[velocities[0]],
    )
    sim = simulate(env, actions)

    t = np.arange(len(real)) / CONTROL_HZ
    plt.figure(figsize=(9, 4))
    plt.plot(t, real, label="real", linewidth=2)
    plt.plot(t, sim, label="sim (actuator net)", linewidth=2, linestyle="--")
    plt.plot(t, actions, label="target (action)", color="gray", alpha=0.4)
    plt.xlabel("time (s)")
    plt.ylabel("position (normalized, 1.0 = π rad)")
    plt.title(f"rollout {index}  type={rollout['type']}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"saved plot -> {save_path}")
    else:
        plt.show()
    return sim, real


if __name__ == "__main__":
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_rollout(index)
