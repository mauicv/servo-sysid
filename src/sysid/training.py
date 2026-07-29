"""Train the actuator network: history of (position error, velocity) -> torque.

Hwangbo et al. style. The input window is the current state plus HISTORY-1
previous states; the target is the torque at the current (last) state.

A fixed fraction of whole rollouts is held out for validation (split at the
rollout level so no window leaks between train and val).

Run:  python -m sysid.training
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sysid.data_pipeline.data_interface import SysidDSInterface

HISTORY = 3                 # samples per window: current + (HISTORY-1) previous
INPUT_SIZE = 2 * HISTORY    # (position error, velocity) per state
VAL_FRAC = 0.15             # fraction of rollouts held out for validation
OUT_PATH = Path(__file__).parent / "actuator_net.pt"


class ActuatorModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


def load_actuator_net(path=OUT_PATH):
    """Load a trained ActuatorModel checkpoint (eval mode)."""
    ckpt = torch.load(path, weights_only=False)
    model = ActuatorModel(input_size=ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _tensor(arr):
    # torch.from_numpy is broken here (torch 2.2.2 vs numpy 2 ABI); go via lists
    # until torch is upgraded. See training TODO.
    return torch.tensor(np.asarray(arr).tolist(), dtype=torch.float32)


def _window_features(states, actions, velocities, sl):
    """features for one window (indices `sl`): pos_err then velocity over the
    window; target is the current-state torque (caller supplies it)."""
    pos_err = actions[sl] - states[sl]
    return np.concatenate([pos_err, velocities[sl]])


def make_batch(dataset, count):
    """Draw `count` random windows -> (features, target) tensors.

    features: (count, 2*HISTORY)   target: (count, 1) torque at current state.
    """
    features, target = [], []
    for s in dataset.sample(count, HISTORY):
        sl = slice(0, HISTORY)
        features.append(_window_features(s["states"], s["actions"], s["velocities"], sl))
        target.append(s["torque"][-1])              # current (last) state
    return _tensor(features), _tensor(np.reshape(target, (-1, 1)))


def build_all_windows(rollouts):
    """Every sliding window in `rollouts` -> fixed (features, target) tensors."""
    features, target = [], []
    for r in rollouts:
        states = np.asarray(r["sensor_data"], dtype=np.float64)
        actions = np.asarray(r["actions"], dtype=np.float64)
        velocities = np.asarray(r["velocities"], dtype=np.float64)
        torque = np.asarray(r["torque"], dtype=np.float64)
        for t in range(HISTORY - 1, len(states)):
            sl = slice(t - HISTORY + 1, t + 1)
            features.append(_window_features(states, actions, velocities, sl))
            target.append(torque[t])
    return _tensor(features), _tensor(np.reshape(target, (-1, 1)))


def split_rollouts(dataset, val_frac=VAL_FRAC, seed=0):
    """Split rollouts into (train_view, val_rollouts). Mutates `dataset` in place
    to sample from the train rollouts only."""
    rollouts = list(dataset.iter_rollouts())
    perm = np.random.default_rng(seed).permutation(len(rollouts))
    n_val = int(len(rollouts) * val_frac)
    val = [rollouts[i] for i in perm[:n_val]]
    train = [rollouts[i] for i in perm[n_val:]]
    dataset.data["data"] = train
    dataset.num_rollouts = len(train)
    dataset.compute_weights()
    return dataset, val


def train_model(model, dataset, val_features, val_target, num_epochs, batch_size=1024):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        features, target = make_batch(dataset, batch_size)
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_rmse = loss_fn(model(val_features), val_target).item() ** 0.5
            print(f"epoch {epoch:5d}  train_rmse={loss.item() ** 0.5:.4f}  "
                  f"val_rmse={val_rmse:.4f} N.m")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = SysidDSInterface(dataset_name="dataset")
    dataset, val_rollouts = split_rollouts(dataset)
    val_features, val_target = build_all_windows(val_rollouts)
    print(f"train rollouts={dataset.num_rollouts}  "
          f"val rollouts={len(val_rollouts)}  val windows={len(val_target)}")

    model = ActuatorModel(input_size=INPUT_SIZE)
    train_model(model, dataset, val_features, val_target, num_epochs=1000)

    torch.save(
        {"state_dict": model.state_dict(), "input_size": INPUT_SIZE, "history": HISTORY},
        OUT_PATH,
    )
    print(f"saved -> {OUT_PATH}")
