"""Quick sanity check / viewer for the servo model via the sysid Env.

Usage:
    ./venv/bin/python test.py          # build Env, print model info, step a bit
    ./venv/bin/python test.py --view   # launch interactive viewer, drive the servo
"""

import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import mujoco
import mujoco.viewer

from sysid.env import Env

# Reasonable defaults matching the actuator in robot.xml (kp=25, kv=5, ...).
DEFAULT_PARAMS = {
    "kp": 55.16425524424094,
    "kv": 2.8441638016834414,
    "tau": 0.013807696318891036,
    "damping": 1.5439016340657283,
    "frictionloss": 0.0,
    "armature": 0.12637940307607023,
    "force_limit": 8.40629855285546
}
# DEFAULT_PARAMS = {
#     "kp": 25.0,
#     "kv": 5.0,
#     "tau": 0.1,
#     "damping": 0.1,
#     "frictionloss": 0.1,
#     "armature": 0.005,
#     "force_limit": 2.70,
# }

CONTROL_HZ = 50


def main() -> None:
    env = Env(params=DEFAULT_PARAMS)
    model = env.model

    print("Env built OK")
    print(f"  nbody={model.nbody}  njnt={model.njnt}  ngeom={model.ngeom}")
    print(f"  nmesh={model.nmesh}  nu={model.nu}  nsensor={model.nsensor}")
    print(f"  physics_dt={model.opt.timestep}  n_substeps/ctrl={env.n_substeps}")
    for i in range(model.nmesh):
        print("  mesh:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i))

    if "--view" in sys.argv:
        view(env)
    else:
        # Drive a slow sine on the action and confirm the servo responds.
        env.reset()
        obs = None
        for i in range(CONTROL_HZ):  # 1 second of control steps
            action = 0.5 * math.sin(2 * math.pi * i / CONTROL_HZ)
            obs = env.step(action)
        print(f"  stepped {CONTROL_HZ} control steps, last action={action:+.3f}, "
              f"sensor={obs[0]:+.4f}")


def view(env: Env) -> None:
    """Drive the servo with a slow sine sweep, synced to the passive viewer."""
    env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            action = math.sin(2 * math.pi * 0.25 * t)  # 0.25 Hz sweep, +/- 1
            action = None
            env.step(action)
            viewer.sync()
            time.sleep(1.0 / CONTROL_HZ)


if __name__ == "__main__":
    main()
