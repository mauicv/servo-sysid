"""Quick sanity check / viewer for the merged robot.xml model.

Usage:
    ./venv/bin/python test.py          # compile + print model info
    ./venv/bin/python test.py --view   # launch interactive viewer
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

XML = Path(__file__).parent / "src" / "sysid" / "desc" / "robot.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    print(f"compiled OK: {XML.name}")
    print(f"  nbody={model.nbody}  njnt={model.njnt}  ngeom={model.ngeom}")
    print(f"  nmesh={model.nmesh}  nu={model.nu}  nsensor={model.nsensor}")
    for i in range(model.nmesh):
        print("  mesh:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i))

    # # Step once to confirm the model is dynamically valid.
    # mujoco.mj_step(model, data)
    # print(f"  stepped once, time={data.time:.4f}")

    if "--view" in sys.argv:
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
