"""Single-state check of axis-specific conservative cones at recorded failures."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.load_transfer_residual_env import LoadTransferResidualEnv
from training.axis_robust_contact_controller import AxisRobustContactController


def main():
    results=[]
    for name in ['nominal','row01_z_+5mm','row01_x_-10mm']:
        z=dict(np.load(ROOT/'outputs/robust_allocation_diagnostic_v1'/(name+'.npz')))
        task=LoadTransferResidualEnv(ROOT/'outputs/clearance_back20_leg14_grid_v1',[name]);task.reset(options={'case':name})
        e=task.env;e.data.qpos[:]=z['qpos'];e.data.qvel[:]=z['qvel'];e.data.time=float(z['time_s']);mujoco.mj_forward(e.model,e.data)
        q,v,active=task.reference(e.data.time)
        c=AxisRobustContactController(e,torque_weight=10000.,normal_bias_y_deg=3.)
        try:torque,info=c.command(q,v,active);result=dict(feasible=True,torque_peak_ratio=info['torque_ratio'])
        except RuntimeError as exc:result=dict(feasible=False,reason=str(exc))
        results.append(dict(case=name,**result));task.close()
    out=ROOT/'outputs/axis_uncertainty_failure_probe_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
