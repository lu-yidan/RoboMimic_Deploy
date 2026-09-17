"""Read-only log audit; legacy bridge tau_est zeros are unavailable, not zero torque."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.logger import Logger
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+');p.add_argument('--out',type=Path,required=True);a=p.parse_args();result={}
for name in a.logs:
 path=Path(name).with_suffix('.bin');d=Logger.load(str(path));t=d['time_s'].astype(float);dt=np.diff(t)
 legacy_bridge=d['_meta'].get('robot_type')=='real_policy_bridge' and d['_meta'].get('logger_schema_version',1)<3
 tau_valid=d.get('tau_est_valid',np.zeros(len(t)))>0
 v={'frames':len(t),'span_s':float(t[-1]-t[0]),'frame_dt_percentiles_s':dict(zip(['p50','p95','p99','max'],np.percentile(dt,[50,95,99,100]).tolist())),
    'gaps_over_100ms':np.where(dt>.1)[0].tolist(),'firmware_torque_available':bool(tau_valid.any()) and not legacy_bridge,
    'executed_policy_identifiable':'executed_fsm_state' in d,'profile':d['_meta'].get('smp_recovery',{}).get('profile'),
    'joint_speed_abs_max_rad_s':float(abs(d['dq']).max()),'host_pd_torque_abs_max_Nm':float(abs(d['tau_cmd_est']).max()),
    'host_pd_power_abs_max_W':float(abs(d['tau_cmd_est']*d['dq']).max()),
    'waist_yaw_roll_pitch_min_deg':np.rad2deg(d['q'][:,12:15].min(0)).tolist(),
    'waist_yaw_roll_pitch_max_deg':np.rad2deg(d['q'][:,12:15].max(0)).tolist(),
    'warnings':['PD quantities are host estimates, not measured torque/power; no foot forces or external forces.']}
 if legacy_bridge:v['warnings'].append('Legacy bridge did not forward tau_est; zeros are placeholders. No per-frame FSM ID, mixed policies cannot be separated reliably.')
 result[str(path)]=v
 print(path.name, json.dumps(v,ensure_ascii=False))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
