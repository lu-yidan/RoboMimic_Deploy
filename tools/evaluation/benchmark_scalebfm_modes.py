"""Warm CUDA-synchronized inference timing for all released masks; no DDS."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco,numpy as np,torch
from omegaconf import OmegaConf
from common.ctrlcomp import StateAndCmd,PolicyOutput
from FSM.FSM import FSM
from common.path_config import PROJECT_ROOT
root=Path(PROJECT_ROOT); torch.set_num_threads(1)
from common.deployment_config import load_scalebfm_config
cfg=OmegaConf.load(root/'deploy_mujoco/config/mujoco.yaml')
cfg.scalebfm = load_scalebfm_config(require_enabled=True)
m=mujoco.MjModel.from_xml_path(str(root/'g1_description/scene.xml'))
names=[m.joint(int(m.actuator_trnid[i,0])).name for i in range(29)]
s,o=StateAndCmd(29),PolicyOutput(29);fsm=FSM(s,o,cfg.scalebfm,names,enable_freekick=False)
t=fsm.scalebfm_policy;s.q=fsm.fixed_pose_1._default_out.copy();t.enter()
report={}
with torch.inference_mode():
 for mode in range(8):
  t.agent.control_mode.fill_(mode)
  for _ in range(10): t.agent.get_action(t.obs)
  torch.cuda.synchronize(); samples=[]
  for _ in range(100):
   start=time.perf_counter(); t.agent.get_action(t.obs);torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
  report[mode]=dict(median_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),max_ms=max(samples))
out=root/'outputs/reactive_latency_modes/inference.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
