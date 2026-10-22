"""Explicit confirmation-channel fault injection, without modifying physics forces."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from training.climb_robustness import ReleaseInterlock
import deploy_mujoco.run_climb_cycle as runner


class DropoutInterlock(ReleaseInterlock):
    def sample(self,t):
        super().sample(t)
        if t>=14.:
            # The right-hook confirmation channel is unavailable. Raw contact
            # forces and saved physics state remain untouched and observable.
            self.history[-1][1][3]=False


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=OmegaConf.load(ROOT/'deploy_mujoco/config/mujoco_climb_cycle_guarded.yaml').climb_cycle
    cfg.output_dir=str(a.out.resolve())
    cfg.confirmation_dropout=dict(group='right_hook',from_s=14.,kind='invalid confirmation bit only; true contact remains present')
    runner.ReleaseInterlock=DropoutInterlock
    runner.run_climb_cycle(cfg)
    manifest_path=a.out/'manifest.json';manifest=json.loads(manifest_path.read_text())
    target=a.out/'source_snapshot/tools/run_contact_dropout.py';target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(__file__,target)
    manifest['sources']['tools/run_contact_dropout.py']=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest['fault_injection']=OmegaConf.to_container(cfg.confirmation_dropout)
    manifest_path.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
