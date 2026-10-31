"""Controlled geometry-estimate errors, separate from actual compiled geometry.

This is an observation ablation, not a camera or contact estimator. Unlisted
rod features remain oracle inputs, as do the original whole-body feedback terms.
"""
from collections.abc import Mapping
import numpy as np
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from training.cycle_clock import cycle_index


class EstimatedBarObservation(CycleRelativeBarObservation):
    def __init__(self,env,*,estimated_offsets):
        super().__init__(env,mode='oracle')
        if not isinstance(estimated_offsets,Mapping):raise ValueError('Expected explicit rod-to-offset mapping')
        available={name for row in self.role_rows for name in row}
        self._estimated_offsets={}
        for name,value in estimated_offsets.items():
            if name not in available:raise ValueError('Estimated rod is absent from declared observation roles: '+str(name))
            vector=np.asarray(value,dtype=float)
            if vector.shape!=(3,) or not np.isfinite(vector).all() or abs(vector).max()>0.02:
                raise ValueError('Each estimated offset must be a finite XYZ vector within 20 mm')
            vector=vector.copy();vector.setflags(write=False);self._estimated_offsets[name]=vector
        self.observation_contract='strict-v2-148obs-explicit-estimated-bar-offset-v1'

    def observation(self,observation):
        result=super().observation(observation)
        index=min(cycle_index(float(self.env.env.data.time),self.env.ends),len(self.env.ends)-1)
        for slot,name in enumerate(self.role_rows[index]):
            if name in self._estimated_offsets:
                result[136+3*slot:139+3*slot]=self._estimated_offsets[name]/.02
        return result

    def estimate_contract(self):
        return dict(estimated_offsets={name:value.tolist() for name,value in self._estimated_offsets.items()},
            role_rows=self.role_rows,unlisted_features='oracle geometry',
            base_observation='Original 136 state/reference/contact features unchanged',
            source='Explicit deterministic estimate errors; not visual sensing',
            contract=self.observation_contract)
