"""Repeat the declared sensing-bias window using explicit cycle boundaries."""
import numpy as np
from training.window_force_history_controller import WindowContactController

class PeriodicForceHistoryController(WindowContactController):
    def __init__(self,*args,cycle_ends,**kwargs):
        super().__init__(*args,**kwargs)
        self.cycle_ends=np.asarray(cycle_ends,dtype=float)
        if self.cycle_ends.ndim!=1 or len(self.cycle_ends)==0 or not np.all(np.diff(np.r_[0.,self.cycle_ends])>61.56):raise ValueError('Cycle must contain full declared load window')

    def bias_window_active(self):
        index=int(np.searchsorted(self.cycle_ends,float(self.d.time),side='right'))
        if index>=len(self.cycle_ends):return False
        start=0. if index==0 else self.cycle_ends[index-1]
        return bool(start+49.<=self.d.time<start+61.56)
