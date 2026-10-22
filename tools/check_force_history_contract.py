"""Verify persistence needs neither current gap query nor unbounded memory age."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.force_history_contact_controller import ForceHistoryContactController

c=object.__new__(ForceHistoryContactController)
c.d=SimpleNamespace(xmat=np.eye(3).reshape(1,9),xpos=np.array([[.5,.2,.7]]))
c.support_memory={'lh':dict(time_s=1.,points=[(np.array([0.,0.,.02]),np.eye(3),.7,1,2)])}
c.memory_window_s=.02;c.memory_events=[]
with patch('mujoco.mj_geomDistance',side_effect=AssertionError('Forbidden gap query')):
    points=c.memory_points('lh',0,1.01)
assert np.allclose(points[0][0],[.5,.2,.72]) and c.memory_events[-1]['gap_query_used'] is False
for key,t in [('lh',1.021),('lh',.99),('rh',1.01)]:
    try:c.memory_points(key,0,t)
    except RuntimeError:pass
    else:raise AssertionError('Invalid history accepted')
assert len(c.memory_events)==1
print('PASS: no gap query, bounded age, absent/negative/expired history rejected')
