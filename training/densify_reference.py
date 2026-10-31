"""Add samples to linear reference intervals without changing their function."""
import numpy as np


def densify_after(plan, start_s, maximum_dt=.02):
    if not np.isfinite([start_s, maximum_dt]).all() or maximum_dt <= 0:
        raise ValueError('Invalid reference sampling parameters')
    times = plan['time']
    if not np.any(abs(times-start_s)<1e-7):
        raise ValueError('Densification must start at an existing knot')
    result = {k: v.copy() for k, v in plan.items()}
    rows = {k: [plan[k][0].copy()] for k in ('qpos','time','phase','active','errors')}
    for j in range(len(times)-1):
        divisions = max(1, int(np.ceil((times[j+1]-times[j])/maximum_dt))) if times[j]>=start_s-1e-9 else 1
        for part in range(1, divisions+1):
            u = part/divisions
            # Preserve every original endpoint bit-for-bit.
            rows['time'].append(times[j+1].copy() if part==divisions else (1-u)*times[j]+u*times[j+1])
            rows['qpos'].append(plan['qpos'][j+1].copy() if part==divisions else (1-u)*plan['qpos'][j]+u*plan['qpos'][j+1])
            # Runtime selects the ending knot's mask and phase for an interval.
            for key in ('phase','active','errors'):
                rows[key].append(plan[key][j+1].copy())
    for key, values in rows.items():
        result[key] = np.asarray(values, dtype=plan[key].dtype)
    return result
