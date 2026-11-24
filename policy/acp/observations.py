"""Raw ACP observations; normalization is already inside policy.onnx."""
import numpy as np

DEFAULT_MAX_GRAVITY_ERROR_RAD = np.pi / 18  # 10 degrees; independent of world yaw.

_grid_x, _grid_y = np.meshgrid(np.arange(-8, 9) * .1, np.arange(-5, 6) * .1)
_GRID_XY = np.stack((_grid_x.ravel(), _grid_y.ravel()), axis=-1).astype(np.float32)
_GRID_XY.setflags(write=False)
del _grid_x, _grid_y


def grid_xy():
    # Validation runs before and after inference. Reuse the fixed geometry,
    # retaining an independent writable result for existing callers.
    return _GRID_XY.copy()


def rotation(quat):
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > .02:
        raise ValueError('Invalid pelvis quaternion (expected unit wxyz)')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def yaw(quat):
    r = rotation(quat)
    return np.arctan2(r[1, 0], r[0, 0])


def check_gravity_alignment(pelvis_quat, imu_quat, limit=DEFAULT_MAX_GRAVITY_ERROR_RAD):
    """Compare body-frame gravity, allowing the two world frames different yaw."""
    if not np.isfinite(limit) or not 0 < limit < np.pi/2:
        raise ValueError('Invalid ACP gravity disagreement limit')
    cosine = np.dot(rotation(pelvis_quat)[2], rotation(imu_quat)[2])
    error = float(np.arccos(np.clip(cosine, -1., 1.)))
    if error > limit:
        raise ValueError('ACP pelvis gravity disagrees with robot IMU '
                         f'({np.degrees(error):.2f} deg > {np.degrees(limit):.2f} deg)')
    return error


def goal_command(position, quaternion, target, heading, radius):
    if radius <= 0:
        raise ValueError('Goal radius must be positive')
    psi = yaw(quaternion)
    delta = np.asarray(target, dtype=np.float64) - np.asarray(position)[:2]
    c, s = np.cos(psi), np.sin(psi)
    local = np.array([c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]])
    local /= max(np.linalg.norm(delta), radius)
    angle = np.arctan2(np.sin(heading-psi), np.cos(heading-psi))
    return np.r_[local, angle].astype(np.float32)


class ObservationHistory:
    """Five frames per term, oldest first; reset repeats the first measurement."""
    def __init__(self):
        self.terms = None

    def reset(self):
        self.terms = None

    def append(self, angular_velocity, quaternion, joint_position, joint_velocity, last_action):
        terms = [np.asarray(angular_velocity), rotation(quaternion).T @ [0., 0., -1.],
                 np.asarray(joint_position), np.asarray(joint_velocity), np.asarray(last_action)]
        if [x.shape for x in terms] != [(3,), (3,), (23,), (23,), (23,)]:
            raise ValueError('Invalid ACP proprioception shape')
        if not all(np.isfinite(x).all() for x in terms):
            raise ValueError('Nonfinite ACP proprioception')
        if self.terms is None:
            self.terms = [np.repeat(x[None], 5, axis=0).astype(np.float32) for x in terms]
        else:
            for history, term in zip(self.terms, terms):
                history[:-1] = history[1:]
                history[-1] = term
        return np.concatenate([x.ravel() for x in self.terms])
