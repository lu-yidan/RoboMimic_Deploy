"""Ball detector service — runs on G1 onboard computer.

Subscribes to:
  /livox/lidar  (Livox MID360 point cloud)
  /lowstate     (Unitree G1 joint states, for waist/head angles)

Publishes via DDS:
  "rt/ball_state"  (BallState — ball position in pelvis body frame, ~10 Hz)

Usage (on G1 onboard):
    cd RoboMimicDeploy_G1
    python onboard/perception/lidar/ball_detector.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from livox_ros_driver2.msg import CustomMsg
from unitree_hg.msg import LowState

from onboard.perception.lidar.mid360_to_base import transform_point_mid360_to_base
from common.ball_state_dds import BallStatePublisher


# ---------------------------------------------------------------------------
# Least-squares ball center estimator
# ---------------------------------------------------------------------------

def estimate_ball_center_ls(points, r=0.115, max_iter=10):
    """Fit sphere of known radius r to point cloud. Returns center (3,)."""
    c = points.mean(axis=0).astype(np.float64)
    for _ in range(max_iter):
        v     = points - c[None, :]
        dist  = np.linalg.norm(v, axis=1) + 1e-9
        resid = dist - r
        J     = -(v / dist[:, None])
        A     = J.T @ J + 1e-6 * np.eye(3)
        dc    = np.linalg.solve(A, -J.T @ resid)
        c    += dc
        if np.linalg.norm(dc) < 1e-5:
            break
    return c.astype(np.float32)


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class BallDetector(Node):
    def __init__(self):
        super().__init__("ball_detector")

        # ---- Detection params ----
        self.r           = 0.115   # ball radius [m]
        self.reflect_thr = 150
        self.min_points  = 3
        self.max_range   = 1.8
        self.min_range   = 0.2
        self.z_low       = -1.5
        self.z_high      =  1.5
        self.x_low       =  0.0
        self.x_high      =  5.0

        # ---- Temporal smoothing (EMA) ----
        self.alpha      = 0.6
        self.center_ema = None

        # ---- Joint angles (updated from /lowstate) ----
        self.q_wy   = 0.0
        self.q_wr   = 0.0
        self.q_wp   = 0.0
        self.q_head = 0.593412
        self.q_mid  = 0.0

        # ---- DDS publisher ----
        self._dds = BallStatePublisher(domain_id=0)
        self.get_logger().info("DDS publisher ready on 'rt/ball_state'")

        # ---- ROS2 subscriptions ----
        self.create_subscription(CustomMsg, "/livox/lidar",
                                 self.cb_lidar, 5)
        self.create_subscription(LowState, "/lowstate",
                                 self.cb_lowstate, qos_profile_sensor_data)

        self.get_logger().info("BallDetector ready.")

    # ------------------------------------------------------------------

    def cb_lowstate(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]

    # ------------------------------------------------------------------

    def cb_lidar(self, msg: CustomMsg):
        t0  = time.time()
        pts = msg.points
        if not pts:
            self._publish_invalid()
            return

        xyz  = np.empty((len(pts), 3), dtype=np.float32)
        refl = np.empty((len(pts),),   dtype=np.int16)
        for i, p in enumerate(pts):
            xyz[i]  = (p.x, p.y, p.z)
            refl[i] = p.reflectivity

        d    = np.linalg.norm(xyz, axis=1)
        mask = (
            (refl >= self.reflect_thr) &
            (d    >= self.min_range) & (d    <= self.max_range) &
            (xyz[:, 2] >= self.z_low)  & (xyz[:, 2] <= self.z_high) &
            (xyz[:, 0] >= self.x_low)  & (xyz[:, 0] <= self.x_high)
        )
        cand = xyz[mask]

        if cand.shape[0] < self.min_points:
            self._publish_invalid()
            return

        center_lidar = estimate_ball_center_ls(cand, self.r)

        # EMA with gating
        if self.center_ema is None:
            self.center_ema = center_lidar
        elif np.linalg.norm(center_lidar - self.center_ema) < 0.6:
            self.center_ema = (self.alpha * center_lidar
                               + (1.0 - self.alpha) * self.center_ema)

        center_base = transform_point_mid360_to_base(
            self.center_ema,
            self.q_wy, self.q_wr, self.q_wp, self.q_head, self.q_mid,
        )

        x, y, z = float(center_base[0]), float(center_base[1]), float(center_base[2])
        self._dds.publish(x, y, z, valid=True)

        dt_ms = (time.time() - t0) * 1000.0
        self.get_logger().info(
            f"ball (pelvis): ({x:.3f}, {y:.3f}, {z:.3f})  "
            f"cand={cand.shape[0]}  cost={dt_ms:.1f}ms"
        )

    def _publish_invalid(self):
        if self.center_ema is not None:
            cb = transform_point_mid360_to_base(
                self.center_ema,
                self.q_wy, self.q_wr, self.q_wp, self.q_head, self.q_mid,
            )
            self._dds.publish(float(cb[0]), float(cb[1]), float(cb[2]), valid=False)
        else:
            self._dds.publish(0.0, 0.0, 0.0, valid=False)


# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = BallDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
