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
import threading
import select
import termios
import tty
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from livox_ros_driver2.msg import CustomMsg
from unitree_hg.msg import LowState

from onboard.perception.lidar.center_kalman_filter import CenterKalmanFilter
from onboard.perception.lidar.mid360_to_base import transform_point_mid360_to_base
from onboard.perception.lidar.rviz_publisher import RvizPublisher
from common.ball_state_dds import BallStatePublisher


# ---------------------------------------------------------------------------
# Least-squares ball center estimator
# ---------------------------------------------------------------------------

def estimate_ball_center_ls(points, r=0.115, offset=0.05, max_iter=10):
    """Fit sphere of known radius r to point cloud. Returns center (3,)."""
    c = points.mean(axis=0).astype(np.float64)
    offset_in_dir = c/np.linalg.norm(c) * offset
    return c + offset_in_dir
    # for _ in range(max_iter):
    #     v     = points - c[None, :]
    #     dist  = np.linalg.norm(v, axis=1) + 1e-9
    #     resid = dist - r
    #     J     = -(v / dist[:, None])
    #     A     = J.T @ J + 1e-6 * np.eye(3)
    #     dc    = np.linalg.solve(A, -J.T @ resid)
    #     c    += dc
    #     if np.linalg.norm(dc) < 1e-5:
    #         break
    # return c.astype(np.float32)


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class BallDetector(Node):
    def __init__(self):
        super().__init__("ball_detector")

        # ---- Detection params ----
        self.r           = 0.115   # ball radius [m]
        self.reflect_thr = 150
        self.min_points  = 4
        self.max_range   = 1.8
        self.min_range   = 0.2
        self.z_low       = -1.5
        self.z_high      =  1.5
        self.x_low       =  0.0
        self.x_high      =  5.0
        self.center_offset = 0.05  # [m], adjustable at runtime from keyboard

        # ---- Temporal smoothing (Kalman filter) ----
        self.center_kf = CenterKalmanFilter()
        self._last_lidar_ts = None

        # ---- Joint angles (updated from /lowstate) ----
        self.q_wy   = 0.0
        self.q_wr   = 0.0
        self.q_wp   = 0.0
        self.q_head = 0.593412
        self.q_mid  = 0.0

        # ---- DDS publisher ----
        self._dds = BallStatePublisher(domain_id=0)
        self.get_logger().info("DDS publisher ready on 'rt/ball_state'")

        # ---- RViz2 publisher ----
        self._rviz = RvizPublisher(self, frame_id="livox_frame", ball_r=self.r)

        # ---- ROS2 subscriptions ----
        self.create_subscription(CustomMsg, "/livox/lidar",
                                 self.cb_lidar, 5)
        self.create_subscription(LowState, "/lowstate",
                                 self.cb_lowstate, qos_profile_sensor_data)

        # ---- Keyboard control for center offset ----
        self._keyboard_thread = None
        self._keyboard_running = False
        self._stdin_fd = None
        self._stdin_old_term = None
        self._start_keyboard_listener()

        self.get_logger().info("BallDetector ready.")

    # ------------------------------------------------------------------

    def cb_lowstate(self, msg: LowState):
        q = [m.q for m in msg.motor_state]
        self.q_wy = q[12]
        self.q_wr = q[13]
        self.q_wp = q[14]

    # ------------------------------------------------------------------

    def cb_lidar(self, msg: CustomMsg):
        t0    = time.time()
        pts   = msg.points
        stamp = msg.header.stamp
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

        # Publish full cloud + candidates regardless of detection outcome.
        self._rviz.publish_clouds(xyz, cand, stamp)

        if cand.shape[0] < self.min_points:
            self._publish_invalid()
            return

        center_lidar = estimate_ball_center_ls(
            cand,
            r=self.r,
            offset=self.center_offset,
        )
        self.get_logger().info(
            f"ball (raw): ({center_lidar[0]:.3f}, {center_lidar[1]:.3f}, {center_lidar[2]:.3f})"
        )
        self._rviz.publish_ball_raw(center_lidar, stamp)

        now = time.time()
        if self._last_lidar_ts is None:
            dt = 0.1
        else:
            dt = now - self._last_lidar_ts
        self._last_lidar_ts = now
        center_filtered = self.center_kf.step(center_lidar, dt)
        self.get_logger().info(
            f"ball (kf): ({center_filtered[0]:.3f}, {center_filtered[1]:.3f}, {center_filtered[2]:.3f})"
        )
        self._rviz.publish_ball_kf(center_filtered, stamp)

        center_base = transform_point_mid360_to_base(
            center_filtered,
            self.q_wy, self.q_wr, self.q_wp, self.q_head, self.q_mid,
        )

        x, y, z = float(center_base[0]), float(center_base[1]), float(center_base[2])
        # self._dds.publish(x, y, z, valid=True)

        dt_ms = (time.time() - t0) * 1000.0
        self._rviz.publish_text(center_filtered, cand.shape[0],
                                self.center_offset, dt_ms, stamp)
        self.get_logger().info(
            f"ball (pelvis): ({x:.3f}, {y:.3f}, {z:.3f})  "
            f"cand={cand.shape[0]}  cost={dt_ms:.1f}ms"
        )

    def _publish_invalid(self):
        if self.center_kf.initialized:
            cb = transform_point_mid360_to_base(
                self.center_kf.position,
                self.q_wy, self.q_wr, self.q_wp, self.q_head, self.q_mid,
            )
            self._dds.publish(float(cb[0]), float(cb[1]), float(cb[2]), valid=False)
        else:
            self._dds.publish(0.0, 0.0, 0.0, valid=False)

    def _start_keyboard_listener(self):
        if not sys.stdin.isatty():
            self.get_logger().warn("Keyboard offset control disabled (stdin is not a TTY).")
            return

        self._stdin_fd = sys.stdin.fileno()
        self._stdin_old_term = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._keyboard_running = True
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True
        )
        self._keyboard_thread.start()
        self.get_logger().info(
            "Offset keys: '+' increase, '-' decrease, '0' reset."
        )

    def _keyboard_loop(self):
        step = 0.005
        while self._keyboard_running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch in ["+", "="]:
                self.center_offset += step
                self.get_logger().info(f"center_offset = {self.center_offset:.3f} m")
            elif ch in ["-", "_"]:
                self.center_offset = max(0.0, self.center_offset - step)
                self.get_logger().info(f"center_offset = {self.center_offset:.3f} m")
            elif ch == "0":
                self.center_offset = 0.05
                self.get_logger().info(f"center_offset reset to {self.center_offset:.3f} m")

    def destroy_node(self):
        self._keyboard_running = False
        if self._stdin_fd is not None and self._stdin_old_term is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_term)
        return super().destroy_node()


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
