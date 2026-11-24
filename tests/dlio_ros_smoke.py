"""Offline ROS2/Foxy integration: synthetic Livox -> real DLIO -> pelvis/ACP IPC.

No physical driver, Unitree/DDS subscriber or motor publisher is created.
Run in a sourced Foxy + ACP DLIO workspace using system Python.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onboard.localization.dlio.core import atomic_json, read_json, WaistHistory
from onboard.perception.lidar.mid360_to_base import compute_mid360_to_base_transform
from policy.acp.state_ipc import boot_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '78':
        parser.error('Use ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=78 (isolated offline domain)')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from nav_msgs.msg import Odometry
    directory = Path(tempfile.mkdtemp(prefix='acp-dlio-synthetic-'))
    here = ROOT/'onboard/localization/dlio'
    native_log = (directory/'dlio.log').open('w')
    relay_log = (directory/'relay.log').open('w')
    native = subprocess.Popen(['ros2', 'run', 'direct_lidar_inertial_odometry', 'dlio_odom_node',
                               '--ros-args', '--params-file', str(here/'config/dlio.yaml'),
                               '-r', 'pointcloud:=/acp/livox/points_checked', '-r', 'imu:=/acp/livox/imu_si',
                               '-r', 'odom:=/dlio/odom_node/odom',
                               '-r', 'deskewed:=/dlio/odom_node/pointcloud/deskewed'],
                              stdout=native_log, stderr=subprocess.STDOUT, start_new_session=True)
    relay = subprocess.Popen(['/usr/bin/python3', '-u', str(here/'relay.py'),
                              '--site', str(here/'config/site.example.json'),
                              '--ipc-dir', str(directory), '--publish-acp'],
                             stdout=relay_log, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(hardware_started=False, directory=str(directory), passed=False)
    rclpy.init()

    class Synthetic(Node):
        def __init__(self):
            super().__init__('acp_synthetic_livox')
            self.imu_pub = self.create_publisher(Imu, '/livox/imu', qos_profile_sensor_data)
            self.cloud_pub = self.create_publisher(PointCloud2, '/livox/lidar', qos_profile_sensor_data)
            self.create_subscription(Odometry, '/acp/pelvis/odom', self.pose, 10)
            self.history, self.tick, self.poses = WaistHistory(), 0, []
            self.cloud_enabled = self.imu_enabled = True
            self.create_timer(.005, self.imu)
            self.create_timer(.1, self.cloud)
            mount = compute_mid360_to_base_transform()
            self.accel_g = mount[:3, :3].T @ [0., 0., 1.]
            rng = np.random.RandomState(42)
            # Room walls + floor, full 3D structure, fixed points in sensor coordinates.
            xyz = rng.uniform(-3., 3., (1600, 3))
            xyz[:400, 2] = -1.25
            xyz[400:800, 0] = 3.
            xyz[800:1200, 1] = 3.
            xyz[1200:, 1] = -3.
            self.points = np.zeros(len(xyz), dtype=np.dtype(dict(
                names=['x', 'y', 'z', 'intensity', 'timestamp'],
                formats=['<f4', '<f4', '<f4', '<f4', '<f8'], offsets=[0, 4, 8, 12, 16], itemsize=24)))
            for i, key in enumerate(('x', 'y', 'z')):
                self.points[key] = xyz[:, i]
            self.points['intensity'] = 20.

        def stamp(self, header, seconds):
            header.stamp.sec = int(seconds)
            header.stamp.nanosec = int((seconds-int(seconds))*1e9)
            header.frame_id = 'livox_frame'

        def imu(self):
            now = time.monotonic()
            self.tick += 1
            self.history.add(self.tick, now-.005, np.zeros(29))
            atomic_json(directory/'acp-waist.json', dict(schema=1, boot_id=boot_id(), source_id='synthetic', rows=self.history.rows))
            if not self.imu_enabled:
                return
            message = Imu()
            self.stamp(message.header, time.time()-.005)
            message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = map(float, self.accel_g)
            self.imu_pub.publish(message)

        def cloud(self):
            if not self.cloud_enabled:
                return
            first = time.time()-.12
            self.points['timestamp'] = (first+np.linspace(0., .09, len(self.points)))*1e9
            message = PointCloud2()
            self.stamp(message.header, first)
            message.width, message.height = len(self.points), 1
            message.point_step, message.row_step = 24, 24*len(self.points)
            message.is_dense = True
            message.fields = [PointField(name=key, offset=offset, datatype=typ, count=1)
                              for key, offset, typ in [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 12, 7), ('timestamp', 16, 8)]]
            message.data = self.points.tobytes()
            self.cloud_pub.publish(message)

        def pose(self, message):
            p, q = message.pose.pose.position, message.pose.pose.orientation
            self.poses.append([time.monotonic(), p.x, p.y, p.z, q.w, q.x, q.y, q.z])

    node = Synthetic()
    try:
        started, calibrated = time.monotonic(), False
        valid_frames = 0
        while time.monotonic()-started < 14.:
            rclpy.spin_once(node, timeout_sec=.005)
            if native.poll() is not None or relay.poll() is not None:
                raise RuntimeError('DLIO or relay exited; see logs')
            try:
                status = read_json(directory/'acp-dlio-status.json')
                if status['pose_valid'] and not calibrated:
                    atomic_json(directory/'acp-dlio-calibration.json', dict(session=status['session'],
                                mount_reviewed=True, mount_angles_rad=status['mount_angles_rad'],
                                ground_height=status['pelvis_pos_w'][2]-.78))
                    calibrated = True
                if status['acp_output_valid']:
                    valid_frames += 1
            except OSError:
                pass
        if not node.poses or not calibrated or valid_frames < 30:
            raise RuntimeError('No sustained fresh pelvis/ACP output')
        pose = np.array(node.poses)
        norms = np.linalg.norm(pose[:, 4:], axis=1)
        if not np.allclose(norms, 1., atol=1e-5) or np.max(np.ptp(pose[-100:, 1:4], axis=0)) > .05:
            raise RuntimeError('Stationary synthetic root drift or invalid quaternion')
        with np.load(directory/'acp-state.npz', allow_pickle=False) as packet:
            assert str(packet['frame_id']) == 'acp_odom'
            assert abs(float(packet['pelvis_pos_w'][2])-float(packet['ground_height'])-.78) < .03
        node.cloud_enabled = False
        then = time.monotonic()
        while time.monotonic()-then < .7:
            rclpy.spin_once(node, timeout_sec=.005)
        status = read_json(directory/'acp-dlio-status.json')
        if (directory/'acp-state.npz').exists() or status['acp_output_valid']:
            raise RuntimeError('Cloud stall did not invalidate ACP output')
        result.update(passed=True, pelvis_messages=len(node.poses), valid_status_samples=valid_frames,
                      stationary_drift_m=np.ptp(pose[-100:, 1:4], axis=0).tolist(),
                      cloud_stall_invalidated=True, final_status=status)
    except Exception as error:
        result['error'] = str(error)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        for process in (relay, native):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
        native_log.close()
        relay_log.close()
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
