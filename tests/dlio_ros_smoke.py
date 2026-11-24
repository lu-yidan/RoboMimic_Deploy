"""Offline ROS2/Foxy integration: synthetic Livox -> real DLIO -> pelvis/ACP IPC.

No physical driver, Unitree/DDS subscriber or motor publisher is created.
Run in a sourced Foxy + ACP DLIO workspace using system Python.
"""
import argparse
from array import array
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
from onboard.localization.dlio.check import assess
from onboard.perception.lidar.mid360_to_base import compute_mid360_to_base_transform
from policy.acp.state_ipc import boot_id


def udp_counters():
    lines = Path('/proc/net/snmp').read_text().splitlines()
    for header, values in zip(lines, lines[1:]):
        if header.startswith('Udp:') and values.startswith('Udp:'):
            return dict(zip(header.split()[1:], map(int, values.split()[1:])))
    raise RuntimeError('No Linux UDP counters')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--points', type=int, default=1600)
    parser.add_argument('--cloud-age-ms', type=float, default=120., help='Age of first point when generating each 100 ms cloud')
    parser.add_argument('--duration', type=float, default=14.)
    args = parser.parse_args()
    if not 1600 <= args.points <= 100000:
        parser.error('--points must be between 1600 and 100000')
    if not 90 <= args.cloud_age_ms <= 200:
        parser.error('--cloud-age-ms must be between 90 and 200')
    if not 14 <= args.duration <= 60:
        parser.error('--duration must be between 14 and 60 seconds')
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '78':
        parser.error('Use ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=78 (isolated offline domain)')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    from nav_msgs.msg import Odometry
    directory = Path(tempfile.mkdtemp(prefix='acp-dlio-synthetic-'))
    udp_before = udp_counters()
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
    result = dict(hardware_started=False, directory=str(directory), passed=False,
                  points_per_cloud=args.points, source_cloud_age_ms=args.cloud_age_ms,
                  duration_s=args.duration, kernel_rmem_max=int(Path('/proc/sys/net/core/rmem_max').read_text()),
                  cyclone_config=os.environ.get('CYCLONEDDS_URI', '(default)'))
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
            xyz = rng.uniform(-3., 3., (args.points, 3))
            quarter = args.points//4
            xyz[:quarter, 2] = -1.25
            xyz[quarter:2*quarter, 0] = 3.
            xyz[2*quarter:3*quarter, 1] = 3.
            xyz[3*quarter:, 1] = -3.
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
            first = time.time()-args.cloud_age_ms/1000
            self.points['timestamp'] = (first+np.linspace(0., .09, len(self.points)))*1e9
            message = PointCloud2()
            self.stamp(message.header, first)
            message.width, message.height = len(self.points), 1
            message.point_step, message.row_step = 24, 24*len(self.points)
            message.is_dense = True
            message.fields = [PointField(name=key, offset=offset, datatype=typ, count=1)
                              for key, offset, typ in [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('intensity', 12, 7), ('timestamp', 16, 8)]]
            # Foxy validates bytes element-by-element; an already typed buffer
            # avoids blocking this same executor's 200 Hz synthetic IMU timer.
            message.data = array('B', self.points.tobytes())
            self.cloud_pub.publish(message)

        def pose(self, message):
            p, q = message.pose.pose.position, message.pose.pose.orientation
            self.poses.append([time.monotonic(), p.x, p.y, p.z, q.w, q.x, q.y, q.z])

    node = Synthetic()
    try:
        started, calibrated = time.monotonic(), False
        valid_frames = 0
        steady_status = []
        last_sampled = 0.
        while time.monotonic()-started < args.duration:
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
                if time.monotonic()-started > 8. and status['sampled_at'] != last_sampled:
                    steady_status.append(status)
                    last_sampled = status['sampled_at']
            except OSError:
                pass
        if not node.poses or not calibrated or valid_frames < 30:
            raise RuntimeError('No sustained fresh pelvis/ACP output')
        if len(steady_status) < 2:
            raise RuntimeError('No steady-state status observations')
        pose = np.array(node.poses)
        norms = np.linalg.norm(pose[:, 4:], axis=1)
        if not np.allclose(norms, 1., atol=1e-5) or np.max(np.ptp(pose[-100:, 1:4], axis=0)) > .05:
            raise RuntimeError('Stationary synthetic root drift or invalid quaternion')
        with np.load(directory/'acp-state.npz', allow_pickle=False) as packet:
            assert str(packet['frame_id']) == 'acp_odom'
            assert abs(float(packet['pelvis_pos_w'][2])-float(packet['ground_height'])-.78) < .03
        result['steady_observation'] = dict(
            distinct_status_samples=len(steady_status),
            fresh_pelvis_fraction=float(np.mean([s['pose_valid'] for s in steady_status])),
            acp_output_fraction=float(np.mean([s['acp_output_valid'] for s in steady_status])),
            status_interval_p99_s=float(np.percentile(np.diff([s['sampled_at'] for s in steady_status]), 99)),
            last_status=steady_status[-1])
        stages = {}
        for sample in steady_status:
            stage = assess(sample, sample['sampled_at'], boot_id())[0]
            stages[stage] = stages.get(stage, 0)+1
        result['steady_observation']['blocking_stages'] = stages
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
        result['host_udp_counter_delta'] = {key:value-udp_before[key] for key,value in udp_counters().items()}
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
