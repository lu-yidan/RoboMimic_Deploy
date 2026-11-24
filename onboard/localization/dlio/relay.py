"""ROS2 MID360 preflight + DLIO pelvis relay. No DDS motor command interfaces."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from onboard.localization.dlio.core import (HostClock, PoseGate, WaistHistory, atomic_json,
                                           imu_si, pelvis_pose, read_json)
from policy.acp.state_ipc import StateWriter, boot_id
from policy.acp.goal_ipc import write_goal
from policy.acp.observations import yaw, rotation


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec*1e-9


def point_times(message):
    fields = {f.name: f for f in message.fields}
    field = fields.get('timestamp')
    count = message.width*message.height
    if (field is None or field.datatype != 8 or field.count != 1 or message.is_bigendian or
            count <= 64 or message.point_step < field.offset+8 or
            message.row_step < message.width*message.point_step or
            len(message.data) < message.height*message.row_step):
        raise ValueError('Expected nonempty Livox PointCloud2 with float64 absolute-ns timestamp')
    values = np.ndarray((message.height, message.width), dtype='<f8', buffer=message.data,
                        offset=field.offset, strides=(message.row_step, message.point_step))
    # Check every point, including row padding. This does not alter the cloud.
    first, last = float(np.min(values))*1e-9, float(np.max(values))*1e-9
    if not np.isfinite(values).all() or first < 1e9 or not 0 <= last-first <= .25:
        raise ValueError('Point timestamps must be absolute host-epoch ns, not relative times')
    if abs(first-stamp_seconds(message.header)) > .02:
        raise ValueError('Cloud header/point time mismatch')
    return first, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True)
    parser.add_argument('--ipc-dir', default='/dev/shm/robomimic-%d' % os.getuid())
    parser.add_argument('--publish-acp', action='store_true', help='Requires current-session ground/mount calibration')
    args = parser.parse_args()
    site = read_json(args.site)
    if site.get('acceleration_unit') not in ('g', 'm/s2'):
        parser.error('Site must explicitly declare acceleration_unit')
    angles = np.asarray([site['head_angle_rad'], site['lidar_angle_rad']], dtype=float)
    if not np.isfinite(angles).all():
        parser.error('Invalid mount angles')
    directory = Path(args.ipc_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'acp-dlio.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, PointCloud2
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped

    class Relay(Node):
        def __init__(self):
            super().__init__('acp_dlio_relay')
            self.clock_map, self.gate, self.waist = HostClock(), PoseGate(), WaistHistory()
            self.session, self.sequence = str(uuid.uuid4()), 0
            self.last_imu = self.last_raw_cloud = self.last_cloud = self.last_pose = 0.
            self.pose = None
            self.speed = self.turn_rate = None
            self.last_published = 0.
            self.error = 'Waiting for Livox and DLIO'
            self.initial_imu, self.imu_ready = [], False
            self.calibration = None
            self.imu_pub = self.create_publisher(Imu, '/acp/livox/imu_si', qos_profile_sensor_data)
            self.points_pub = self.create_publisher(PointCloud2, '/acp/livox/points_checked', qos_profile_sensor_data)
            self.pelvis_pub = self.create_publisher(Odometry, '/acp/pelvis/odom', 10)
            self.create_subscription(Imu, '/livox/imu', self.imu, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, '/livox/lidar', self.points, qos_profile_sensor_data)
            self.create_subscription(Odometry, '/dlio/odom_node/odom', self.odom, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, '/dlio/odom_node/pointcloud/deskewed', self.cloud, qos_profile_sensor_data)
            self.create_subscription(PoseStamped, '/acp/goal_pose', self.goal, 10)
            self.create_timer(.1, self.status)

        def goal(self, msg):
            try:
                self.clock_map.convert(stamp_seconds(msg.header), max_age=2.)
                if msg.header.frame_id != 'acp_odom':
                    raise ValueError('RViz goal must be in acp_odom')
                q = msg.pose.orientation
                heading = yaw([q.w, q.x, q.y, q.z])
                write_goal(directory, read_json(directory / 'acp-dlio-status.json'),
                           [msg.pose.position.x, msg.pose.position.y], heading)
                self.get_logger().info('Accepted fixed world goal from /acp/goal_pose')
            except (ValueError, KeyError, OSError) as error:
                self.get_logger().warning('Goal rejected: ' + str(error))

        def imu(self, raw):
            try:
                stamp = self.clock_map.convert(stamp_seconds(raw.header))
                if raw.header.frame_id != 'livox_frame' or stamp <= self.last_imu:
                    raise ValueError('IMU frame mismatch or repeated/reordered stamp')
                accel, gyro = imu_si([raw.linear_acceleration.x, raw.linear_acceleration.y, raw.linear_acceleration.z],
                                     [raw.angular_velocity.x, raw.angular_velocity.y, raw.angular_velocity.z],
                                     site['acceleration_unit'])
                if not self.imu_ready:
                    if abs(np.linalg.norm(accel)-9.80665) > 1.5 or np.linalg.norm(gyro) > .15:
                        self.initial_imu = []
                        raise ValueError('Startup IMU: keep still; check g/SI units (expected norm 9.81 m/s2)')
                    self.initial_imu.append(stamp)
                    if len(self.initial_imu) < 100 or stamp-self.initial_imu[0] < 1.:
                        return
                    self.imu_ready = True
                self.last_imu = stamp
                msg = copy.deepcopy(raw)
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, accel)
                # Raw acceleration covariance, if supplied, is in squared input units.
                if site['acceleration_unit'] == 'g' and msg.linear_acceleration_covariance[0] != -1:
                    msg.linear_acceleration_covariance = [float(v)*9.80665**2 for v in msg.linear_acceleration_covariance]
                self.imu_pub.publish(msg)
            except (ValueError, KeyError) as error:
                self.error = str(error)

        def points(self, msg):
            try:
                if msg.header.frame_id != 'livox_frame':
                    raise ValueError('Unexpected raw LiDAR frame; disable driver extrinsic transforms')
                first, last = point_times(msg)
                stamp = self.clock_map.convert(first, max_age=.3)
                self.clock_map.convert(last, max_age=.3)
                if stamp <= self.last_raw_cloud:
                    raise ValueError('Repeated/reordered raw cloud')
                self.last_raw_cloud = stamp
                self.points_pub.publish(msg)
            except (ValueError, KeyError) as error:
                self.error = str(error)

        def cloud(self, msg):
            try:
                stamp = self.clock_map.convert(stamp_seconds(msg.header), max_age=.25)
                if msg.header.frame_id != 'acp_odom' or msg.width*msg.height <= 64 or stamp <= self.last_cloud:
                    raise ValueError('Invalid/reordered DLIO registration heartbeat')
                self.last_cloud = stamp
            except ValueError as error:
                self.error = str(error)

        def odom(self, msg):
            try:
                now = time.monotonic()
                stamp = self.clock_map.convert(stamp_seconds(msg.header))
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                if not self.gate.check(stamp, [p.x, p.y, p.z], [q.w, q.x, q.y, q.z], msg.header.frame_id, msg.child_frame_id):
                    return
                if not self.imu_ready or not 0 <= now-self.last_imu <= .15 or not 0 <= now-self.last_raw_cloud <= .3 or not 0 <= now-self.last_cloud <= .25:
                    raise ValueError('IMU, raw cloud, or registered DLIO cloud stale/not initialized')
                self.waist.load(read_json(directory / 'acp-waist.json'), now)
                waist, joint_stamp = self.waist.nearest(stamp, now)
                pos, quat = pelvis_pose([p.x, p.y, p.z], [q.w, q.x, q.y, q.z], waist, *angles)
                if self.pose is not None and stamp > self.last_pose:
                    dt = stamp-self.last_pose
                    self.speed = float(np.linalg.norm(pos-self.pose[0])/dt)
                    difference = rotation(self.pose[1]).T @ rotation(quat)
                    self.turn_rate = float(np.arccos(np.clip((np.trace(difference)-1)/2, -1, 1))/dt)
                self.pose, self.last_pose = (pos, quat), stamp
                derived = copy.deepcopy(msg)
                derived.child_frame_id = 'pelvis'
                derived.pose.covariance = [0.]*36  # DLIO does not estimate a transformed pelvis covariance.
                derived.pose.pose.position.x, derived.pose.pose.position.y, derived.pose.pose.position.z = map(float, pos)
                derived.pose.pose.orientation.w, derived.pose.pose.orientation.x, derived.pose.pose.orientation.y, derived.pose.pose.orientation.z = map(float, quat)
                # DLIO twist is not a pelvis twist; do not publish it under that name.
                derived.twist = Odometry().twist
                derived.twist.covariance[0] = -1.
                self.pelvis_pub.publish(derived)
                self.error = ''
                if args.publish_acp:
                    calibration = read_json(directory / 'acp-dlio-calibration.json')
                    ground = float(calibration['ground_height'])
                    if (calibration.get('session') != self.session or calibration.get('mount_reviewed') is not True
                            or calibration.get('mount_angles_rad') != angles.tolist()
                            or not np.isfinite(ground) or not .3 <= pos[2]-ground <= 1.2):
                        raise ValueError('Missing current-session mount/ground calibration or invalid clearance')
                    self.calibration = calibration
                    StateWriter(directory / 'acp-state.npz', frame_id='acp_odom').write(
                        pelvis_pos_w=pos, pelvis_quat_w=quat, captured_at=stamp,
                        pose_captured_at=stamp, terrain_captured_at=stamp,
                        sequence=self.sequence, ground_height=ground)
                    self.sequence += 1
                    self.last_published = now
            except (ValueError, KeyError, OSError) as error:
                self.error = str(error)
                # Immediate invalidation; never let a bad callback renew an old valid packet.
                if args.publish_acp:
                    (directory / 'acp-state.npz').unlink(missing_ok=True)

        def status(self):
            now = time.monotonic()
            pose_fresh = self.pose is not None and 0 <= now-self.last_pose <= .15
            payload = dict(schema=1, session=self.session, boot_id=boot_id(), sampled_at=now,
                           frame_id='acp_odom', pose_captured_at=self.last_pose,
                           imu_age_s=now-self.last_imu, registration_age_s=now-self.last_cloud,
                           pose_valid=pose_fresh, acp_output_enabled=args.publish_acp,
                           acp_output_valid=args.publish_acp and now-self.last_published < .15 and not self.error,
                           error=self.error or ('' if pose_fresh else 'Pelvis pose stale'),
                           speed_mps=self.speed, turn_rate_rps=self.turn_rate,
                           mount_angles_rad=angles.tolist())
            if self.pose is not None:
                payload.update(pelvis_pos_w=self.pose[0].tolist(), pelvis_quat_w=self.pose[1].tolist())
            atomic_json(directory / 'acp-dlio-status.json', payload)
            # Missing clouds/IMU must stop output even if odometry keeps propagating.
            if args.publish_acp and (not pose_fresh or self.error):
                (directory / 'acp-state.npz').unlink(missing_ok=True)

    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if args.publish_acp:
            (directory / 'acp-state.npz').unlink(missing_ok=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
