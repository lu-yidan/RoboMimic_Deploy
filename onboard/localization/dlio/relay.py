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

# Only tiny rigid-body matrices are used here. A spinning BLAS worker on G1
# was consuming a full core; keep this process independent of DLIO's OpenMP.
if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
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
            self.stage_errors = {}
            self.counts = dict(raw_imu=0, si_imu=0, raw_cloud=0, checked_cloud=0,
                               registered_cloud=0, odom=0, pelvis=0, acp_state=0)
            self.last_waist_stamp = 0.
            self.last_raw_arrival = self.last_registered_arrival = 0.
            self.max_raw_gap = self.max_registered_gap = 0.
            self.raw_receive_age = self.registration_receive_age = None
            self.waist_skew = None
            self.imu_norm = None
            self.trace = (Path(__file__).resolve().parents[3] / 'logs' / 'acp-dlio' /
                          ('relay-' + self.session + '.jsonl'))
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            self.last_trace = 0.
            print('[ACP DLIO] Diagnostic trace: ' + str(self.trace), flush=True)
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
            # A 10 Hz snapshot could itself be ~100 ms old when check.py read it.
            # This changes diagnostic reporting only, not acquisition timestamps.
            self.create_timer(.02, self.status)

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
            self.counts['raw_imu'] += 1
            try:
                stamp = self.clock_map.convert(stamp_seconds(raw.header))
                if raw.header.frame_id != 'livox_frame' or stamp <= self.last_imu:
                    raise ValueError('IMU frame mismatch or repeated/reordered stamp')
                accel, gyro = imu_si([raw.linear_acceleration.x, raw.linear_acceleration.y, raw.linear_acceleration.z],
                                     [raw.angular_velocity.x, raw.angular_velocity.y, raw.angular_velocity.z],
                                     site['acceleration_unit'])
                self.imu_norm = float(np.linalg.norm(accel))
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
                msg.header.frame_id = 'acp_livox_imu'
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, accel)
                # Raw acceleration covariance, if supplied, is in squared input units.
                if site['acceleration_unit'] == 'g' and msg.linear_acceleration_covariance[0] != -1:
                    msg.linear_acceleration_covariance = [float(v)*9.80665**2 for v in msg.linear_acceleration_covariance]
                self.imu_pub.publish(msg)
                self.counts['si_imu'] += 1
                self.stage_errors.pop('imu', None)
            except (ValueError, KeyError) as error:
                self.error = str(error)
                self.stage_errors['imu'] = self.error

        def points(self, msg):
            self.counts['raw_cloud'] += 1
            arrival = time.monotonic()
            if self.last_raw_arrival:
                self.max_raw_gap = max(self.max_raw_gap, arrival-self.last_raw_arrival)
            self.last_raw_arrival = arrival
            self.raw_receive_age = time.time()-stamp_seconds(msg.header)
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
                self.counts['checked_cloud'] += 1
                self.stage_errors.pop('raw_cloud', None)
            except (ValueError, KeyError) as error:
                self.error = str(error)
                self.stage_errors['raw_cloud'] = self.error

        def cloud(self, msg):
            arrival = time.monotonic()
            if self.last_registered_arrival:
                self.max_registered_gap = max(self.max_registered_gap, arrival-self.last_registered_arrival)
            self.last_registered_arrival = arrival
            self.registration_receive_age = time.time()-stamp_seconds(msg.header)
            try:
                stamp = self.clock_map.convert(stamp_seconds(msg.header), max_age=.25)
                if msg.header.frame_id != 'acp_odom' or msg.width*msg.height <= 64 or stamp <= self.last_cloud:
                    raise ValueError('Invalid/reordered DLIO registration heartbeat')
                self.last_cloud = stamp
                self.counts['registered_cloud'] += 1
                self.stage_errors.pop('registration', None)
            except ValueError as error:
                self.error = str(error)
                self.stage_errors['registration'] = self.error

        def odom(self, msg):
            self.counts['odom'] += 1
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
                self.last_waist_stamp, self.waist_skew = joint_stamp, joint_stamp-stamp
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
                self.counts['pelvis'] += 1
                self.error = ''
                self.stage_errors.pop('odom', None)
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
                    self.counts['acp_state'] += 1
                    self.last_published = now
            except (ValueError, KeyError, OSError) as error:
                self.error = str(error)
                self.stage_errors['odom'] = self.error
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
            payload.update(imu_ready=self.imu_ready, imu_accel_norm_mps2=self.imu_norm,
                           raw_cloud_age_s=now-self.last_raw_cloud if self.last_raw_cloud else None,
                           waist_age_s=now-self.last_waist_stamp if self.last_waist_stamp else None,
                           waist_skew_s=self.waist_skew, counts=self.counts.copy(),
                           stage_errors=self.stage_errors.copy(),
                           calibration_ready=self.calibration is not None,
                           localization_fault=self.gate.failed,
                           localization_fault_details=self.gate.failure_details,
                           diagnostic_trace=str(self.trace))
            payload.update(raw_cloud_arrival_age_s=now-self.last_raw_arrival if self.last_raw_arrival else None,
                           registered_cloud_arrival_age_s=now-self.last_registered_arrival if self.last_registered_arrival else None,
                           raw_cloud_receive_age_s=self.raw_receive_age,
                           registration_receive_age_s=self.registration_receive_age,
                           raw_cloud_max_gap_s=self.max_raw_gap,
                           registered_cloud_max_gap_s=self.max_registered_gap,
                           status_period_s=.02, blas_threads=1)
            if self.pose is not None:
                payload.update(pelvis_pos_w=self.pose[0].tolist(), pelvis_quat_w=self.pose[1].tolist())
            atomic_json(directory / 'acp-dlio-status.json', payload)
            if now-self.last_trace >= .2:
                with self.trace.open('a') as stream:
                    stream.write(json.dumps(payload, allow_nan=False) + '\n')
                self.last_trace = now
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
