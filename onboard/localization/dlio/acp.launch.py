"""Single owner of Livox, DLIO and relay; exiting any node stops this launch only."""
import json
import os
from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configure import validate, driver_config


def generate_launch_description():
    site_file = Path(os.environ['ACP_DLIO_SITE']).expanduser().resolve()
    site = json.loads(site_file.read_text())
    validate(site)
    config = site_file.with_name('MID360_config.json')
    if json.loads(config.read_text()) != driver_config(site):
        raise RuntimeError('MID360 JSON/site mismatch, including driver extrinsic transform')
    driver = Node(package='livox_ros_driver2', executable='livox_ros_driver2_node',
                  name='acp_livox_driver', output='screen', parameters=[dict(
                      xfer_format=0, multi_topic=0, data_src=0, publish_freq=10.,
                      output_data_type=0, frame_id='livox_frame', user_config_path=str(config),
                      cmdline_input_bd_code='livox0000000001')])
    odom = Node(package='direct_lidar_inertial_odometry', executable='dlio_odom_node',
                name='dlio_odom_node', output='screen', parameters=[str(HERE / 'config/dlio.yaml')],
                remappings=[('pointcloud', '/acp/livox/points_checked'), ('imu', '/acp/livox/imu_si'),
                            ('odom', '/dlio/odom_node/odom'), ('pose', '/dlio/odom_node/pose'),
                            ('path', '/dlio/odom_node/path'), ('kf_pose', '/dlio/odom_node/keyframes'),
                            ('kf_cloud', '/dlio/odom_node/keyframes_cloud'),
                            ('deskewed', '/dlio/odom_node/pointcloud/deskewed')])
    command = ['/usr/bin/python3', '-u', str(HERE / 'relay.py'), '--site', str(site_file)]
    if os.environ.get('ACP_DLIO_PUBLISH') == '1':
        command.append('--publish-acp')
    relay = ExecuteProcess(cmd=command, output='screen')
    actions = [driver, odom, relay]
    for process in (driver, odom, relay):
        actions.append(RegisterEventHandler(OnProcessExit(target_action=process, on_exit=[
            EmitEvent(event=Shutdown(reason='ACP localization process exited'))])))
    return LaunchDescription(actions)
