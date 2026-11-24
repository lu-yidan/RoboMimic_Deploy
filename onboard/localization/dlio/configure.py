"""Generate MID360 host configuration; never reconfigure a host network interface."""
import argparse
import ipaddress
import json
from pathlib import Path
import subprocess


def validate(site):
    host, lidar = ipaddress.IPv4Address(site['host_ip']), ipaddress.IPv4Address(site['lidar_ip'])
    if host == lidar or host.is_loopback or lidar.is_loopback or host.is_multicast or lidar.is_multicast:
        raise ValueError('Distinct unicast host/LiDAR IPv4 addresses required')
    addresses = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr'], text=True))
    interfaces = [ipaddress.IPv4Interface('%s/%s' % (a['local'], a['prefixlen']))
                  for interface in addresses for a in interface.get('addr_info', []) if a.get('local') == str(host)]
    if not interfaces or not any(lidar in interface.network for interface in interfaces):
        raise ValueError('Configured host IP is not assigned locally or LiDAR is outside its subnet')


def driver_config(site):
    host = site['host_ip']
    return dict(lidar_summary_info=dict(lidar_type=8), MID360=dict(
        lidar_net_info=dict(cmd_data_port=56100, push_msg_port=56200, point_data_port=56300,
                            imu_data_port=56400, log_data_port=56500),
        host_net_info=dict(cmd_data_ip=host, cmd_data_port=56101, push_msg_ip=host, push_msg_port=56201,
                           point_data_ip=host, point_data_port=56301, imu_data_ip=host, imu_data_port=56401,
                           log_data_ip='', log_data_port=56501)),
        lidar_configs=[dict(ip=site['lidar_ip'], pcl_data_type=1, pattern_mode=0,
                            extrinsic_parameter=dict(roll=0., pitch=0., yaw=0., x=0, y=0, z=0))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-ip', required=True)
    parser.add_argument('--lidar-ip', required=True)
    parser.add_argument('--output-dir', default='~/.config/robomimic/acp-dlio')
    args = parser.parse_args()
    site = json.loads((Path(__file__).parent / 'config/site.example.json').read_text())
    site.update(host_ip=args.host_ip, lidar_ip=args.lidar_ip)
    validate(site)
    directory = Path(args.output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ('site.json', 'MID360_config.json')):
        raise FileExistsError('Site configuration exists; review it explicitly instead of overwriting')
    # No silent replacement of a reviewed site configuration.
    for name, data in [('site.json', site), ('MID360_config.json', driver_config(site))]:
        with (directory / name).open('x') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
    print(directory / 'site.json')


if __name__ == '__main__':
    main()
