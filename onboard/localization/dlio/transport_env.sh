# Source after ROS setup, only in DLIO / RViz / ROS recording shells.
# No sysctl writes: the operator explicitly prepares the host's receive limit.
ACP_DLIO_TRANSPORT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $(</proc/sys/net/core/rmem_max) < 8388608 )); then
  echo 'DLIO needs a larger UDP receive limit for point clouds. Run on G1:' >&2
  echo '  sudo sysctl -w net.core.rmem_max=8388608' >&2
  echo 'This runtime setting expires at reboot; then rerun this pane.' >&2
  return 2
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ACP_ROS_DOMAIN_ID:-37}"
export ROS_LOCALHOST_ONLY=1
export CYCLONEDDS_URI="file://$ACP_DLIO_TRANSPORT_DIR/config/cyclonedds.xml"
