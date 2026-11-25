"""Compile the actual patched observer method in a hardware/ROS-free harness.

Usage: python tools/acp/verify_dlio_observer.py PATH_TO_PATCHED_ODOM_CC
Requires g++ and Eigen headers; creates no DDS participants or motor interfaces.
"""
import pathlib
import subprocess
import sys
import tempfile

source = pathlib.Path(sys.argv[1]).read_text()
start = source.index('bool dlio::OdomNode::updateState() {')
end = source.index('\nsensor_msgs::', start)
method = source[start:end]
harness = r'''
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>
#include <cmath>
#include <cassert>
#include <iostream>
#define RCLCPP_ERROR(...) ((void)0)
namespace dlio {
struct OdomNode {
 struct { Eigen::Vector3f p = Eigen::Vector3f::Zero(); Eigen::Quaternionf q = Eigen::Quaternionf::Identity(); } lidarPose;
 struct { Eigen::Vector3f p = Eigen::Vector3f::Zero(); Eigen::Quaternionf q = Eigen::Quaternionf::Identity();
   struct { Eigen::Vector3f accel = Eigen::Vector3f::Zero(), gyro = Eigen::Vector3f::Zero(); } b;
   struct { struct { Eigen::Vector3f w = Eigen::Vector3f::Zero(); } lin; } v;
 } state;
 struct { std::mutex mtx; Eigen::Vector3f prev_p, prev_vel; Eigen::Quaternionf prev_q; } geo;
 double scan_stamp=100.05, prev_scan_stamp=100.;
 double geo_Kp_=4.5, geo_Kv_=11.25, geo_Kq_=4., geo_Kab_=2.25, geo_Kgb_=1., geo_abias_max_=5., geo_gbias_max_=.5;
 bool updateState();
};
}
'''
tests = r'''
void rejected(double dt, Eigen::Vector3f p, Eigen::Quaternionf q) {
 dlio::OdomNode n; n.scan_stamp=n.prev_scan_stamp+dt; n.lidarPose.p=p; n.lidarPose.q=q;
 assert(!n.updateState());
 assert(n.state.p.isZero()); assert(n.state.q.isApprox(Eigen::Quaternionf::Identity()));
 assert(n.state.b.accel.isZero()); assert(n.state.b.gyro.isZero()); assert(n.state.v.lin.w.isZero());
}
int main() {
 dlio::OdomNode normal; normal.lidarPose.p=Eigen::Vector3f(.01,0,0);
 assert(normal.updateState()); assert(std::abs(normal.state.p.x()-.00225)<1e-6);
 const auto q=Eigen::Quaternionf::Identity();
 // Reproduce a 10 cm observer injection: 0.45 m residual at 20 Hz, Kp=4.5.
 rejected(.05, Eigen::Vector3f(0,0,.45), q);
 rejected(.05, Eigen::Vector3f(NAN,0,0), q);
 rejected(0., Eigen::Vector3f::Zero(), q);
 rejected(-.01, Eigen::Vector3f::Zero(), q);
 rejected(.6, Eigen::Vector3f::Zero(), q);
 rejected(.05, Eigen::Vector3f::Zero(), Eigen::Quaternionf(Eigen::AngleAxisf(1.,Eigen::Vector3f::UnitZ())));
 std::cout << "7 observer cases passed; rejected updates preserved state and biases\n";
}
'''
with tempfile.TemporaryDirectory(prefix='acp-observer-') as tmp:
    path=pathlib.Path(tmp)
    (path/'test.cc').write_text(harness+method+tests)
    subprocess.run(['g++','-std=c++14','-O2','-pthread','-I/usr/include/eigen3',str(path/'test.cc'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
