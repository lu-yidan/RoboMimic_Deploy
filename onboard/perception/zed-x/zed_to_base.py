"""
zed_to_base.py

Transform a 3D point from ZED X camera (body frame: X-forward, Y-left, Z-up)
to the pelvis (base) frame.

ZED X 安装位置：胸部（Chest）
运动学链（参考 camera_to_base.py 的胸部相机链）：
    pelvis
      └─ waist_yaw_joint   (Rz, origin=[0, 0, 0])
          └─ waist_roll_joint  (Rx, origin=[-0.0039635, 0, 0.044])
              └─ waist_pitch_joint (Ry, origin=[0, 0, 0])
                  └─ zed_x_joint  (fixed, TODO: 标定后替换 _ZED_XYZ / _ZED_RPY)

TODO (P1): 测量 / 标定 ZED X 在 G1 胸部的安装外参，替换以下占位值：
  _ZED_XYZ = [x, y, z]   # m，ZED X 相对 waist_pitch_link 原点的安装偏移
  _ZED_RPY = (roll, pitch, yaw)  # rad，安装姿态

Note: ZED X 输出光学坐标系 (Z-forward, X-right, Y-down)。
      传入本模块前请先调用 optical_to_body() 转换到 body 系。
      这与 D435 的光学系定义相同，optical_to_body() 函数完全一致。
"""

import numpy as np


def _Rx(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])


def _Ry(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def _Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _rpy_to_R(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# ── ZED X 胸部安装外参（占位，待标定）────────────────────────────────────────
# TODO (P1): 用实际测量值替换以下两行
# 参考 camera_to_base.py 中 _CHEST_XYZ / _CHEST_RPY 的标定方式
_ZED_XYZ = [0.13444, 0.00, 0.06228]   # TODO: 测量（metres，waist_pitch 系下）
_ZED_RPY = (0.00, 0.63931411, 0.00)   # TODO: 标定（roll, pitch, yaw，rad）

_T_ZED_CHEST = _T(
    _rpy_to_R(*_ZED_RPY),
    _ZED_XYZ,
)


def transform_point_zed_to_base(p_zed, q_wy, q_wr, q_wp):
    """
    将 ZED X body 系下的点变换到 G1 pelvis（base）系。

    Args:
        p_zed : array-like (3,)，ZED X body 系下的点（X-forward, metres）
                注意：请先调用 optical_to_body() 将光学系转换为 body 系
        q_wy  : waist_yaw_joint 角度（rad）
        q_wr  : waist_roll_joint 角度（rad）
        q_wp  : waist_pitch_joint 角度（rad）

    Returns:
        np.ndarray (3,)，pelvis 系下的点（metres）

    Note:
        外参 _ZED_XYZ / _ZED_RPY 为占位值，P1 阶段标定后替换。
    """
    T = (
        _T(_Rz(q_wy), [0.0, 0.0, 0.0])
        @ _T(_Rx(q_wr), [-0.0039635, 0.0, 0.044])
        @ _T(_Ry(q_wp), [0.0, 0.0, 0.0])
        @ _T_ZED_CHEST
    )
    return (T @ np.array([*p_zed, 1.0]))[:3]


def optical_to_body(p_optical):
    """
    ZED X 光学系 → body 系（REP-103，与 D435 完全相同）。

    光学系: Z-forward, X-right, Y-down
    Body系: X-forward, Y-left,  Z-up

    Args:
        p_optical : array-like (3,)

    Returns:
        np.ndarray (3,)
    """
    x, y, z = p_optical
    return np.array([z, -x, -y])
