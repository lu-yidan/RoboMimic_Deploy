"""Build configurable MuJoCo obstacles on top of a robot XML model."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


_DEPLOY_DEFAULT_JOINT_POS = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Exact support-ready supine arm initialization from smp/v34_reset_legacy.py.
_SUPINE_CROSSED_ARM_POS = np.array(
    [
        -2.104, -1.105, 0.0, 0.744, 0.0, 0.0, 0.0,
        -2.104, 1.105, 0.0, 0.744, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)


def _vec(value: Any, size: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value)
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(values)}")
    return values


def _text(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)


def _add_table(worldbody: Any, config: Any) -> None:
    position = _vec(config.get("pos", [1.0, 0.0, 0.0]), 3, "table.pos")
    length = float(config.get("length", 1.6))
    width = float(config.get("width", 0.6))
    height = float(config.get("height", 0.4))  # Top surface, measured from table.pos.z.
    thickness = float(config.get("thickness", 0.04))
    leg_width = float(config.get("leg_width", 0.05))
    leg_inset = float(config.get("leg_inset", 0.0))
    if any(key in config for key in ("top_size", "leg_size")):
        raise ValueError("Use full length/width/height/thickness/leg_width; legacy half-sizes are unsupported")
    if not all(np.isfinite(v) for v in (length, width, height, thickness, leg_width, leg_inset)):
        raise ValueError("Table dimensions must be finite")
    if min(length, width, thickness, leg_width) <= 0 or height <= thickness or leg_inset < 0:
        raise ValueError("Invalid table dimensions or clearance")
    if 2 * (leg_inset + leg_width) >= min(length, width):
        raise ValueError("Table legs overlap; reduce inset/leg_width")
    top_size = (length / 2, width / 2, thickness / 2)
    leg_size = (leg_width / 2, leg_width / 2, (height - thickness) / 2)
    rgba = _vec(config.get("rgba", [0.53, 0.32, 0.15, 1.0]), 4, "table.rgba")
    friction = _vec(config.get("friction", [1.2, 0.01, 0.001]), 3, "table.friction")

    print(f"[Table] {length:.3f} x {width:.3f} m; top={position[2]+height:.3f} m; underside={position[2]+height-thickness:.3f} m; thickness={thickness:.3f} m")
    table = ET.SubElement(worldbody, "body", name="wood_table", pos=_text(position))
    table_attrs = {
        "type": "box",
        "size": _text(top_size),
        "pos": f"0 0 {height - thickness / 2:g}",
        "rgba": _text(rgba),
        "friction": _text(friction),
    }
    ET.SubElement(table, "geom", name="table_top", **table_attrs)
    leg_z = leg_size[2]
    x = top_size[0] - leg_inset - leg_size[0]
    y = top_size[1] - leg_inset - leg_size[1]
    for index, leg_pos in enumerate(((x, y), (x, -y), (-x, y), (-x, -y))):
        ET.SubElement(
            table,
            "geom",
            name=f"table_leg{index}",
            type="box",
            size=_text(leg_size),
            pos=f"{leg_pos[0]:g} {leg_pos[1]:g} {leg_z:g}",
            rgba=_text(rgba),
            friction=_text(friction),
        )


def _add_vertical_plate(worldbody: Any, config: Any) -> None:
    position = _vec(config.get("pos", [0.0, 0.0, 0.8]), 3, "plate.pos")
    size = _vec(config.get("size", [0.45, 0.32, 0.035]), 3, "plate.size")
    slide_range = _vec(config.get("slide_range", [-1.2, 0.0]), 2, "plate.slide_range")
    rgba = _vec(config.get("rgba", [0.12, 0.72, 0.24, 0.8]), 4, "plate.rgba")
    mass = float(config.get("mass", 8.0))

    plate = ET.SubElement(worldbody, "body", name="escape_plate", pos=_text(position))
    ET.SubElement(
        plate,
        "joint",
        name="escape_plate_slide",
        type="slide",
        axis="0 0 1",
        limited="true",
        range=_text(slide_range),
        damping="60",
    )
    ET.SubElement(
        plate,
        "geom",
        name="escape_plate_geom",
        type="box",
        size=_text(size),
        mass=f"{mass:g}",
        rgba=_text(rgba),
        friction="1.2 0.01 0.001",
        priority="1",
        solref="0.01 1",
        solimp="0.98 0.995 0.001 0.5 2",
    )


def load_model(xml_path: str | Path, scene: Any) -> mujoco.MjModel:
    """Load the robot and append the requested static or sliding obstacle."""
    xml_path = Path(xml_path).resolve()
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"robot XML has no worldbody: {xml_path}")
    scene_name = str(scene.get("name", "none"))
    if scene_name in ("table", "table_and_plate"):
        _add_table(worldbody, scene.get("table", {}))
    if scene_name in ("plate", "table_and_plate"):
        _add_vertical_plate(worldbody, scene.get("plate", {}))
    if scene_name not in ("none", "table", "plate", "table_and_plate"):
        raise ValueError(
            "scene.name must be one of none, table, plate, table_and_plate"
        )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=xml_path.parent, delete=True
    ) as generated_xml:
        ET.ElementTree(root).write(generated_xml, encoding="unicode")
        generated_xml.flush()
        return mujoco.MjModel.from_xml_path(generated_xml.name)


def apply_initial_state(model: mujoco.MjModel, data: mujoco.MjData, config: Any) -> None:
    """Apply configured robot and obstacle generalized coordinates."""
    preset = str(config.get("preset", "none"))
    if preset != "none":
        if preset not in ("plate_supine_crossed_arms", "plate_supine_default_arms"):
            raise ValueError(
                "initial_state.preset must be none, plate_supine_crossed_arms, "
                "or plate_supine_default_arms"
            )
        if model.nq < 36 or model.nv < 35:
            raise ValueError("plate_supine_crossed_arms requires a 29-DoF floating-base G1")
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qpos[:3] = _vec(
            config.get("preset_root_pos", [0.0, 0.0, 0.12]),
            3,
            "initial_state.preset_root_pos",
        )
        yaw_deg = config.get("preset_yaw_deg")
        yaw = (
            np.deg2rad(float(yaw_deg))
            if yaw_deg is not None
            else float(config.get("preset_yaw", 0.0))
        )
        half_yaw = yaw / 2.0
        data.qpos[3:7] = (
            np.cos(np.pi / 4.0) * np.cos(half_yaw),
            np.cos(np.pi / 4.0) * np.sin(half_yaw),
            -np.cos(np.pi / 4.0) * np.cos(half_yaw),
            np.cos(np.pi / 4.0) * np.sin(half_yaw),
        )
        data.qpos[7:36] = _DEPLOY_DEFAULT_JOINT_POS
        if preset == "plate_supine_crossed_arms":
            data.qpos[7 + 15:7 + 29] = _SUPINE_CROSSED_ARM_POS

    qpos = config.get("qpos")
    if qpos is not None:
        values = np.asarray(qpos, dtype=np.float64)
        if values.size != model.nq:
            raise ValueError(f"initial_state.qpos must have {model.nq} values")
        data.qpos[:] = values

    qvel = config.get("qvel")
    if qvel is not None:
        values = np.asarray(qvel, dtype=np.float64)
        if values.size != model.nv:
            raise ValueError(f"initial_state.qvel must have {model.nv} values")
        data.qvel[:] = values

    plate_qpos = config.get("plate_slide_qpos")
    if plate_qpos is not None:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "escape_plate_slide"
        )
        if joint_id < 0:
            raise ValueError("initial_state.plate_slide_qpos requires scene.name=plate")
        data.qpos[model.jnt_qposadr[joint_id]] = float(plate_qpos)

    plate_surface_top_z = config.get("plate_surface_top_z")
    if plate_surface_top_z is not None:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "escape_plate_slide"
        )
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "escape_plate_geom"
        )
        if joint_id < 0 or geom_id < 0:
            raise ValueError(
                "initial_state.plate_surface_top_z requires scene.name=plate"
            )
        mujoco.mj_forward(model, data)
        current_top_z = data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2]
        data.qpos[model.jnt_qposadr[joint_id]] += float(plate_surface_top_z) - current_top_z

    mujoco.mj_forward(model, data)