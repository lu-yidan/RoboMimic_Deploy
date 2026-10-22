"""Copy an existing research scene into a portable, hash-bound local bundle."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from policy.climb.contract import ClimbContract

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--license", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "assets/climb_mounted")
    args = parser.parse_args()
    source, out = args.case.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("Preserve existing bundle; select another --out")
    xml = source / "model.xml"
    original = mujoco.MjModel.from_xml_path(str(xml))
    initial = np.load(source / "qpos_initial.npy", allow_pickle=False)
    config = json.loads((ROOT / "policy/climb/config/climb.json").read_text())
    names = config["joint_names"]
    ids = np.array([original.joint(name).id for name in names])
    if (original.nq, original.nv, original.nu, original.neq) != (36, 35, 29, 0):
        raise ValueError("Expected free G1, 29 motors, no equality constraints")
    if initial.shape != (36,) or not np.isfinite(initial).all():
        raise ValueError("Invalid source initial state")
    config["default_q"] = initial[original.jnt_qposadr[ids]].tolist()
    config["joint_lower"] = original.jnt_range[ids, 0].tolist()
    config["joint_upper"] = original.jnt_range[ids, 1].tolist()
    config["scene_variant"] = "mounted_initial_support_pose"
    config["initial_pose_source_sha256"] = sha(source / "qpos_initial.npy")
    config["gain_status"] = "Inherited deployment gains; not validated for climbing"
    contract = ClimbContract(config)
    tree = ET.parse(xml)
    if tree.findall(".//include"):
        raise ValueError("Source must be a standalone XML")
    compiler = tree.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", "."))
    if not mesh_dir.is_absolute():
        mesh_dir = xml.parent / mesh_dir
    mesh_inputs = {}
    files = []
    for node in tree.findall("asset/mesh"):
        path = (mesh_dir / node.attrib["file"]).resolve()
        if path.name in mesh_inputs and mesh_inputs[path.name] != sha(path):
            raise ValueError("Conflicting mesh basenames")
        mesh_inputs[path.name] = sha(path)
        files.append((node, path))
    args.license.read_text(encoding="utf-8")
    out.mkdir(parents=True)
    (out / "meshes").mkdir()
    for node, path in files:
        shutil.copy2(path, out / "meshes" / path.name)
        node.set("file", path.name)
    compiler.set("meshdir", "meshes")
    tree.write(out / "model.xml", encoding="utf-8")
    np.save(out / "initial_qpos.npy", initial)
    shutil.copy2(args.license, out / "LICENSE")
    (out / "contract.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    copied = mujoco.MjModel.from_xml_path(str(out / "model.xml"))
    arrays = ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_pos", "geom_quat",
              "geom_size", "geom_friction", "geom_contype", "geom_conaffinity", "jnt_range",
              "actuator_trnid", "actuator_ctrlrange", "mesh_vert", "mesh_face")
    for name in arrays:
        np.testing.assert_array_equal(getattr(original, name), getattr(copied, name))
    manifest = dict(source_model_sha256=sha(xml), source_initial_sha256=sha(source / "qpos_initial.npy"),
        source_mesh_sha256=mesh_inputs, contract_sha256=contract.fingerprint,
        compiled_arrays_exact=list(arrays), physics_dt=float(copied.opt.timestep),
        source_case=str(source), importer_sha256=sha(__file__),
        output_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
        dynamics_success_inherited=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(bundle=str(out), meshes=len(files), contract_sha256=contract.fingerprint)))

if __name__ == "__main__":
    main()
