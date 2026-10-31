"""Migrate the frozen, image-only legacy estimator; retain historical outputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'g1_bar_perception.py':'28628a32bcd00c942b226781769921553b871a05c2b6b259c5641ba97819b392',
          'g1_bar_perception_v2.py':'a4e64a6b7ab445ee2486192330024914dcf1e81ac88fdb0a826967a34710080e'}
FIXTURES=('clean_initial','high_bar_plus80','orange_planar_distractor','partial_bar_occlusion','heavy_occlusion')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf8')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-existing',action='store_true')
    a=p.parse_args();legacy=a.legacy_root.resolve()
    study=legacy/'results/g1_bar_perception_v2_20261007';measurements=legacy/'results/g1_bar_perception_20261007'
    summary=read(study/'summary.json');assert summary['source_unchanged'] and summary['case_count']==35
    snapshot=study/'source_snapshot'
    for name,digest in EXPECTED.items():assert sha(snapshot/name)==summary['source_sha256'][name]==digest
    package=ROOT/'perception';fixtures=ROOT/'tests/fixtures/bar_perception'
    text=(snapshot/'g1_bar_perception_v2.py').read_text(encoding='utf8')
    assert text.count('from g1_bar_perception import')==1
    transformed=text.replace('from g1_bar_perception import','from .orange_bar import')
    init='"""Image-only visible-rod proposals; no target identity or loading permission."""\nfrom .orange_bar import CameraCalibration\nfrom .visible_targets import select_visible_grasp_targets\n'
    if a.verify_existing:
        assert sha(package/'orange_bar.py')==EXPECTED['g1_bar_perception.py']
        assert (package/'visible_targets.py').read_text(encoding='utf8')==transformed
        assert (package/'__init__.py').read_text(encoding='utf8')==init
        assert fixtures.exists()
    else:
        assert not package.exists() and not fixtures.exists()
        package.mkdir();fixtures.mkdir(parents=True)
        shutil.copy2(snapshot/'g1_bar_perception.py',package/'orange_bar.py')
        (package/'visible_targets.py').write_text(transformed,encoding='utf8')
        (package/'__init__.py').write_text(init,encoding='utf8')
    old_base=load('g1_bar_perception',snapshot/'g1_bar_perception.py')
    old=load('_legacy_visible_targets',snapshot/'g1_bar_perception_v2.py')
    from perception import CameraCalibration,select_visible_grasp_targets
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for item in summary['cases']:
        name=item['name'];record=read(study/name/'result.json');path=measurements/name/'measurement.npz'
        assert sha(path)==record['source_measurement_sha256']
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in ('rgb','depth','camera_position','camera_rotation')}
        calibration=record['prediction']['v1_prediction']['calibration']
        args=(arrays['rgb'],arrays['depth'],CameraCalibration(**calibration),arrays['camera_position'],arrays['camera_rotation'])
        current=select_visible_grasp_targets(*args)
        original=old.select_visible_grasp_targets(arrays['rgb'],arrays['depth'],old_base.CameraCalibration(**calibration),
            arrays['camera_position'],arrays['camera_rotation'])
        encode=lambda x:json.dumps(x,sort_keys=True,allow_nan=False)
        exact=encode(current)==encode(original);archived=encode(current)==encode(record['prediction'])
        assert exact and not current['load_permission']
        rows.append(dict(case=name,ported_matches_frozen_source=exact,matches_archived_prediction=archived,
            measurement_sha256=sha(path),historical_result_sha256=sha(study/name/'result.json'),targets=len(current['targets'])))
        write(a.output/(name+'.json'),current)
        if name in FIXTURES:
            f=fixtures/name
            if a.verify_existing:
                assert sha(f/'measurement.npz')==sha(path)
                assert read(f/'calibration.json')==calibration and encode(read(f/'expected_prediction.json'))==encode(current)
            else:
                f.mkdir();shutil.copy2(path,f/'measurement.npz')
                write(f/'calibration.json',calibration);write(f/'expected_prediction.json',current)
    report=dict(passed=all(r['ported_matches_frozen_source'] for r in rows),frames=len(rows),rows=rows,
        archived_predictions_exact=sum(r['matches_archived_prediction'] for r in rows),
        source_hashes=EXPECTED,ported_source_hashes={p.name:sha(p) for p in package.glob('*.py')},
        historical_summary_sha256=sha(study/'summary.json'),migration_source_sha256=sha(__file__),
        numpy_version=np.__version__,scipy_version=scipy.__version__,new_simulation_steps=0,new_training_steps=0,
        scope='Same 35 historical RGB-D frames, not a new dataset or climbing trial. Algorithm unchanged except relative import and module names. Current frozen-source replay equivalence is separate from historical prediction equality. Image-only proposals assume controlled orange color, known 15 mm radius, supplied camera pose and optical-axis depth. No identity, loading authority or integration into adopted PPO.')
    write(a.output/'summary.json',report)
    write(package/'provenance.json',dict(origin='Local G1Climbing results/g1_bar_perception_v2_20261007/source_snapshot',
        source_hashes=EXPECTED,ported_source_hashes=report['ported_source_hashes'],
        transformation='Module rename and one import changed to package-relative; algorithm unchanged.',
        fixture_files={str(p.relative_to(fixtures)).replace('\\','/'):sha(p) for p in fixtures.rglob('*') if p.is_file()}))
    (a.output/'migration_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=report['passed'],frames=len(rows),archived_predictions_exact=report['archived_predictions_exact'],fixture_count=5)),flush=True)


if __name__=='__main__':main()
