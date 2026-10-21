# Score model assets

The Score runtime expects an ONNX policy and the reference motion file in this
directory. ONNX files are ignored by Git because they are deployment artifacts.
This branch keeps `motion_kit.zip` as the reproducible bundle of the available
Score ONNX variants and `score_motion.npz`.

After cloning, extract the bundle in place:

```bash
unzip -n policy/score/model/motion_kit.zip -d policy/score/model
```

The default configurations select:

- `score.yaml` -> `policy-obs-aic.onnx` (real robot / bridge)
- `score-aggr.yaml` -> `policy-obs-xb-590.onnx` (MuJoCo ball scene)

To use a different policy, copy its `.onnx` file here and update `onnx_path` in
the selected YAML file. The archive is not a substitute for validating a policy
on the robot; test new model/configuration pairs in MuJoCo first.
