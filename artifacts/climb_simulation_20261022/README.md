# G1 hook climbing: selected simulation snapshot

This snapshot packages the selected closure-control policy, black composite hook and rung scene, runtime sources, reference trajectory, and validation evidence. It is a simulation research checkpoint, not a hardware-ready controller.

## Evidence and limits

- Selected policy: `closure_margin_control98304_v1`, 98,304 continuation training steps.
- Validation: 11/12 geometry cases; parent policy: 10/12. This is a finite validation grid, not a general success probability.
- Packaged case: `foot_x+5_z13.75`; full replay duration: 113.66 seconds. Original packaged replay matched all seven golden trajectory arrays.
- Highest-rung closure remains unresolved. The controller uses privileged simulation observations and a scripted reference.
- The ZIP retains its original candidate label. The accompanying selection receipt records its subsequent acceptance as the selected simulation package.
- SHA-256: `050476faa1c820f3a12810ffbab5b96b2614bce6a8b827dc5644b383797297b9`.

## Reproduce in PowerShell

Run from the repository root. Use a separate Python 3.12 environment. The requirements pin the recorded runtime; installing them on a new machine has not been validated. The archive includes its own README and audit records.

```powershell
py -3.12 -m venv .venv
$ClimbPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$Snapshot = ".\artifacts\climb_simulation_20261022"
Expand-Archive -LiteralPath "$Snapshot\closure_control_simulation_candidate_v1.zip" -DestinationPath ".\outputs\downloaded_climb_snapshot"
Get-ChildItem .\outputs\downloaded_climb_snapshot
```

Change into the extracted package directory containing `release_manifest.json` and `requirements-runtime.txt`, then run:

```powershell
& $ClimbPython -m pip install -r requirements-runtime.txt
& $ClimbPython -m deploy_mujoco.run_packaged_geometry --verify-only
& $ClimbPython -m deploy_mujoco.run_packaged_geometry --output user_viewer_replay --viewer
```

The archive preserves exact source bytes used by the audit, independently of Git line-ending conversion. The Python environment is not included. Absolute paths inside historical provenance records describe the original machine; package runtime paths are relative.

For the full research history, see [training report](../../docs/baselines/full_cycle_training_20261018.md). Large experimental outputs, virtual environments, and private infrastructure notes are excluded from this branch upload. Historical report links into `outputs/` refer to local evidence; the portable ZIP contains the evidence needed for its packaged case, not every historical experiment.
