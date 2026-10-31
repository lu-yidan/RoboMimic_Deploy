# geometry PPO verdict  baseline 17/24 -> final 12/24
gained=[]  lost=['boundary_07', 'training_04', 'training_06', 'training_10', 'training_11']

| case | base | final | closure mrad (base -> final) | rise max mm (base -> final) |
|---|---|---|---|---|
| training_00 | fail | fail | None -> None | None -> None |
| training_01 | fail | fail | None -> None | None -> None |
| training_02 | fail | fail | 39.286 -> 39.651 | 260.026 -> 260.027 |
| training_09 | fail | fail | 40.072 -> 40.428 | 260.023 -> 260.024 |
| training_13 | fail | fail | 40.212 -> 40.575 | 259.970 -> 259.972 |
| training_14 | fail | fail | 40.446 -> 40.806 | 259.972 -> 259.974 |
| training_15 | fail | fail | 40.350 -> 40.710 | 260.008 -> 260.010 |

| all cases | base | final |
|---|---|---|
| interior_00 | PASS | PASS |
| interior_04 | PASS | PASS |
| interior_07 | PASS | PASS |
| boundary_00 | PASS | PASS |
| boundary_04 | PASS | PASS |
| boundary_05 | PASS | PASS |
| boundary_06 | PASS | PASS |
| boundary_07 | PASS | fail |
| training_03 | PASS | PASS |
| training_04 | PASS | fail |
| training_05 | PASS | PASS |
| training_06 | PASS | fail |
| training_07 | PASS | PASS |
| training_08 | PASS | PASS |
| training_10 | PASS | fail |
| training_11 | PASS | fail |
| training_12 | PASS | PASS |
