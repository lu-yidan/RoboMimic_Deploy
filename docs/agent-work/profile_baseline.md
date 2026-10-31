# 剖面报告：父模型（训练前，24 工况）

- 工况数 24；物理判定通过 22/24（物理失败：['training_00', 'training_01']）
- 闭合分布：mean 38.43 / median 39.43 / P95 40.34 / max 40.45 mrad
- 越过预警线 35 mrad 的：19 个；越过旧拟合线 40 mrad 的：4 个

| 工况 | 分类 | 闭合 mrad | 抬升 mm | 端点误差 mm | 倾斜 ° |
|---|---|---|---|---|---|
| training_14 | closure_over_old_line | 40.446 | 259.972 | 6.13 | 0.73 |
| training_15 | closure_over_old_line | 40.350 | 260.008 | 6.11 | 0.75 |
| training_13 | closure_over_old_line | 40.212 | 259.970 | 6.13 | 0.73 |
| training_09 | closure_over_old_line | 40.072 | 260.023 | 6.12 | 0.76 |
| training_11 | closure_over_warning | 39.993 | 259.969 | 6.14 | 0.73 |
| training_06 | closure_over_warning | 39.973 | 259.990 | 6.14 | 0.73 |
| boundary_07 | closure_over_warning | 39.898 | 259.980 | 6.14 | 0.73 |
| training_04 | closure_over_warning | 39.756 | 259.993 | 6.15 | 0.74 |
| training_10 | closure_over_warning | 39.737 | 259.964 | 6.14 | 0.73 |
| training_08 | closure_over_warning | 39.569 | 259.963 | 6.15 | 0.73 |
| training_05 | closure_over_warning | 39.518 | 259.980 | 6.15 | 0.74 |
| training_07 | closure_over_warning | 39.341 | 259.958 | 6.15 | 0.73 |
| training_02 | closure_over_warning | 39.286 | 260.026 | 6.14 | 0.76 |
| boundary_06 | closure_over_warning | 38.915 | 259.784 | 6.07 | 0.75 |
| training_12 | closure_over_warning | 38.331 | 259.938 | 5.99 | 0.75 |
| training_03 | closure_over_warning | 38.024 | 259.966 | 6.01 | 0.75 |
| boundary_05 | closure_over_warning | 37.148 | 259.867 | 6.17 | 0.74 |
| boundary_04 | closure_over_warning | 36.512 | 259.825 | 6.06 | 0.72 |
| boundary_00 | closure_over_warning | 35.539 | 259.533 | 5.79 | 0.61 |
| interior_00 | pass | 34.706 | 254.541 | 9.81 | 0.63 |
| interior_04 | pass | 34.594 | 256.972 | 7.57 | 0.66 |
| interior_07 | pass | 33.571 | 259.590 | 5.77 | 0.59 |
| training_00 | incomplete | — | — | — | — |
| training_01 | incomplete | — | — | — | — |
