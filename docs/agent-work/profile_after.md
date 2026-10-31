# 剖面报告：训练后（131072 步，24 工况）

- 工况数 24；物理判定通过 22/24（物理失败：['training_00', 'training_01']）
- 闭合分布：mean 38.72 / median 39.71 / P95 40.73 / max 40.83 mrad
- 越过预警线 35 mrad 的：19 个；越过旧拟合线 40 mrad 的：8 个

| 工况 | 分类 | 闭合 mrad | 抬升 mm | 端点误差 mm | 倾斜 ° |
|---|---|---|---|---|---|
| training_14 | closure_over_old_line | 40.834 | 259.974 | 6.04 | 0.73 |
| training_15 | closure_over_old_line | 40.739 | 260.010 | 6.01 | 0.76 |
| training_13 | closure_over_old_line | 40.609 | 259.972 | 6.04 | 0.73 |
| training_09 | closure_over_old_line | 40.455 | 260.025 | 6.03 | 0.76 |
| training_11 | closure_over_old_line | 40.384 | 259.971 | 6.05 | 0.74 |
| training_06 | closure_over_old_line | 40.367 | 259.993 | 6.05 | 0.74 |
| boundary_07 | closure_over_old_line | 40.292 | 259.982 | 6.05 | 0.74 |
| training_10 | closure_over_old_line | 40.132 | 259.966 | 6.05 | 0.74 |
| training_08 | closure_over_warning | 39.961 | 259.965 | 6.06 | 0.74 |
| training_05 | closure_over_warning | 39.918 | 259.982 | 6.06 | 0.74 |
| training_07 | closure_over_warning | 39.735 | 259.959 | 6.06 | 0.74 |
| training_02 | closure_over_warning | 39.679 | 260.028 | 6.05 | 0.76 |
| boundary_06 | closure_over_warning | 39.317 | 259.785 | 5.97 | 0.75 |
| training_04 | closure_over_warning | 38.887 | 259.985 | 5.91 | 0.75 |
| training_12 | closure_over_warning | 38.742 | 259.940 | 5.89 | 0.75 |
| training_03 | closure_over_warning | 38.436 | 259.968 | 5.91 | 0.75 |
| boundary_05 | closure_over_warning | 37.534 | 259.868 | 6.08 | 0.74 |
| boundary_04 | closure_over_warning | 36.677 | 259.826 | 5.96 | 0.73 |
| boundary_00 | closure_over_warning | 35.959 | 259.535 | 5.79 | 0.61 |
| interior_00 | pass | 34.793 | 254.540 | 9.81 | 0.63 |
| interior_04 | pass | 34.745 | 256.981 | 7.57 | 0.67 |
| interior_07 | pass | 33.598 | 259.591 | 5.76 | 0.59 |
| training_00 | incomplete | — | — | — | — |
| training_01 | incomplete | — | — | — | — |
