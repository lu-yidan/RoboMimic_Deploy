# 可见横杆几何候选

本模块迁自旧项目的冻结 RGB-D 基线。35 帧在同一解释器下与原实现、存档预测均逐项完全一致，证据见 `outputs/bar_perception_migration_20261029_v2/summary.json`。仓库内保留五个代表性回归样本。

在仓库根目录运行：

```powershell
python -m pip install -r requirements-perception.txt
python -m perception --input-dir tests/fixtures/bar_perception/clean_initial --output outputs/perception_demo
python -m unittest discover -s tests -p test_bar_perception_portable.py
```

输出目录必须不存在。`prediction.json` 保存可见段和段内候选点；`manifest.json` 记录输入与源文件哈希。模块只依赖 NumPy、SciPy，无需加载 MuJoCo、机器人模型或训练框架。

输入目录包含 `measurement.npz`（`rgb`、`depth`、`camera_position`、`camera_rotation`）与 `calibration.json`（`width`、`height`、`fx`、`fy`、`cx`、`cy`）。深度为米制光轴深度；采用 MuJoCo/OpenGL 相机坐标：x 向右、y 向上、z 向后。相机旋转把局部坐标变换到世界坐标。真实深度相机必须先转换坐标约定并验证标定。

当前约束是受控橙色、已知 15 mm 杆半径、已知相机位姿与针孔内参。质量分数是启发式量，不是概率。端点表示实际可见范围，不推断遮挡后的整杆端点。25 mm 端部裕量是开发参数，尚未由实际钩爪尺寸验证。

输出始终不授予目标身份或承载许可。平面干扰和严重遮挡样本用于检查拒绝行为。本模块尚未接入 PPO 或闭环攀爬；保存图像的回归通过不能证明真实相机或两米闭环感知成功。
