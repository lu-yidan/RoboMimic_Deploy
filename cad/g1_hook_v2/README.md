# 黑色组合式钩爪 V2

根据用户要求，将V1单板钩体改为黑色、多零件组合结构。这里的“复合结构”按金属侧板、跨板支撑和独立聚合物衬块的组合解释；没有宣称碳纤维或其他复合材料层合结构已经设计完成。

## 组成

- 两片黑色侧钩板：每片4 mm厚，内侧间距18 mm，总宽26 mm；底端和钩尖采用平面切角。
- 顶部跨板支撑件：连接两片侧板，预留两处贯穿孔。
- 独立承载衬块：内径44 mm，供后续研究可更换接触材料；具体锁固、止退或燕尾连接尚未设计，此版只显示其与支撑件相接的面。
- 两个后部间隔套及四组简化紧固件。螺纹、内六角和预紧力没有建模，头部与螺母采用实心六角代理，不能作为标准件加工图。

18个独立网格零件都通过封闭性与朝向检查。GLB保存各零件和黑色材质，已重新读入验证零件数。整件STL只是多零件装配网格，不是经过布尔融合的单一打印实体；STL格式不保存黑色外观。

名义横杆直径30 mm，衬块承载内径44 mm；侧板钩口48 mm。外宽72 mm、总高91 mm；包括简化紧固件的轴向宽度31 mm。质量约179 g，按钢侧板/紧固件7850、铝支撑/间隔套2700、聚合物衬块1200 kg/m³的假设计算，不含腕部转接件。具体材料和表面处理未定，黑色仅代表目标外观。

## 查看与修改

- `model_final/preview.png`：实际网格绘制的装配图与分解图。
- `model_final/assembly.glb`：米制、保留零件和黑色材质的三维装配模型。
- `model_final/assembly_mm.stl`：毫米制装配网格；分零件STL在`model_final/parts`。
- `parameters.json`与`build_model.py`：参数化建模源码。
- `model_final/manifest.json`：零件清单、质量估计、输出SHA。
- `model_final/hook_fit.xml`与`fit_check.json`：MuJoCo几何检查夹具与结果。

从仓库根目录重建，必须使用新输出目录：

```powershell
.\.venv-cad\Scripts\python.exe -m pip install -r cad\g1_hook_v2\requirements.txt
.\.venv-cad\Scripts\python.exe -X utf8 cad\g1_hook_v2\build_model.py --out outputs\hook_black_v2_rerun
```

## 验证范围

检查夹具使用mocap规定钩体位姿，并显式指定横杆碰撞对。161个垂直进入位置验证了中心对准时的进入通道和最终衬块接触。碰撞体采用凸弧段、矩形柱及圆柱端带半球的保守代理；忽略孔、切角、螺纹，不能用于证明精确的装配或脱钩空间。

没有执行动力学承载测试、强度分析、螺栓连接分析或G1腕部装配验证。腕部接口与衬块固定方式仍是下一轮结构设计内容。本次不修改V1或既有机器人训练场景，不把外形升级解释为已验证的性能提升。STEP仍未生成，沿用V1记录的本机CAD内核限制。

## G1 仿真装配预览（2026-10-09）

新增独立场景 `outputs/g1_hook_v2_mount_v2`，左右腕部使用V2的18零件几何及质量/惯量。仅作静态装配检查，原训练场景和策略不变。下述装配工作更新了上文“尚未做仿真装配”的状态，但没有验证真实腕部机械接口。

第一次装配发现尾板与腕壳的凸碰撞代理重叠4.23 mm。将腕部安装位置沿腕部局部X增加12 mm，使用左右臂7自由度逆运动学保持钩爪原有世界位姿，最大关节调整8.35度。最终尾板与腕壳代理间隙7.77 mm；衬块与横杆初始间隙约0.183 mm，此姿态尚未加载。显式检查包含被父子碰撞过滤排除的腕部几何；没有检测到超过0.01 mm的钩体外部穿透。一对网格距离查询返回零，世界坐标包围盒仍有约75 mm分离下界，单独记录在检查报告中。

琥珀色半透明方框是转接件占位，不参与质量或碰撞。实际转接件、衬块固定、紧固件强度与载荷路径仍待设计。所有零件在仿真中刚性附着；此预览不运行动力学、不调用策略，不证明挂接承重或运动范围通过。

从仓库根目录依次重建（已有目录会拒绝覆盖）：

```powershell
.\.venv-cad\Scripts\python.exe cad\g1_hook_v2\build_g1_mount.py
.\.venv\Scripts\python.exe cad\g1_hook_v2\fit_g1_mount.py
.\.venv\Scripts\python.exe cad\g1_hook_v2\check_g1_mount.py
```

使用学长平台原入口打开检查模式：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u deploy_mujoco\deploy_mujoco.py --config-name mujoco_climb_v2_preview hydra.job.chdir=false
```

按1看全身、2看左钩、3看右钩、A切换转接件占位、R复位。检查结果为`clearance_check.json`，完整装配预览为`g1_mount_preview.png`。下一步优先完成腕部转接件与衬块固定结构，再做受控挂入、加载、脱钩测试。
