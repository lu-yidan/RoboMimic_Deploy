# G1 开口式钩爪概念模型 V1

这是独立的参数化钩体设计，用于讨论尺寸、装配和仿真接触。不是已通过强度验证的成品，不是厂家手部安装图，也不是现有训练模型的直接替换件。当前训练场景及冻结实验保持不变。

## 设计起点

现有仿真横杆半径15 mm。本模型采用开口向下的圆弧钩体，长侧用于后续连接转接件，短侧保留进出通道。参数为设计假设，后续可调整：

| 参数 | V1 |
|---|---|
| 横杆直径 | 30 mm |
| 内圆弧半径 / 入口宽度 | 22 mm / 44 mm |
| 钩体截面宽度 / 板厚 | 12 mm / 8 mm |
| 外宽 / 总高 | 68 mm / 89 mm |
| 长侧长度 / 短端长度 | 55 mm / 12 mm |
| 两个接口占位孔 | Ø5.5 mm，中心间距16 mm |
| 密度假设 | 7850 kg/m³，未选择具体钢材牌号 |
| 几何质量估计 | 113.8 g，不含转接件、紧固件 |

14 mm是入口宽度与杆直径之差，不等于所有位置、角度下的可容错范围。此版没有弹簧锁扣，不能据几何包络推断抗脱钩能力。边缘倒角、内角过渡、材料、疲劳和局部应力均留待结构设计；本版网格没有倒角。

占位孔位于钩体长侧，局部坐标为X=-28 mm，Z=-44/-28 mm，沿Y贯穿。**这不是G1腕部孔距**。G1末端法兰、连接方向、负载路径、腕部电机能力、线缆和脱钩空间仍待核对。本次没有建造或验证腕部转接件。

## 文件与坐标

- `parameters.json`和`build_hook.py`：参数和可重建源码。
- `model/hook.scad`：可在OpenSCAD中编辑的参数化实体描述。当前电脑未安装OpenSCAD，尚未在该软件内渲染验证；交付STL由Python独立生成。
- `model/hook_mm.stl`：封闭实体网格，坐标单位mm。导入MuJoCo时必须缩放0.001。
- `model/hook_profile_mm.dxf`：单位mm的平面轮廓，使用真实半圆弧和孔圆；无加工公差和材料指定。
- `model/preview.png`：实际模型网格的三维图及正视尺寸示意。
- `model/hook_fit.xml`：单位m的独立MuJoCo几何检查夹具，横杆为固定体，钩体为自由体；没有机器人腕部。
- `model/manifest.json`：质量、质心、惯量、体积和输出文件SHA。
- `model/fit_check.json`：几何进入与承载圆弧接触检查。不是力学试验。

局部X跨越钩口，Y沿横杆轴线/板厚，Z向上，原点为内圆弧圆心。不能沿用旧箱体钩爪的安装位姿，否则接触点会改变。

MuJoCo将多数网格碰撞按凸包处理。直接使用整个凹钩作为单一碰撞网格会封住入口，因此本模型将显示与碰撞分开：完整STL只显示，碰撞使用24段凸圆弧和两段矩形柱。碰撞忽略安装孔，内弧径向近似误差上界约0.0471 mm。质量/质心/惯量来自带孔的封闭实体，不从重叠碰撞体反算。

## 已完成的检查

Python构建检查：单个连通封闭实体、面朝向一致、STL重新读入、与解析体积比较、惯量正定、DXF重新读入。几何检查脚本核验MuJoCo编译后的质量/质心/惯量，并沿161个规定位置检查中心对准的横杆进入路径和最终承载面接触。没有测试动态挂载、斜向进入、腕部装配或强度。

STEP未交付：CadQuery已尝试安装，但其OpenCascade内核被本机Windows应用控制策略阻止加载，普通与正常本地运行均失败。未修改系统安全策略。后续可在可用CAD环境中由DXF/SCAD重建并导出STEP。

## 重建

在仓库根目录使用独立CAD环境，不更改训练环境：

```powershell
py -3.12 -m venv .venv-cad
.\.venv-cad\Scripts\python.exe -m pip install -r cad\g1_hook_v1\requirements.txt
.\.venv-cad\Scripts\python.exe -X utf8 cad\g1_hook_v1\build_hook.py --out outputs\hook_v1_rerun
.\.venv\Scripts\python.exe cad\g1_hook_v1\check_fit.py --model outputs\hook_v1_rerun
```

输出目录必须是新目录。修改参数后应采用新的模型版本，不能直接覆盖既有训练场景。

## 仓库说明

用户本次指定参考仓库为[ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)。2026-10-09只读核对得到master/HEAD为`3a72ec1a55dcf155be88c9dcc4a32ecb5c11e313`，没有查询到`lu_pub`分支。

当前工作区仍是此前从`lu-yidan/RoboMimic_Deploy`的`lu_pub`创建的本地`liuze-climb`。此次未更改origin、切换基点或合并另一个仓库；也未声称两个仓库版本相同。钩爪模型是本项目新增设计，不来自该部署仓库。

下一步先核对G1腕部接口并设计转接件，然后在独立场景检验挂入、承载和脱钩，再决定是否替换机器人上的旧简化钩爪。
