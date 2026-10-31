"""Rewrite the T5/T6 rows of the collaboration board by line index (exact-match safe)."""
from pathlib import Path

path = Path(r'D:\G1Climbing\collab\BOARD.md')
lines = path.read_text(encoding='utf-8').splitlines()

assert lines[31].startswith('| T5 |'), lines[31]
assert lines[32].startswith('| T6 |'), lines[32]

lines[31] = (
    '| T5 | 控制器静态下垂（+23.1 mrad） | DSH | **blocked（结构性）** | '
    '四种补法全败、失败方式各异：参考积分三组 `base_tracking_loss`（94.9/96.2/103.1 s）、'
    '力矩前馈 125 N 接触、降 QP 力矩权重反使闭合 39.87→43.46/43.92。'
    '该下垂是控制器结构固有属性，需换 WBC 结构。见 `.dsh-watch/T5_下垂能不能补掉_20261029.md` |')
lines[32] = (
    '| T6 | 首周期 → 后续周期耦合诊断 | DSH | done | '
    '24×8 矩阵：首周期最差 24/24、首周期 38.115 vs 周期 2–8 28.023、corr +0.332（不传染）、'
    '周期 2–8 漂移 +0.136/周期、累计抬升无累积（恒少 1.47 mm）。'
    '见 `.dsh-watch/T6_周期耦合诊断_20261029.md`；Codex 侧两项 two-cycle 运行仍缺 summary.json，记 paused |')

for i, line in enumerate(lines):
    if line.startswith('- **T6**：用四段数据'):
        lines[i] = '- ~~T6~~、~~T5~~：均已完成（见上方状态表）。T5 结论为**结构性 blocked**，不再做参数扫描。'
    elif line.startswith('- **T5**：把前馈补偿按'):
        lines[i] = '- 两把锁已全部释放；DSH 未改动仓库代码，也未改动 `outputs/` 下任何已审计产物。'

path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
for i in (31, 32):
    print(lines[i][:100])
