"""Update the handover speech: v2 finished, replace the 'still running' lines with the outcome."""
from pathlib import Path

path = Path(r'D:\G1Climbing\.dsh-watch\交接讲话稿_20261028.md')
text = path.read_text(encoding='utf-8')

text = text.replace(
    '## 四、现在在跑什么（1 分钟）',
    '## 四、v2 的结果与误差预算（2 分钟）')

text = text.replace(
    '3. **正在跑 v2**：`outputs/calibrated_geometry_ppo131072_20261028_v2`，131,072 步，约 21:00 结束，之后自动 24 工况评估 + 四项审计。',
    '3. **v2 已跑完（21:35），结论：无效。** 闭合 mean 38.66（父模型 38.43）、P95 40.70（父 40.34），'
    '与 v1 只差 0.06 mrad。\n'
    '4. **但误差预算拆开了**（同样 5 个工况）：参考端点 28.6 → 控制器跟踪 51.7（**+23.1**）→ 学到的残差 42.1'
    '（**−9.6**）→ 校准偏置 39.8（**−2.3**），还剩 11.2 补不动。\n'
    '5. **而且控制器是静态下垂**：末段 2 秒的跟踪误差波动只有 0.001°，稳态偏移集中在 '
    '`left_ankle_pitch −1.861°`、`right_knee −1.659°`。修法是加积分或力矩前馈，不是调增益。')

text = text.replace(
    '- **今天**：v2 跑完，看平均闭合是否 < 38.4（父模型），P95 是否 < 40.3。',
    '- **P0（最高优先）**：控制器末段跟踪 6 mm → 3 mm（积分/前馈）。预期把闭合整体拉低 11 mrad 量级，'
    '而且**不用重新训练**。这是当前性价比最高的一步。\n'
    '- **P1**：落地《验收口径 v2》（物理判据 + 多周期端到端 + 闭合降级为质量指标）。')

text = text.replace(
    '已经改好奖励（终局有界裕量）并接进审计链，v2 正在跑，评判标准改成"平均闭合下降"。',
    '新奖励已跑完（v2），结论是无效——因为误差预算已经用尽；把误差拆开后发现控制器贡献 +23.1 mrad、'
    '是静太下垂，下一步是给控制器加积分或前馈，而不是继续训。')

path.write_text(text, encoding='utf-8')
print('speech updated; lines:', len(text.splitlines()))
for line in text.splitlines():
    if line.startswith('## '):
        print('  ', line)
