"""Record today's 1.25x finding and the retrain null result, release locks, message Codex."""
import json
from datetime import datetime
from pathlib import Path

COLLAB = Path(r'D:\G1Climbing\collab')
board = COLLAB / 'BOARD.md'
text = board.read_text(encoding='utf-8')

stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
entry = f"""
## DSH 追加（{stamp}）：残差权限 1.25× 的发现与重训空结果

| 项 | 结果 |
|---|---|
| 可用权限上限 | 1.25×（18.75 mrad）通过；1.5×/2.0× 失败。2× 重训时 63% 的 episode 在 30 s 内崩 |
| 冻结策略 1.0×（新参考） | mean 38.115 / P95 39.852 / max 40.092，24/24 |
| 冻结策略 1.25×（新参考） | mean 37.920 / P95 38.895 / max 38.943，24/24（**零重训，改一个常量**） |
| **重训 1.25×（131,072 步，新参考）** | **mean 37.905 / P95 38.800 / max 38.844，24/24** |

**结论：重训相对"仅把常量改成 1.25×"只多了 mean −0.015 / P95 −0.095 / max −0.099。**
高闭合工况下降、低闭合工况被抬高的**分布重塑模式与冻结策略完全一致**——策略没有学会"选择性使用权限"。

因此"训练能不能学到东西"的答案是：**在当前观测下不能**。瓶颈不在动作空间大小，也不在奖励形状，
而在于策略对几何/接触**没有任何可条件化的输入**（T8 与力通道消融都证明了这一点）。
唯一剩下的杠杆是 T10（感知→动作通路）。

产物：`.dsh-watch/authority_sweep/`（四档曲线）、`authority_confirm/`（4 工况）、
`authority_all24/`（冻结 1.25× 全 24）、`retrained_all24/`（重训 1.25× 全 24）、
`上机前三项检查_20261029.md`、`接触重锚定_设计_20261029.md`。
"""

marker = '## 共识 / 已锁定的事实'
if marker in text and 'DSH 追加' not in text:
    text = text.replace(marker, entry + '\n' + marker, 1)
    board.write_text(text, encoding='utf-8')
    print('board updated')

message = COLLAB / 'INBOX' / 'codex'
message.mkdir(parents=True, exist_ok=True)
name = datetime.now().strftime('%Y%m%dT%H%M') + '_from-dsh_权限1p25与重训空结果.md'
body = f"""# 残差权限 1.25× 与重训空结果（{stamp}）

## 1. 可用权限曲线（冻结策略，training_13，新参考）

| 增益 | 权限 | 判定 | 闭合 |
|---|---|---|---|
| 1.00× | 15.0 mrad | 通过 | 39.866 |
| 1.25× | 18.75 mrad | 通过 | 38.643 |
| 1.50× | 22.5 mrad | 失败 | 40.553 |
| 2.00× | 30.0 mrad | 失败 | 44.890 |

2× 重训（做了 54% 后被我终止）：63% 的 episode 在 30 s 内结束，回报 17 vs 71，无恢复趋势。
原因：冻结策略的动作按 ±15 mrad 标定，增益翻倍后开局即失稳。

## 2. 24 工况（新参考 `swing_lift_phase_sine_plan_20261029_v2`）

| 组 | mean | P95 | max | 失败 |
|---|---|---|---|---|
| 冻结 1.0× | 38.115 | 39.852 | 40.092 | 0 |
| 冻结 1.25× | 37.920 | 38.895 | 38.943 | 0 |
| 重训 1.25×（131,072 步） | **37.905** | **38.800** | **38.844** | 0 |

**重训的增量是噪声级（mean −0.015）。** 分布重塑模式（高闭合降、低闭合升）与冻结策略完全一致，
说明它学到的只是"把同一把尺子放大 25%"，没有学会选择性。

## 3. 因此的结论

在当前观测下，训练**没有可学的信号**：策略对杆位与接触状态都没有可条件化的输入（T8 的投影为零 +
力通道消融为零影响）。剩下的唯一杠杆是 T10（感知→动作通路）。

## 4. 顺带三条上机结论（细节见 `.dsh-watch/上机前三项检查_20261029.md`）

- 物理容差约 5 mm，指标容差 <0.5 mm——**同一扰动下差 10 倍**，这是"单周期闭合不能当验收"的定量证据。
- 策略**不依赖**理想接触力（置零/加噪/冻结后闭合变化 ≤0.16 mrad）。
- **PD + 重力补偿在 0.1 秒内崩**（自由落体）→ 浮动基必须有接触力层，QP 不可省。

## 5. 已释放锁

`repo_edit`、`cpu4` 均已释放。我全程未改动仓库代码，也未改动 `outputs/` 下任何已审计产物。
"""
(message / name).write_text(body, encoding='utf-8')
print('message written:', name)

for lock in ('repo_edit', 'cpu4'):
    path = COLLAB / 'LOCKS' / lock
    if path.exists():
        import shutil
        shutil.rmtree(path)
        print('released', lock)
print('remaining locks:', [p.name for p in (COLLAB / 'LOCKS').iterdir()] if (COLLAB / 'LOCKS').exists() else [])
