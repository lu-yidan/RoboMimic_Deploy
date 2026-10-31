"""Morning housekeeping: record the oracle anomaly, release the lock."""
import json
from datetime import datetime
from pathlib import Path

COLLAB = Path(r'D:\G1Climbing\collab')
board = COLLAB / 'BOARD.md'
text = board.read_text(encoding='utf-8')
stamp = datetime.now().strftime('%Y-%m-%d %H:%M')

entry = f"""
## DSH 追加（{stamp}）：位移失效曲线 + oracle 评估的未解异常

### 已确定：位移失效曲线（决定性测量）

24 个"几何工况"其实只偏移 `rung_01`，其中 16 个落在 z=15.3–15.4 mm（彼此差 <0.1 mm）。

| z (mm) | 5.0 | 12.0 | 14.0 | 15.0 | 15.3–15.4 | **16.0** | 17.0 | 19.0 | 20.0 |
|---|---|---|---|---|---|---|---|---|---|
| 闭合 | 35.0 | 32.6 | 34.4 | 38.2 | 38.6 | **39.6** | 48.8 | 66.3 | 75.0 |

- 拐点前斜率 1.8 mrad/mm，拐点后 8.8–9.2 mrad/mm（**放大 5 倍**）
- **40 mrad 交叉点 z ≈ 16.1 mm**（实测；线性外推会错成 19 mm）
- **真实位移裕度只有 0.7 mm**；声明域 ±20 mm 比真实指标能力大 25%
- 结论已写入 `验收口径_v2_补充_位移域_20261029.md`

### 未解异常：两套不同策略跑出逐字节相同的轨迹

oracle 微调（131,072 步）产出 checkpoint `e8f731a8`，与 blind 的 `101d987a` **不同**；
评估时 arm 里装的也确实是各自的策略；但 24 个工况的闭合**每一位都相同**，
`qpos`/`actions` 数组 `np.array_equal` 为 True。

已排除：
- 前后缀路由（`suffix_start=70.2`，闭合窗口 111.66–113.66 s 在 suffix 区，没有走 prefix）
- checkpoint 未更新（`policy.zip` ≠ `policy_before.zip`，两者都变）
- 装载错文件（arm 内 policy 的 sha 与各自 run 一致）

**下一步诊断**（30 分钟内）：固定一个观测，直接比对两个 checkpoint 的输出动作；
若输出不同而轨迹相同，说明缺陷在评估链；若输出也相同，说明两个 checkpoint 在确定性前向路线上等价。

**在异常解释清楚之前，"oracle 有没有用"这个问题没有答案**；但位移曲线、权限曲线、
上机三项检查都不受影响（它们各自的数值互不相同，来自真实有差异的 run）。
"""

marker = '## 共识 / 已锁定的事实'
if marker in text and '未解异常' not in text:
    text = text.replace(marker, entry + '\n' + marker, 1)
    board.write_text(text, encoding='utf-8')
    print('board updated')

message = COLLAB / 'INBOX' / 'codex'
message.mkdir(parents=True, exist_ok=True)
name = datetime.now().strftime('%Y%m%dT%H%M') + '_from-dsh_位移曲线与oracle异常.md'
body = f"""# 位移失效曲线 + oracle 评估异常（{stamp}）

## 1. 位移失效曲线（可直接用）

把 24 个工况摊开后发现它们只偏移 `rung_01`，16 个挤在 z=15.3–15.4 mm。
补测合成点后得到：**拐点在 z≈16.1 mm，之后斜率放大 5 倍，20 mm 处闭合 75 mrad**。

- 真实位移裕度 **0.7 mm**
- 声明域 ±20 mm 比真实能力大 25%
- 已写入 `.dsh-watch/验收口径_v2_补充_位移域_20261029.md`
- 数据：`.dsh-watch/displacement_verify_result_v1.json`、`displacement_verify/result.json`

## 2. 未解异常（需要你或我明天查）

oracle 微调产出 `e8f731a8`，blind 重训产出 `101d987a`，两者不同，评估时也各自装载正确，
但 24 工况闭合**逐位相同**，`qpos`/`actions` `array_equal` 为 True。

已排除前后缀路由、checkpoint 未更新、装错文件三项。
建议下一步：固定一个观测，直接比对两个 checkpoint 的确定性输出动作。

在这条解释清楚前，**不要引用"oracle 实验"的任何结论**。
"""

(message / name).write_text(body, encoding='utf-8')
print('message written:', name)

lock = COLLAB / 'LOCKS' / 'cpu4'
if lock.exists():
    import shutil
    shutil.rmtree(lock)
    print('released cpu4')
print('remaining locks:', [p.name for p in (COLLAB / 'LOCKS').iterdir()] if (COLLAB / 'LOCKS').exists() else [])
