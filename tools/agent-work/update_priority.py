"""Update the revised-priority table: P0 becomes the reference geometry, not the controller."""
from pathlib import Path

path = Path(r'D:\G1Climbing\.dsh-watch\误差预算与v2结论_20261028.md')
text = path.read_text(encoding='utf-8')

old_table = """| 优先级 | 动作 | 预期收益 | 代价 |
|---|---|---|---|
| **P0** | 控制器末段跟踪：6 mm → 3 mm | **−11 mrad 量级**，全组落进 40 以内 | 改控制器，不动 RL |
| P1 | 验收口径落文档：物理判据 + 多周期端到端 + 闭合降级为质量指标 | 消除"5 个假失败" | 已提案，无算力 |
| P2 | 参考摆动间隙 1.4–3.7 mm → ≥10 mm（00/01 的真问题） | 消除 2 个真失败 | 重建审计链 + 重训 |
| P3 | RL 加预算（10 工况 × 30 episode ≈ 10 h）或放宽残差幅度 | 剩余 11.2 里可能再拿 1–3 | 要动 gate 的 24 工况断言 |

**不建议现在做 P3 的原因**：起点 39.8 距阈值 40 只有 0.2 mrad，而 131k 步的随机漂移就有 0.36。
起点不动、只加预算，等于在噪声里找 1–3 mrad。等 P0 把起点做到 ~34，同样的漂移就不再致命。"""

new_table = """| 优先级 | 动作 | 预期收益 | 代价 |
|---|---|---|---|
| **P0** | **改参考几何**：把摆动/落脚间隙抬到 ≥10 mm，让"参考端点"同时满足姿态与接触 | 解开姿态↔接触的冲突，才有机会逼近 28.6 | 重建 plan/audit/prefix 链 |
| P1 | 验收口径落文档（已完成：`验收口径_v2_规范_20261028.md`） | 消除"5 个假失败" | 无 |
| P2 | 校准偏置保持 1.0（已到撞杆边界） | — | 不动 |
| P3 | RL 加预算（10 工况 × 30 episode ≈ 10 h） | 参考改好之后才有意义 | 要动 gate 的 24 工况断言 |

**为什么把"改控制器"从 P0 撤下**：两次尝试（参考积分、力矩前馈）都在施加后 0.3 秒内触发接触失败，
加上"偏置加到 1.25 倍也撞杆"，说明姿态与接触的权衡已经用尽 —— 这不是控制环路的问题。
详见上面「六之补充」。"""

if old_table not in text:
    raise SystemExit('table anchor not found')
path.write_text(text.replace(old_table, new_table), encoding='utf-8')
print('priority table updated')
