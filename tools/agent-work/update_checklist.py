"""Insert the revised priority block at the top of the action checklist."""
from pathlib import Path

path = Path(r'D:\G1Climbing\.dsh-watch\改造落地清单_20261028.md')
lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[0].startswith('# 改造落地清单'), lines[0]
assert lines[7].startswith('## 0.'), lines[7]

head = """# 改造落地清单（2026-10-28）

目标：把当前"几何残差 PPO"从**学不到东西**，推进到**可以正式开展强化学习**的状态。
本清单只列可执行动作、影响面和验证方法。

> **2026-10-28 夜 · 状态更新（本节的优先级已取代下文旧顺序）**
>
> - **v2（终局裕量奖励）实测：无效。** 24 工况闭合 mean 38.66（父模型 38.43），P95 40.70（父 40.34）；
>   与 v1 只差 0.06 mrad。详见 `误差预算与v2结论_20261028.md`。
> - **原因已定位：不是奖励，是误差预算已经用尽。**
>   闭合分层：参考端点 28.6 → 控制器跟踪 51.7（**+23.1**）→ 学到的残差 42.1（**−9.6**）→ 校准偏置 39.8（**−2.3**）。
> - **P0 诊断：控制器是静态下垂（末段 2 s 波动仅 0.001°），不是滞后。**
>   稳态偏移集中在 `left_ankle_pitch −1.861°`、`right_knee −1.659°`。
>   修法：**加积分项或力矩前馈**；只调增益压不下去（e = τ/Kp）。
>
> **修订后的优先级：**
>
> | 优先级 | 动作 | 预期收益 | 代价 |
> |---|---|---|---|
> | **P0** | 控制器末段跟踪：6 mm → 3 mm（积分/前馈） | −11 mrad 量级，全组进 40；**不用重训** | 改控制器 |
> | **P1** | 落地验收口径 v2（见 `验收口径_v2_规范_20261028.md`） | 消除 5 个假失败 | 无算力 |
> | P2 | 参考摆动间隙 1.4–3.7 mm → ≥10 mm（00/01 的真问题） | 消除 2 个真失败 | 重建审计链 + 重训 |
> | P3 | RL 加预算（10 工况 × 30 episode ≈ 10 h）或放宽残差幅度 | 剩余 11.2 里再拿 1–3 | 要动 gate 的 24 工况断言 |
>
> **P3 暂缓**：起点 39.8 距阈值只有 0.2 mrad，而 131k 步的随机漂移就有 0.36 —— 起点不动等于在噪声里找 1–3 mrad。

---

## 0. 先冻结本轮结果（10 分钟）

- v1 输出目录 `outputs/calibrated_geometry_ppo131072_20261027_v1` 与 v2 的 `..._20261028_v2` 都保持只读，
  不要原地再训（两个目录的 manifest/summary 已绑定 hash 链）。
- 训练前审计产物已备份在 `.dsh-watch/backup/20261028_000442/`（15 个文件，sha256 逐个校验过）。
- 判定表与报告：`.dsh-watch/geometry_verdict.json`、`G1_几何PPO诊断报告_20261028.md`、`误差预算与v2结论_20261028.md`。

"""

path.write_text(head + ''.join(lines[13:]), encoding='utf-8')
print('updated; lines:', len(path.read_text(encoding='utf-8').splitlines()))
