"""Rewrite the mitigation section of the pre-hardware report after the 3b correction."""
from pathlib import Path

path = Path(r'D:\G1Climbing\.dsh-watch\上机前三项检查_20261029.md')
text = path.read_text(encoding='utf-8')

old = """### 对策（更新）

单纯"慢速下降 + 接触即停"**不够**——它只能避免压杆，不能保证落点准确。需要：

1. **接触驱动 + 参考重锚定**：检测到接触后，用实际接触点反推落点，**重新锚定剩余参考**
   （把计划里"应该落在哪"换成"实际落在哪"），再继续走。这样 2–5 mm 的误差被吸收在锚定里。
2. **外部标定**：关节零位到 0.1°、连杆到 0.2%，写进 `deploy_real/config/real.yaml`。
3. 足底软垫，把硬碰变软接触（降低冲击，但不解决落点误差）。"""

new = """### 对策（按更正后的事实重写）

既然物理可行、只有指标脆，那么**优先修指标，而不是修运动**：

1. **真机验收口径重定**（必做，零成本）：用"不摔 + 落在预期杆 + 抬升 250±10 mm + 无非预期接触"
   判通过；闭合只作质量指标，并**相对实际接触构型测量**，不相对仿真初始位姿。
2. **外部标定**（建议）：关节零位到 0.1°、连杆到 0.2%，写进 `deploy_real/config/real.yaml`——
   能把闭合抬高的幅度压小，但不是通过的必需条件。
3. **落点搜索 / 参考重锚定**（可选，优先级降级）：只在"希望真机也保住闭合质量指标"时才需要；
   对"能不能上机跑通"不再是阻塞项。"""

if old not in text:
    raise SystemExit('anchor not found')
path.write_text(text.replace(old, new), encoding='utf-8')
print('mitigation section rewritten')
