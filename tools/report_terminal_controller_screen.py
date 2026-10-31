"""Summarize all declared terminal-control screens, including failed trials."""
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    families={
        'leg_feedback':'coordinated_closure_screen_v1',
        'leg_and_body_drop':'body_coordinated_closure_screen_v1',
        'wholebody_feedback':'wholebody_closure_screen_v1',
        'terminal_acceleration_qp':'acceleration_closure_screen_v1',
        'constrained_terminal_qp':'constrained_closure_screen_v1'}
    rows=[]
    for family,name in families.items():
        root=Path('outputs')/name
        protocol=json.loads((root/'protocol.json').read_text())
        expected=len(protocol['gains'])*len(protocol['cases'])*len(protocol.get('drops',[0]))
        audits=json.loads((root/'independent_audits.json').read_text())
        if len(audits)!=expected:raise ValueError('Incomplete declared screen: '+name)
        for a in audits:
            p=root/a['run'];s=json.loads((p/'summary.json').read_text());detail=json.loads((p/'independent_audit.json').read_text())
            if sha(p/'summary.json')!=detail['summary_sha256'] or sha(p/'trajectory.npz')!=detail['trajectory_sha256']:raise ValueError('Audited evidence changed')
            rows.append(dict(family=family,run=str(p),gain=s['gain'],body_drop_m=s.get('body_drop_m'),qp_blend=s.get('qp_blend'),
                time_s=s['runtime']['time_s'],integrity_passed=detail['integrity_passed'],accepted=detail['independently_accepted'],
                joint_closure_mrad=None if s['endpoints'][0]['joint_closure_rad'] is None else 1000*s['endpoints'][0]['joint_closure_rad'],
                rise_max_mm=None if s['root_rise_max_m'] is None else 1000*s['root_rise_max_m'],
                prefix_exact=all(s['prefix'].values()),audit_sha256=sha(p/'independent_audit.json')))
    out=Path('outputs/terminal_controller_screen_report_v1');out.mkdir(exist_ok=False)
    result=dict(rows=rows,trial_count=len(rows),integrity_passed=sum(r['integrity_passed'] for r in rows),
        accepted_count=sum(r['accepted'] for r in rows),selected_baseline_replaced=False,
        scope='Adaptive known-case controller screening only. No additional training. Same +15mm failure case; no held-out evaluation or statistical significance claim.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'report_source.py').write_bytes(Path(__file__).read_bytes())
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for family in families:
        subset=[r for r in rows if r['family']==family and r['joint_closure_mrad'] is not None]
        ax.scatter([r['rise_max_mm'] for r in subset],[r['joint_closure_mrad'] for r in subset],label=family.replace('_',' '))
    ax.axvline(260,color='black',linestyle='--',linewidth=1);ax.axhline(40,color='black',linestyle='--',linewidth=1)
    ax.set(xlabel='Maximum root rise in final 2 seconds (mm)',ylabel='Maximum joint closure error (mrad)',title='Terminal-controller screen on known +15 mm case')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'endpoint_tradeoff.png',dpi=160);fig.savefig(out/'endpoint_tradeoff.pdf');plt.close(fig)
    lines=['# 最高杆收尾控制试验：2026-10-22','',
        '固定已选策略和 +15 mm 已知失败工况，逐步检查关节反馈、机身目标补偿和终端加速度/接触力联合求解。所有候选均从初始状态连续运行至成功或失败，不跳过前缀；未在运行中覆盖状态或施加外力，未放宽原验收门槛。',
        '',f"共 {len(rows)} 项候选；记录完整性通过 {result['integrity_passed']} 项，任务验收通过 {result['accepted_count']} 项。原选定策略及发布包未替换。",
        '', '最高杆问题尚未解决；不能将部分指标通过计为整项成功。' if result['accepted_count']==0 else '存在最高杆通过的候选，但仍须复测原成功工况才能采用。',
        '', '| 方案 | 增益 | 额外参数 | 最大闭合误差/mrad | 最大抬升/mm | 完整性 | 任务通过 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        parameter=f"目标下降 {r['body_drop_m']*1000:g} mm" if r['body_drop_m'] is not None else (f"QP混合 {r['qp_blend']:g}" if r['qp_blend'] is not None else '无')
        j='未到终点' if r['joint_closure_mrad'] is None else f"{r['joint_closure_mrad']:.3f}"
        h='未到终点' if r['rise_max_mm'] is None else f"{r['rise_max_mm']:.3f}"
        lines.append(f"| {r['family']} | {r['gain']:g} | {parameter} | {j} | {h} | {r['integrity_passed']} | {r['accepted']} |")
    lines+=['','原关节闭合门槛为 40 mrad，机身抬升为 250±10 mm；表中两列不能代替足端位置、速度、接触承载、原生限位等完整验收。',
        '', '## 解释与限制','',
        '双腿反馈曾将踝部误差压入门槛，但肩部误差与抬升偏差成为新的限制；降低机身目标可改善高度，却显著增大踝部偏差。这些是当前候选的动态结果，不能据此断言所有有界控制策略都不可行。',
        '', '机身目标补偿增加了原策略动作之外的控制权限；联合QP替换了部分力矩分配，使用真实仿真接触与软粘着近似。二者都不是强化学习收益。联合QP沿用既有模块的100关节跟踪增益，与原终端腿部300增益不同，不能将差异单独归因于接触相容性。',
        '', '所有候选的收尾前缀均要求与原策略七组记录精确一致。保存状态的独立审计复核动作、限位、碰撞、力矩与终点指标；亚步接触力与接触记忆仍由原运行环境检查。未额外声称第二次动力学精确重放。',
        '', '这些工况已用于诊断和选参，不属于独立测试集。只有最高杆通过且原成功工况无退步后才考虑采用新控制器。',
        '', '约束QP进一步加入相对原始姿态38mrad、相对250mm抬升8mm的加速度屏障（omega=2）。这是比任务门槛更紧的模型内部目标；模型可行不代表真实执行满足，也不构成安全证明。复现入口为 `tools.evaluate_constrained_closure`。',
        '', '## 预测与物理响应核对','',
        '对约束QP轨迹108/110/112秒的三个保存状态做独立冷启动重建，QP预测根部向下加速度约0.0115–0.0131 m/s²，而施加对应力矩后的MuJoCo瞬时根部加速度接近零（约−5.90e−5至3.15e−5 m/s²）。模型屏障残差约1e−16，但重建物理响应不满足同一加速度屏障。状态与时间未推进。',
        '', '该检查未恢复求解器暖启动历史，不是运行中加速度的精确重放，也不能单独证明唯一因果。它说明当前QP模型可行与实际接触动力学响应存在差异，后续应记录运行中的预测/实测加速度和接触力，并修正接触约束/力矩实现，再重新验证收尾；不能只继续增大姿态反馈或放宽高度阈值。详见 `outputs/terminal_qp_response_diagnostic_v1/summary.json`。',
        '', '证据：`outputs/terminal_controller_screen_report_v1/summary.json` 及表中对应实验目录；每条轨迹均保留SHA-256和独立审计。',
        '', '复现：在仓库根目录使用 `.venv\\Scripts\\python.exe -m tools.evaluate_coordinated_closure --output outputs/新的目录`；其他三组对应 `evaluate_body_coordinated_closure`、`evaluate_wholebody_closure`、`evaluate_acceleration_closure`。各目录须不存在，默认参数与本轮冻结协议一致。']
    Path('docs/baselines/highrod_closure_20261022.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(trials=len(rows),accepted=result['accepted_count'],integrity=result['integrity_passed'])))


if __name__=='__main__':main()
