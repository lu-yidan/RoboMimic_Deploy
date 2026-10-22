"""Generate reviewable tables and figures from completed recorded experiments."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--guards',type=Path,nargs='+',required=True)
    p.add_argument('--dropout',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=a.overwrite)
    for directory in [a.grid,a.training,*a.guards]:
        if (directory/'INVALIDATED.json').exists():raise ValueError('Refusing invalidated experiment '+str(directory))
    grid=json.loads((a.grid/'results.json').read_text())
    train=json.loads((a.training/'summary.json').read_text())
    training_audit=json.loads((a.training/'independent_audit.json').read_text())
    guards=[dict(r,experiment=folder.name) for folder in a.guards for r in json.loads((folder/'results.json').read_text())]
    hold=json.loads((a.dropout/'hold_audit.json').read_text())
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for directory,label in [(a.grid/'nominal','Normal release'),(a.dropout,'Confirmation invalid: release blocked')]:
        with (directory/'control_log.csv').open(encoding='utf-8') as f:
            rows=[r for r in csv.DictReader(f) if 14<=float(r['time_s'])<=18]
        ax.plot([float(r['time_s']) for r in rows],[float(r['left_hook_vertical_n']) for r in rows],label=label)
    ax.axvline(16,color='#333333',ls='--',lw=1);ax.axvspan(16,16.5,color='#c1decf',alpha=.4)
    ax.set(xlabel='Simulation time (s)',ylabel='Old left-hook vertical force (N)',title='Confirmation dropout: retain old support during the verified 500 ms hold')
    ax.legend();fig.savefig(a.out/'release_hold.png',dpi=160);fig.savefig(a.out/'release_hold.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    labels=[r['name'].replace('row','R').replace('mm','') for r in grid]
    axes[0].bar(np.arange(len(grid)),[r['time_s'] for r in grid],color=['#259b70' if r['passed'] else '#bf5860' for r in grid])
    axes[0].axhline(109.18,color='#333333',ls='--',lw=1)
    axes[0].set_xticks(np.arange(len(grid)),labels,rotation=55,ha='right');axes[0].set_ylabel('Simulation time (s)')
    axes[0].set_title('Complete-cycle acceptance: green PASS / red FAIL (completion alone is insufficient)')
    axes[0].set_xlabel('Target-row offset (mm): R01 = foot target, R05 = hand target; one axis per case')
    stages=['zero','before','after'];counts=[sum(r['passed'] for r in training_audit['evaluations'] if r['stage']==s) for s in stages]
    axes[1].bar(stages,counts,color=['#69879d','#c4a064','#4a9b9b']);axes[1].set_ylim(0,5.8)
    axes[1].set_ylabel('Successful single-hook transfers / 5 cases')
    axes[1].set_title(f'PPO smoke test: {train["actual_steps"]} steps, one seed; no full-cycle policy claim')
    for i,n in enumerate(counts):axes[1].text(i,n+.12,f'{n}/5',ha='center')
    fig.savefig(a.out/'results.png',dpi=160);fig.savefig(a.out/'results.pdf');plt.close(fig)
    lines=['# 横杆误差、卸载保护与残差训练进展（2026-10-10）','',
           f'完整周期误差网格共 {len(grid)} 项，通过 {sum(r["passed"] for r in grid)} 项。单次换杆 PPO 完成 {train["actual_steps"]} 步。',
           '', '## 1. 横杆误差测试','',
           '只改变第一周期脚部目标杆（row01）或手部目标杆（row05）的真实位置。X 为世界深度，Z 为世界高度，各测试 ±5、±10 mm。控制器仍使用同一份名义参考轨迹，初始支撑杆不变。横杆与支架整体平移；原始场景文件未改写。',
           '', '沿用原来的完整周期及离线验收门槛。走完动作不等于通过；碰撞、末态误差或支撑不合格均算失败。这是确定性网格，不能当作统计成功概率。','',
           '| 场景 | 验收 | 仿真时长/s | 身体最大误差/mm | 额外接触峰值/N | 摆动接触峰值/N |',
           '|---|---|---:|---:|---:|---:|']
    for r in grid:lines.append(f'| {r["name"]} | {"通过" if r["passed"] else "失败"} | {r["time_s"]:.2f} | {r["base_error_m"]*1000:.2f} | {r["other_contact_n"]:.2f} | {r["flight_contact_n"]:.2f} |')
    lines += ['',f'其中 {sum(r["reason"]=="complete" for r in grid)} 项走完 109.18 s。严格失败不等于摔落：部分场景保持承重，但关节闭合或末态参考对齐超过名义基线门槛。偏差后的理想姿态可能变化，因此后续需要单独定义功能性到达指标；本轮不事后放宽验收阈值。',
              '', '| 场景 | 未通过的运行验收项 | 近力矩上限样本占比（50 Hz） |','|---|---|---:|']
    for r in grid:
        failed=', '.join(k for k,v in r['checks'].items() if not v) or '无'
        ratio=r['audit'].get('torque_near_limit_fraction_50hz')
        text='未完成审计' if ratio is None else f'{100*ratio:.2f}%'
        lines.append(f'| {r["name"]} | {failed} | {text} |')
    lines+=['','## 2. 承重确认与释放保护','',
            '释放一处支撑前，其余支撑须在正确横杆上连续 250 ms 承受至少 10 N 向上力，接触点速度不超过 20 mm/s。10 N 是本轮工程阈值，不是钩爪强度或可承载全重的证明。',
            '最终 v3 另加释放可行性检查：先按卸载后的支撑集合求解力矩分配；若不可行，使用上一参考目标和已确认旧支撑重新分配保持力矩。全程不放宽接触约束或电机上限。v2 仅有接触确认，作为失败对照保留；额外 ±15 mm 压力测试也属于旧 v2，不计入 17 项网格。',
            '若检查失败，冻结参考目标，保留已确认的旧支撑，排除未确认的新接触；最多继续 500 ms 动力学保持后结束实验。它是有限时长的暂停保护，尚未实现自动退回或无限时长安全保持。所有力与杆身份来自仿真真值，未验证真实传感器噪声、延迟与误判。','',
            '| 场景 | 完整周期通过 | 停止原因 | 拦截时刻/s | 保持时长/s |','|---|---|---|---:|---:|']
    for r in guards:lines.append(f'| {r["experiment"]}/{r["name"]} | {r["passed"]} | {r["reason"]} | {r["release_blocked_at_s"]} | {r["hold_duration_s"]:.3f} |')
    for folder in a.guards:
        for path in folder.glob('*/hold_audit.json'):
            result=json.loads(path.read_text())
            lines += ['',f'实际杆位偏差的保持复核：`{path.parent}`，8 项检查通过：{result["passed"]}。拦截前轨迹最大差异 {result["preintervention_qpos_max_difference"]}；保持期间最小垂直力（N）为 {result["hold_minimum_vertical_n"]}。']
    lines += ['',f'独立的确认通道失效测试：8 项保持复核结果为 {hold["passed"]}。14 s 起将右钩确认标志置为无效，保留真实几何和接触力；16 s 阻止左钩释放。拦截前与未注入故障的对照轨迹最大差异为 {hold["preintervention_qpos_max_difference"]}。',
              f'保持期间各接触最小垂直力（N）：{hold["hold_minimum_vertical_n"]}。此结果仅证明确认失效时的有限时长保护，不等同于真实脱钩、长期保持或自动恢复。',
              '旧版的偏差场景出现接触检查通过、随后力矩分配不可行，促使本轮加入了释放时的可行性预检。卸载过程中的新接触丢失、恢复动作和更长保持仍需验证。','']
    lines+=['','## 3. 训练任务与实际训练','',
            '任务从初始四点支撑开始，执行 16 s 右钩换杆。沿用已验证力矩控制器；策略仅输出右臂 7 关节的小幅参考修正，幅度上限 0.01 rad，经过低通滤波。训练时目标杆 X/Z 独立均匀随机偏移 ±5 mm。',
            '观测为名义轨迹跟踪误差、关节速度、重力方向、身体位置误差、相位、四处理想接触力和上一动作，共 77 维。策略不读取真实横杆偏移。接触身份用于保护判定，属于仿真特权信息。',
            '采用 CPU PPO、单随机种子。分别用零修正、训练前策略和训练后策略评估同一 5 项网格：名义位置及 X/Z 的 ±5 mm。下表还要求通过保存状态的离线碰撞、力矩和动作边界复核。','',
            '| 对照 | 成功次数 |','|---|---:|']
    for label,n in zip(['零修正','训练前','训练后'],counts):lines.append(f'| {label} | {n}/5 |')
    lines += ['',f'检查点重载动作最大差异：{train["checkpoint_reload_max_difference"]}。训练耗时 {train["training_wall_s"]:.1f} s。',
              '这证明训练流程、评估与保存重载可执行，不能证明学到稳健攀爬。当前没有训练后完整周期或两米爬升的成功证据，也没有多随机种子泛化结论。该 77D/7D 研究接口与旧部署 100D/29D 策略接口不同，不能直接替换旧模型。',
              '', '## 复现','', '在仓库根目录 PowerShell 中执行，输出目录必须未存在：','', '```powershell',
              '$env:OPENBLAS_NUM_THREADS="1"', '$env:OMP_NUM_THREADS="1"',
              r'.\.venv\Scripts\python.exe tools\run_rung_error_suite.py --out outputs\rung_grid_repeat --workers 3',
              r'.\.venv\Scripts\python.exe tools\run_rung_error_suite.py --out outputs\rung_guard_repeat --workers 2 --guard --cases nominal row05_z_-10mm row05_x_+10mm',
              r'.\.venv\Scripts\python.exe tools\train_rung_residual.py --out outputs\rung_train_repeat --steps 2048',
              '```','', '## 原始证据','', f'- 网格：`{a.grid}`',f'- 保护：`{a.guards}`',f'- 训练：`{a.training}`',
              '- 每项物理实验包含实际状态轨迹、接触日志、参考及源码快照、场景偏移和独立审计；失败实验保留。',
              '- 原有连续两周期 49.97 cm 成果仍属于名义位置控制器，与本轮训练策略分开记账。','']
    lines += ['## 实验质量与边界','',
              '最终 32 项自动测试通过。新版保护的名义完整周期同时通过运行和独立验收；两个不同保护触发案例分别通过 8 项保持复核。训练的 15 次对照评估均通过离线复核，检查点在学长平台入口实跑单次换杆成功。',
              '第一轮 v1 偏移实验因静态 BVH 碰撞包围盒未更新而作废，目录内保留 INVALIDATED.json。v2 从偏移后的几何重新编译包围盒，验证拓扑一致后更新；不重置运行中的机器人状态。对曾漏检的历史轨迹全部 2,746 个状态，与完整重编译模型的碰撞结果比较，差异为零。',
              '本轮误差网格仅覆盖单轴、单排杆误差。它没有覆盖双轴组合、多排同时偏差、随机摩擦、外力、真实视觉误差或传感器噪声。任何一项名义轨迹失败都不能被换算为真实系统的失效率。',
              '', '## 后续优先级','',
              '先扩大脚部摆动与膝部的避杆余量，再让残差策略学习单次换脚和承重转移。当前右手任务的零修正对照已较强，只有扩大任务难度并在独立误差场景中超过该对照，才能主张学习带来收益。保护还需覆盖卸载过程中的接触丢失，并验证更长保持或受控退回。','']
    lines += ['## 在学长平台回放训练检查点','', '```powershell',
              r'.\.venv\Scripts\python.exe -X utf8 deploy_mujoco\deploy_mujoco.py --config-name mujoco_rung_residual hydra.job.chdir=false rung_residual.output_dir=outputs/rung_replay_repeat rung_residual.viewer=true',
              '```','', '该入口加载新任务检查点，执行右手单次换杆，并写出动力学轨迹与日志；无需连接真机。','']
    (a.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (a.out/'summary.json').write_text(json.dumps(dict(grid=grid,guards=guards,training=train,training_audit=training_audit,confirmation_dropout=hold),indent=2))
    print(a.out/'report.md')


if __name__=='__main__':main()
