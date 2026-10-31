"""Publish a bounded report of completed known-case calibration evidence."""
import argparse,json
from datetime import datetime
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    suites={};evidence={};geometry=[]
    for name in ('original8','grid12','old16'):
        directory=Path('outputs')/f'boundary_combined_{name}_20261027_v1'
        run=load_run(directory)
        comparison=Path('outputs')/f'boundary_combined_compare_{name}_20261027_v1/comparison.json'
        c=json.loads(comparison.read_text())
        assert c['after']['audit_sha256']==run['audit_sha256'] and not c['lost']
        assert c['after_successes']==len(run['rows'])
        m=json.loads((directory/'manifest.json').read_text())
        for f in ('independent_candidate_audit.json','composite_prefix_audit_v2.json','controller_modifier_audit_v2.json','adaptive_reference_audit.json'):
            assert json.loads((directory/f).read_text())['passed'];evidence[str(directory/f)]=sha(directory/f)
        for f in ('manifest.json','summary.json','regression.json'):evidence[str(directory/f)]=sha(directory/f)
        evidence[str(comparison)]=sha(comparison)
        suites[name]=dict(directory=str(directory.resolve()),before=c['before_successes'],after=c['after_successes'],cases=len(run['rows']),gained=c['gained'],lost=c['lost'],rows=c['rows'])
        geometry.extend(json.dumps(x,sort_keys=True) for x in run['cases'].values())
    assert len(geometry)==len(set(geometry))==36
    physics=Path('outputs/boundary_combined_physics_comparison_20261027_v1/summary.json')
    dynamics=json.loads(physics.read_text());assert all(dynamics['prefix_exact'].values());evidence[str(physics)]=sha(physics)
    middle=Path('outputs/boundary_action_bias_old16_20261027_v1/regression.json');mid=json.loads(middle.read_text());assert mid['successes']==15 and not mid['lost'];evidence[str(middle)]=sha(middle)
    reference=Path('outputs/boundary_reference_search_20261027_v1/stopped_report.json');evidence[str(reference)]=sha(reference)
    authority=Path('outputs/boundary_action_authority_20261027_v1/summary.json');evidence[str(authority)]=sha(authority)
    selected=[next(r for r in suites['original8']['rows'] if r['case']=='foot_z_plus15')]
    selected.extend(r for r in suites['old16']['rows'] if r['case'] in ('boundary_05','boundary_06','boundary_07'))
    before_acc=dynamics['rows']['original']['final_root_acceleration_rms_m_s2'];after_acc=dynamics['rows']['modified']['final_root_acceleration_rms_m_s2']
    result=dict(checked_at=datetime.now().astimezone().isoformat(),known_suite_complete=True,new_training_steps=0,
        suites=suites,selected_endpoints=selected,intermediate=mid,dynamics=dynamics,
        acceleration_rms_change_percent=100*(after_acc/before_acc-1),calibration=m['closure_action_bias'],reference_command=m['adaptive_reference_command'],
        checkpoint_sha256=run['checkpoint_sha256'],unique_known_cases=36,before_successes=33,after_successes=36,
        minimum_known_scaled_margin=min(r['after_scaled_margin'] for s in suites.values() for r in s['rows']),
        evidence_sha256=evidence,source_sha256=sha(__file__),automatic_promotion=False,
        scope='Completed 113.66-second single-cycle known-case evaluation. Fixed calibration of existing PPO outputs and reference commands; no new PPO weights, camera validation, hardware experiment, or multicycle claim. Neighborhood and continuous-transfer experiments are separate and are not counted here.')
    a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    lines=['G1 攀爬：三个失败工况修复的单周期阶段报告',f"核验时间：{result['checked_at']}",'',
        '结论：原 36 个不同已知工况从 33/36 提升到 36/36；没有新增失败。',
        '原八工况 8/8，十二项网格 12/12，十六开发工况由 13/16 到 16/16。',
        '这次是固定膝踝动作与原五维参考的联合校准，PPO 权重未变，新增训练步数为 0。',
        '关节 40 mrad、根部抬升 240–260 mm、原生位置/力矩/接触限值全部保持。','',
        '工况                         关节误差 mrad（前→后）       最大根部抬升 mm（前→后）']
    for r in selected:lines.append(f"{r['case']:<27} {r['before_closure_mrad']:.6f} → {r['after_closure_mrad']:.6f}       {r['before_rise_max_mm']:.6f} → {r['after_rise_max_mm']:.6f}")
    lines.extend(['','修复依据：三个失败点的左踝输出在末两秒全部达到 +1；单独调高度会与左踝闭合误差冲突。',
        '先在原动作范围内校准左膝/左踝，已审计 15/16；再联合参考量，达到已审计 16/16。',
        '最终膝踝归一化偏置均为 -0.5，自 70.2 秒开始以 3 秒平滑进入，保留原几何权重与最终 [-1,1] 截断。',
        f"最终五维参考命令：{m['adaptive_reference_command']}",
        '三套完整轨迹均通过独立动作/状态检查、冻结前段、固定控制修正和参考命令四类审计。',
        '最初四条搜索完整轨迹按参数及哈希核对后复用，其余 32 个工况重新从零运行。','',
        '最高杆连续遥测：227,320 个 2000 Hz 样本，轨迹复现及采样对齐审计通过。',
        f'末两秒根部加速度分量 RMS：{before_acc:.10f} → {after_acc:.10f} m/s²（增加 {result["acceleration_rms_change_percent"]:.2f}%）。',
        '四组末段最小竖直支撑力均为正，但不能因此声称所有动力学指标都更好。','',
        '局限：boundary_07 关节裕量仅约 0.032 mrad，高度裕量约 0.019 mm。已知点通过不等于稳健泛化。',
        '邻近扰动与连续多周期验证单独记录，本报告不预先计入其结果；不宣称当前版本已连续攀升两米。',
        '没有替换默认入口、连接真机或自动推送 GitHub。旧入口不包含新的动作适配器。','',
        '通过专用入口打开已验证单周期（在仓库根目录执行，输出目录需未存在）：',
        r'.\.venv\Scripts\python.exe -m deploy_mujoco.run_action_bias_full_cycle_v2 --experiment outputs/boundary_combined_original8_20261027_v1 --case foot_z_plus15 --output outputs/boundary_combined_viewer_v1 --viewer'])
    (a.output/'report.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');x=np.arange(len(selected))
    for ax,key,limit,title in [(axes[0],'closure_mrad',40.,'Joint closure error (mrad)'),(axes[1],'rise_max_mm',260.,'Maximum final root rise (mm)')]:
        for label,color,offset in [('before','#b75242',-.1),('after','#24758b',.1)]:
            ax.plot(x+offset,[r[label+'_'+key] for r in selected],'o',color=color,label='PPO parent' if label=='before' else 'Fixed joint/reference calibration')
        for i,r in enumerate(selected):ax.plot([i-.1,i+.1],[r['before_'+key],r['after_'+key]],color='#999999',lw=1)
        ax.axhline(limit,color='#a33',ls='--',lw=1);ax.set_xticks(x,['Highest +15 mm','B05','B06','B07']);ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Known cases: 33/36 to 36/36, unchanged acceptance and PPO weights')
    fig.savefig(a.output/'endpoints.png',dpi=180);fig.savefig(a.output/'endpoints.pdf');plt.close(fig)
    print(json.dumps(dict(output=str(a.output.resolve()),before=33,after=36,new_training_steps=0,minimum_margin=result['minimum_known_scaled_margin'])),flush=True)


if __name__=='__main__':main()
