# 工具目录

所有命令从仓库根目录执行。**唯一 tmux 入口：**

```bash
bash tools/deploy/start_tmux_layout.sh robomimic
```

保留原六屏；只预填、不自动按 Enter。PHP 由部署 YAML 开启，详见 [本体部署和 Dashboard](../docs/php_real_bridge_zh.md)。旧 `start_tmux_php.sh` 和 `start_tmux_scalebfm.sh` 已删除；旧 `tools/start_tmux_layout.sh` 已移到 `tools/deploy/`。更新个人 alias、定时任务和外部脚本中的旧路径；仓库内部引用已更新。不保留重复包装入口。

| 目录 | 用途 | 常用入口 |
| --- | --- | --- |
| `deploy/` | 六屏启动与终端环境 | `start_tmux_layout.sh`、`tmux.bashrc` |
| `diagnostics/` | 状态、连接、输入诊断 | `check_php_real.py`（离线）、`check_ball_state.py`（DDS）、`joystick_test.py` |
| `capture/` | 实际相机采集 | `run_camera.py`（按配置常驻）、`preview_php_depth.py`（独立检查）、`record_d455_rgb.py` |
| `validation/` | 策略与仿真验证 | `smoke_php_mujoco.py`、`validate_smp_sim.py`、`test_scalebfm_real.py` |
| `evaluation/` | 评估、benchmark、批量比较 | `evaluate_amp_deployment.py`、`evaluate_scalebfm_motion.py` |
| `motion/` | 模型准备、动作转换和流式实验 | `setup_scalebfm.py`、`prepare_duobox_scalebfm.py`、`reactive_realtime.py` |
| `analysis/` | 日志审计、回放与汇总 | `audit_recovery_log.py`、`playback_log.py` |
| `export/` | 模型导出、视频及 trace 渲染 | `export_amp_guidance.py`、`export_video.py` |

目录分类不代表所有工具都是离线的。`capture/` 会访问真实相机，部分诊断工具会建立 DDS 订阅；部署启动后的 C++ bridge 会控制电机。每个工具的参数见文件说明或 `--help`。

PHP 离线检查：

```bash
python tools/diagnostics/check_php_real.py --out /tmp/php-real-offline.json
python -m unittest discover -s tests -p 'test_php_*.py'
```

新建 tmux 会话的每个面板顶部显示位置和用途；在空闲 shell 输入 `tmux_help` 可查看六屏职责、网页端口与操作提示。已有会话不会被脚本重建，新布局请使用一个新的会话名。
