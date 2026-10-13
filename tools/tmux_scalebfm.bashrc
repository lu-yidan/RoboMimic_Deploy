# Deliberately avoid ~/.bashrc's interactive ROS selection prompt.
# This file only initializes environment/aliases; it never starts robot control.
if [[ -f "$HOME/robomimic-env.sh" ]]; then
  source "$HOME/robomimic-env.sh"
else
  printf 'Missing ~/robomimic-env.sh; configure the robot environment first.\n'
fi
alias rb='cd ~/RoboMimic_Deploy'
alias sr='env -u LD_LIBRARY_PATH bash ~/RoboMimic_Deploy/tools/start_tmux_layout.sh'
PS1='(robot tmux) \u@\h:\w\$ '
printf '\nScaleBFM: rb 进入项目目录。\n只读检查: python tools/check_scalebfm_onboard.py --steps 100\n控制入口（发送电机指令）: python deploy_real/deploy_real.py\n未自动启动任何控制程序。Ctrl-b d 分离，sr 重新连接。\n\n'
