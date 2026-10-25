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
printf '\nC++ bridge 六屏部署：左上为电机通信，左中为 Python 策略。\n命令仅预填，按 Enter 启动；rb 进入项目目录，Ctrl-b d 分离。\n离线检查：python tools/check_scalebfm_onboard.py --steps 100\n\n'
