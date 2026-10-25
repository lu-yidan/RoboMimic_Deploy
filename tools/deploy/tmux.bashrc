# Deliberately avoid ~/.bashrc's interactive ROS selection prompt.
# This file only initializes environment/aliases; it never starts robot control.
if [[ -f "$HOME/robomimic-env.sh" ]]; then
  source "$HOME/robomimic-env.sh"
else
  printf 'Missing ~/robomimic-env.sh; configure the robot environment first.\n'
fi
_robomimic_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rb() { cd "$_robomimic_repo"; }
sr() { env -u LD_LIBRARY_PATH bash "$_robomimic_repo/tools/deploy/start_tmux_layout.sh" "$@"; }
PS1='(robot tmux) \u@\h:\w\$ '
tmux_help() {
  cat <<'HELP'
六屏用途（命令只预填，按 Enter 才启动）：
  左上 Bridge     C++ 电机通信；启动后会发送电机命令。
  左中 Policy     策略/FSM/手柄；读取深度，不打开相机。
  左下 Dashboard  浏览器 http://机器人IP:8091/，含 PHP 深度图。
  右上 Camera     camera.mode 选择 gray 或 php_depth。
                  gray 网页 :8080；PHP 图像在 Dashboard :8091。
  右中 LiDAR      雷达球检测；仅 PHP 时可不启动。
  右下 Fuser      融合相机/雷达球位置；仅 PHP 时可不启动。
  monitor 窗口   只读最终球状态 DDS，不负责控制。
只看 PHP 图像：启动右上和左下，不需要左上/左中。
未启动相机：其他策略照常，禁止进入 PHP。
Ctrl-b + 方向键切面板；Ctrl-b n 切窗口；Ctrl-b d 分离（不会停止程序）。
rb 返回仓库；sr 打开六屏；tmux_help 再次查看此说明。
HELP
}
printf '\n六屏用途见边框标题；输入 tmux_help 查看完整说明。\n命令只预填，按 Enter 才启动。Ctrl-b d 仅分离，不停止程序。\n\n'
