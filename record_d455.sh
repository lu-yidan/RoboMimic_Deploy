#!/usr/bin/env bash
# Record the D455 RGB stream from the onboard V4L2 device.
#
# Examples:
#   bash record_d455.sh
#   bash record_d455.sh --device /dev/video4 --output recordings/test.mp4

set -euo pipefail

device="/dev/video4"
output="d455_rgb_640x480_30.mp4"
input_format="yuyv422"
width=640
height=480
fps=30

usage() {
    cat <<'EOF'
Usage: record_d455.sh [options]

Options:
  --device PATH       V4L2 device (default: /dev/video4)
  --output PATH       Output MP4 path (default: d455_rgb_640x480_30.mp4)
  --input-format NAME V4L2 pixel format (default: yuyv422)
  --width N           Capture width (default: 640)
  --height N          Capture height (default: 480)
  --fps N             Capture frame rate (default: 30)
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --device)       device="$2"; shift 2 ;;
        --output)       output="$2"; shift 2 ;;
        --input-format) input_format="$2"; shift 2 ;;
        --width)        width="$2"; shift 2 ;;
        --height)       height="$2"; shift 2 ;;
        --fps)          fps="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg is required but was not found in PATH" >&2
    exit 1
}
[[ -e "$device" ]] || {
    echo "V4L2 device does not exist: $device" >&2
    exit 1
}

mkdir -p "$(dirname "$output")"
echo "Recording $device at ${width}x${height}@${fps} to $output"
echo "Press Ctrl-C to stop."

exec ffmpeg -hide_banner -f v4l2 \
    -input_format "$input_format" \
    -video_size "${width}x${height}" \
    -framerate "$fps" \
    -i "$device" \
    -c:v libx264 -preset veryfast -crf 18 \
    -movflags +faststart \
    "$output"
