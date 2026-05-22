"""Shared defaults for camera perception services."""

DEFAULT_BOARD_ARGS = (
    "--tag-id", "0",
    "--tag-id", "1",
    "--tag-id", "2",
    "--tag-id", "3",
    "--tag-offset", "0", "0.10", "0.175", "0.00",
    "--tag-offset", "1", "0.10", "-0.175", "0.00",
    "--tag-offset", "2", "-0.10", "-0.175", "0.00",
    "--tag-offset", "3", "-0.10", "0.175", "0.00",
)

GRAY_V4L2_ARGS = (
    "--camera-profile", "gray-ir",
    "--color-backend", "v4l2",
    "--v4l2-device", "/dev/video3",
    "--v4l2-fourcc", "GREY",
    "--v4l2-fps", "30",
)

BRIGHT_BALL_ARGS = (
    "--ball-bright",
    "--ball-bright-max-hz", "0",
    "--ball-bright-max-abs-y", "5.0",
)
