import numpy as np
import yaml
import os


class Config:
    def __init__(self) -> None:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_yaml_path = os.path.join(current_dir, "config", "real.yaml")
        with open(mujoco_yaml_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            from common.deployment_config import load_scalebfm_config, load_php_real_config
            self.scalebfm = load_scalebfm_config()
            self.php = load_php_real_config()
            self.joint_names = config.get("joint_names", [])
            self.torch_threads = int(config.get("torch_threads", 1))
            if self.torch_threads < 1:
                raise ValueError("torch_threads must be positive")
            if self.php["enabled"] and (config["num_joints"] != 29 or abs(config["control_dt"]-.02) > 1e-9):
                raise ValueError("PHP requires 29 named joints and 50 Hz control")
            if self.scalebfm.enabled:
                if config["num_joints"] != 29 or abs(config["control_dt"] - .02) > 1e-9:
                    raise ValueError("ScaleBFM requires 29 joints and 50 Hz control")
                if not self.scalebfm.reference_forcing:
                    raise ValueError("LowState has no root localization: reference_forcing must be true")
            self.net = os.environ.get("BRIDGE_NETWORK_INTERFACE") or os.environ.get("ROBOT_IFACE") or config["net"]
            self.num_joints = config["num_joints"]
            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.control_dt = config["control_dt"]
            self.error_over_time = config["error_over_time"]
            log_cfg = config.get("logging", {})
            self.log_enabled = log_cfg.get("enabled", False)
            self.log_dir     = log_cfg.get("log_dir", "logs")
            self.log_tag     = log_cfg.get("tag", "freekick")
            self.log_states  = log_cfg.get("states", ["SKILL_FREEKICK"])
            self.bridge_domain_id = int(os.environ.get(
                "BRIDGE_DOMAIN_ID",
                config.get("bridge_domain_id", 0),
            ))
            self.bridge_state_topic = os.environ.get(
                "BRIDGE_STATE_TOPIC",
                config.get("bridge_state_topic", "rt/policy_bridge_state"),
            )
            self.bridge_cmd_topic = os.environ.get(
                "BRIDGE_CMD_TOPIC",
                config.get("bridge_cmd_topic", "rt/policy_bridge_cmd"),
            )
            self.bridge_state_stale_ms = int(os.environ.get(
                "BRIDGE_STATE_STALE_MS",
                config.get("bridge_state_stale_ms", 200),
            ))
            self.bridge_cmd_timeout_ms = int(os.environ.get(
                "BRIDGE_CMD_TIMEOUT_MS",
                config.get("bridge_cmd_timeout_ms", 100),
            ))
            self.bridge_loop_period_ms = int(os.environ.get(
                "BRIDGE_LOOP_PERIOD_MS",
                config.get("bridge_loop_period_ms", 2),
            ))
            
