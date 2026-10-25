"""One policy-selection file for every deployment entry point."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_deployment_config():
    # An alternate file is useful for isolated simulation/validation runs.
    path = Path(os.environ.get("LYD_DEPLOY_CONFIG", ROOT / "config/deploy.yaml")).expanduser()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid deployment config: {path}")
    for section in ("scalebfm", "amp", "smp"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing deployment section: {section}")
    if type(cfg["scalebfm"].get("enabled")) is not bool:
        raise ValueError("scalebfm.enabled must be true or false")
    for section in ("amp", "smp"):
        if not isinstance(cfg[section].get("profile"), str) or not cfg[section]["profile"].strip():
            raise ValueError(f"{section}.profile must be a non-empty string")
    return cfg


def load_scalebfm_config(require_enabled=False):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(load_deployment_config()["scalebfm"])
    if require_enabled and not cfg.enabled:
        raise ValueError("This tool requires scalebfm.enabled: true in config/deploy.yaml")
    return cfg


def resolve_amp_config(cfg, profile=None):
    """Select a complete policy contract; profile entries override shared values."""
    selected = profile or load_deployment_config()["amp"]["profile"]
    profiles = cfg["model_profiles"]
    if selected not in profiles:
        raise ValueError(f"Unknown AMP profile {selected!r}; choose from {', '.join(profiles)}")
    return dict(cfg, **profiles[selected]), selected
