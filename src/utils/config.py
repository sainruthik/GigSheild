from pathlib import Path
import yaml

def load_config(path: str | None = None):
    """Load YAML config and normalize path strings."""
    cfg_path = Path(path or "config/config.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Normalize path strings
    p = cfg.get("paths", {})
    for k, v in p.items():
        p[k] = str(Path(v))
    cfg["paths"] = p
    return cfg
