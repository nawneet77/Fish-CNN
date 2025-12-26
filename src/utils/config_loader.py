"""
Configuration Loader
Loads and validates configuration from YAML files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration container with dot notation access"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default"""
        try:
            parts = key.split('.')
            value = self._config

            for part in parts:
                value = value[part]

            return value
        except (KeyError, TypeError):
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._config

    def update(self, updates: Dict[str, Any]):
        """Update configuration"""
        self._config.update(updates)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Config object
    """
    if config_path is None:
        # Use default config
        config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, output_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Config object
        output_path: Path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
