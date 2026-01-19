import os
from typing import Optional, Dict, Any
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


def load_config(config_dir: str = "config",
             environment: Optional[str] = None) -> DictConfig:
    """
    Загрузка конфигурации

    Returns:
        Объект конфигурации OmegaConf
    """

    config_dir = Path(config_dir)
    base_config = OmegaConf.load(config_dir / "base.yaml")
    if environment is None:
        environment = base_config.environment

    env_config_path = config_dir / f"{environment}.yaml"

    if not env_config_path.exists():
        raise FileNotFoundError(
            f"Конфигурационный файл не найден: {env_config_path}"
        )

    env_config = OmegaConf.load(env_config_path)
    config = OmegaConf.merge(base_config, env_config)

    print(f"Конфигурация загружена: {environment}")
    return config