import numpy as np
from dataclasses import dataclass


@dataclass
class InterpolationConfig:
    """Конфигурация для интерполяции Лагранжа."""
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    x_eval: np.ndarray

    def __post_init__(self):
        if len(self.x_nodes) != len(self.y_nodes):
            raise ValueError("x_nodes и y_nodes должны иметь одинаковую длину")
        if len(self.x_nodes) < 2:
            raise ValueError("Требуется минимум 2 узла интерполяции")
