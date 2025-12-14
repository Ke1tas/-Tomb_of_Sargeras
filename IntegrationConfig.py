from dataclasses import dataclass


@dataclass
class IntegrationConfig:
    """Конфигурация для численного интегрирования."""
    a: float
    b: float
    epsilon: float = 1e-6
    max_n: int = 400

    def __post_init__(self):
        if self.a >= self.b:
            raise ValueError("Левая граница должна быть меньше правой")
        if self.epsilon <= 0:
            raise ValueError("Точность должна быть положительной")
