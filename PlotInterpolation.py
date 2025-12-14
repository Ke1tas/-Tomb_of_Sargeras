import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def plot_interpolation(
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        P_func: Callable,
        figsize: Tuple[int, int] = (7, 4),
        show: bool = True
) -> None:
    """
    Визуализирует результаты интерполяции.

    Args:
        x_nodes: Узлы интерполяции
        y_nodes: Значения в узлах
        x_eval: Точки оценки
        y_eval: Значения в точках оценки
        P_func: Числовая функция полинома
        figsize: Размер фигуры
        show: Показать ли график
    """
    x_plot = np.linspace(x_nodes.min(), x_nodes.max(), 200)
    y_plot = P_func(x_plot)

    plt.figure(figsize=figsize)
    plt.plot(x_plot, y_plot, 'k-', linewidth=1.5, label='P(x)')
    plt.plot(x_nodes, y_nodes, 'bo', markersize=6, label='Узлы')
    plt.plot(x_eval, y_eval, 'rs', markersize=6, label='Оценённые значения')
    plt.xlabel('x', fontsize=11)
    plt.ylabel('y', fontsize=11)
    plt.title('Интерполяционный многочлен Лагранжа', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()
