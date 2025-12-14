import sympy as sp
import numpy as np
from typing import Tuple, Callable, Optional


def estimate_max_derivative(
        f: sp.Expr,
        sym_x: sp.Symbol,
        a: float,
        b: float,
        deriv_order: int = 1,
        points: int = 1000
) -> float:
    """
    Оценивает максимум абсолютного значения производной на интервале.

    Args:
        f: Символическое выражение функции
        sym_x: Символическая переменная
        a: Левая граница
        b: Правая граница
        deriv_order: Порядок производной
        points: Количество точек для сетки

    Returns:
        Максимальное значение модуля производной
    """
    f_deriv = sp.diff(f, sym_x, deriv_order)
    f_deriv_func = sp.lambdify(sym_x, f_deriv, 'numpy')

    x_test = np.linspace(a, b, points)
    max_deriv_val = np.max(np.abs(f_deriv_func(x_test)))

    return max_deriv_val


def find_optimal_steps(
        max_deriv_val: float,
        a: float,
        b: float,
        epsilon: float,
        max_n: int = 400
) -> Optional[Tuple[int, float, float]]:
    """
    Находит оптимальное число шагов N для метода Симпсона.

    Условие: M * |b-a| * h^2 / 12 < epsilon
    где h = (b-a) / N, N кратно 4

    Args:
        max_deriv_val: Оценка максимума второй производной
        a: Левая граница
        b: Правая граница
        epsilon: Требуемая точность
        max_n: Максимальное значение N

    Returns:
        Кортеж (N, h, error_estimate) или None
    """
    length = abs(b - a)

    for N in range(4, max_n + 1, 4):
        h = length / N
        err_est = max_deriv_val * length * h ** 2 / 12

        if err_est < epsilon:
            return N, h, err_est

    return None


def simpson_rule(
        f_num: Callable,
        a: float,
        b: float,
        intervals_num: int
) -> float:
    """
    Вычисляет определённый интеграл по композитной формуле Симпсона.

    Args:
        f_num: Числовая функция
        a: Левая граница
        b: Правая граница
        intervals_num: Число интервалов (должно быть чётным)

    Returns:
        Значение интеграла

    Raises:
        ValueError: Если N не чётное
    """
    if intervals_num % 2 != 0:
        raise ValueError("N должно быть чётным для формулы Симпсона")

    h = (b - a) / intervals_num
    x_vals = np.linspace(a, b, intervals_num + 1)
    y_vals = f_num(x_vals)

    square = h / 3 * (
            y_vals[0]
            + 4 * np.sum(y_vals[1:-1:2])
            + 2 * np.sum(y_vals[2:-1:2])
            + y_vals[-1]
    )

    return square


def apply_runge_rule(
        i_h: float,
        i_2h: float,
        order: int = 4
) -> Tuple[float, float]:
    """
    Применяет правило Рунге для уточнения интеграла.

    Для формулы порядка O(h^k):
    i_refined = I_h + (I_h - I_2h) / (2^k - 1)

    Args:
        i_h: Интеграл с шагом h
        i_2h: Интеграл с шагом 2h
        order: Порядок точности метода

    Returns:
        Кортеж (уточнённое значение, оценка погрешности)
    """
    denominator = 2 ** order - 1
    i_refined = i_h + (i_h - i_2h) / denominator
    error_estimate = abs(i_refined - i_h)

    return i_refined, error_estimate


def compute_exact_integral(
        f_antideriv: sp.Expr,
        sym_x: sp.Symbol,
        a: float,
        b: float
) -> float:
    """
    Вычисляет точный определённый интеграл по формуле Ньютона–Лейбница.

    Args:
        f_antideriv: Первообразная функции
        sym_x: Символическая переменная
        a: Левая граница
        b: Правая граница

    Returns:
        Значение определённого интеграла
    """
    return float(sp.N(f_antideriv.subs(sym_x, b) - f_antideriv.subs(sym_x, a)))