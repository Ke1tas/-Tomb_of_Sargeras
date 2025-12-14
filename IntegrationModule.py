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
    M = np.max(np.abs(f_deriv_func(x_test)))

    return M


def find_optimal_steps(
        M: float,
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
        M: Оценка максимума второй производной
        a: Левая граница
        b: Правая граница
        epsilon: Требуемая точность
        max_n: Максимальное значение N

    Returns:
        Кортеж (N, h, error_estimate) или None
    """
    L = abs(b - a)

    for N in range(4, max_n + 1, 4):
        h = L / N
        err_est = M * L * h ** 2 / 12

        if err_est < epsilon:
            return N, h, err_est

    return None


def simpson_rule(
        f_num: Callable,
        a: float,
        b: float,
        N: int
) -> float:
    """
    Вычисляет определённый интеграл по композитной формуле Симпсона.

    Args:
        f_num: Числовая функция
        a: Левая граница
        b: Правая граница
        N: Число подинтервалов (должно быть чётным)

    Returns:
        Значение интеграла

    Raises:
        ValueError: Если N не чётное
    """
    if N % 2 != 0:
        raise ValueError("N должно быть чётным для формулы Симпсона")

    h = (b - a) / N
    x_vals = np.linspace(a, b, N + 1)
    y_vals = f_num(x_vals)

    # Формула Симпсона: S = h/3 * (y0 + 4*сумма_нечётных + 2*сумма_чётных + yn)
    S = h / 3 * (
            y_vals[0]
            + 4 * np.sum(y_vals[1:-1:2])
            + 2 * np.sum(y_vals[2:-1:2])
            + y_vals[-1]
    )

    return S


def apply_runge_rule(
        I_h: float,
        I_2h: float,
        order: int = 4
) -> Tuple[float, float]:
    """
    Применяет правило Рунге для уточнения интеграла.

    Для формулы порядка O(h^k):
    I_refined = I_h + (I_h - I_2h) / (2^k - 1)

    Args:
        I_h: Интеграл с шагом h
        I_2h: Интеграл с шагом 2h
        order: Порядок точности метода

    Returns:
        Кортеж (уточнённое значение, оценка погрешности)
    """
    denominator = 2 ** order - 1
    I_refined = I_h + (I_h - I_2h) / denominator
    error_estimate = abs(I_refined - I_h)

    return I_refined, error_estimate


def compute_exact_integral(
        F: sp.Expr,
        sym_x: sp.Symbol,
        a: float,
        b: float
) -> float:
    """
    Вычисляет точный определённый интеграл по формуле Ньютона–Лейбница.

    Args:
        F: Первообразная функции
        sym_x: Символическая переменная
        a: Левая граница
        b: Правая граница

    Returns:
        Значение определённого интеграла
    """
    return float(sp.N(F.subs(sym_x, b) - F.subs(sym_x, a)))