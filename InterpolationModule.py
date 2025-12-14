import sympy as sp
import numpy as np
from typing import Tuple, Callable


def build_lagrange_polynomial(
        x_nodes: np.ndarray,
        y_nodes: np.ndarray,
        sym_x: sp.Symbol
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Строит интерполяционный многочлен Лагранжа.

    Args:
        x_nodes: Массив узлов интерполяции
        y_nodes: Массив значений в узлах
        sym_x: Символическая переменная

    Returns:
        Кортеж (базовый полином, развёрнутый полином)
    """
    n = len(x_nodes)
    P = 0

    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (sym_x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        P += y_nodes[i] * Li

    P_simplified = sp.simplify(P)
    P_expanded = sp.expand(P_simplified)

    return P_simplified, P_expanded


def evaluate_polynomial(
        P_expanded: sp.Expr,
        x_eval: np.ndarray,
        sym_x: sp.Symbol
) -> Tuple[Callable, np.ndarray]:
    """
    Вычисляет значения многочлена в точках.

    Args:
        P_expanded: Развёрнутый символический полином
        x_eval: Точки для вычисления
        sym_x: Символическая переменная

    Returns:
        Кортеж (числовая функция, значения в точках)
    """
    P_func = sp.lambdify(sym_x, P_expanded, 'numpy')
    y_eval = P_func(x_eval)
    return P_func, y_eval
