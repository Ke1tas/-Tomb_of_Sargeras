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
    poly = 0

    for i in range(n):
        poly_term = 1
        for j in range(n):
            if i != j:
                poly_term *= (sym_x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        poly += y_nodes[i] * poly_term

    poly_simplified = sp.simplify(poly)
    poly_expanded = sp.expand(poly_simplified)

    return poly_simplified, poly_expanded


def evaluate_polynomial(
        poly_expanded: sp.Expr,
        x_eval: np.ndarray,
        sym_x: sp.Symbol
) -> Tuple[Callable, np.ndarray]:
    """
    Вычисляет значения многочлена в точках.

    Args:
        poly_expanded: Развёрнутый символический полином
        x_eval: Точки для вычисления
        sym_x: Символическая переменная

    Returns:
        Кортеж (числовая функция, значения в точках)
    """
    poly_func = sp.lambdify(sym_x, poly_expanded, 'numpy')
    y_eval = poly_func(x_eval)
    return poly_func, y_eval
