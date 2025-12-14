from dataclasses import dataclass
from typing import Tuple, Callable, Optional

import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt


# ============================================================================
# Constants and Configuration
# ============================================================================

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


# ============================================================================
# Interpolation Module
# ============================================================================


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


# ============================================================================
#  Integration Module
# ============================================================================


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


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Основная функция исполнения."""

    # Инициализация символьной переменной
    x = sp.symbols('x')

    # ===== Часть 1: Интерполяция Лагранжа =====
    print("=" * 70)
    print("ЗАДАЧА 1: ИНТЕРПОЛЯЦИЯ МНОГОЧЛЕНОМ ЛАГРАНЖА")
    print("=" * 70)

    # Конфигурация интерполяции
    interpolation_config = InterpolationConfig(
        x_nodes=np.array([0.35, 0.41, 0.47, 0.51, 0.56, 0.64], dtype=float),
        y_nodes=np.array([2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310], dtype=float),
        x_eval=np.array([0.526, 0.453, 0.482, 0.552, 0.436], dtype=float)
    )

    # Построение полинома
    print("\n[1] Построение многочлена Лагранжа...")
    P, P_expanded = build_lagrange_polynomial(
        interpolation_config.x_nodes,
        interpolation_config.y_nodes,
        x
    )

    print(f"P(x) = {P_expanded}")

    # Вычисление значений
    print("\n[2] Вычисление значений в заданных точках...")
    P_func, y_eval = evaluate_polynomial(
        P_expanded,
        interpolation_config.x_eval,
        x
    )

    print("\nЗначения интерполяционного многочлена:")
    for xv, yv in zip(interpolation_config.x_eval, y_eval):
        print(f"  x = {xv:.3f}, P(x) = {yv:.6f}")

    # Визуализация
    print("\n[3] Построение графика интерполяции...")
    plot_interpolation(
        interpolation_config.x_nodes,
        interpolation_config.y_nodes,
        interpolation_config.x_eval,
        y_eval,
        P_func
    )

    # ===== Часть 2: Численное интегрирование =====
    print("\n" + "=" * 70)
    print("ЗАДАЧА 2: ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ (f(x) = x² ln(x))")
    print("=" * 70)

    # Конфигурация интегрирования
    integration_config = IntegrationConfig(
        a=0.35,
        b=0.64,
        epsilon=1e-6,
        max_n=400
    )

    # Определение функции
    f = x ** 2 * sp.log(x)
    f_num = sp.lambdify(x, f, 'numpy')

    print(f"\nФункция: f(x) = x² ln(x)")
    print(f"Интервал: [{integration_config.a}, {integration_config.b}]")
    print(f"Требуемая точность: ε = {integration_config.epsilon}")

    # Оценка шага
    print("\n[1] Оценка оптимального шага...")
    M = estimate_max_derivative(
        f, x,
        integration_config.a,
        integration_config.b,
        deriv_order=2
    )
    L = abs(integration_config.b - integration_config.a)
    h_raw = math.sqrt(12 * integration_config.epsilon / (M * L))

    print(f"Вторая производная: f''(x) = 2·ln(x) + 3")
    print(f"Оценка максимума: M = {M:.6f}")
    print(f"Теоретический h = {h_raw:.6f}")

    result = find_optimal_steps(
        M,
        integration_config.a,
        integration_config.b,
        integration_config.epsilon,
        integration_config.max_n
    )

    if result is None:
        print(f"\n⚠ Не найден N до {integration_config.max_n}, "
              f"удовлетворяющий точности ε = {integration_config.epsilon}")
        return

    N_opt, h_opt, err_opt = result
    N_2h = N_opt // 2

    print(f"\nОптимальные параметры:")
    print(f"  N = {N_opt} (кратно 4)")
    print(f"  h = {h_opt:.6f}")
    print(f"  Оценка погрешности: {err_opt:.3e}")

    # Вычисление интегралов
    print("\n[2] Вычисление интегралов по формуле Симпсона...")
    I_2h = simpson_rule(f_num, integration_config.a, integration_config.b, N_2h)
    I_h = simpson_rule(f_num, integration_config.a, integration_config.b, N_opt)

    print(f"\nШаг 2h (N = {N_2h}): I_2h = {I_2h:.10f}")
    print(f"Шаг h  (N = {N_opt}): I_h  = {I_h:.10f}")

    # Уточнение по Рунге
    print("\n[3] Уточнение по правилу Рунге...")
    I_refined, err_runge = apply_runge_rule(I_h, I_2h, order=4)

    print(f"Уточнённое значение: I_refined = {I_refined:.10f}")
    print(f"Оценка погрешности: {err_runge:.3e}")

    # Точное значение
    print("\n[4] Точное значение по Ньютону–Лейбницу...")
    F = x ** 3 / 3 * sp.log(x) - x ** 3 / 9
    I_exact = compute_exact_integral(F, x, integration_config.a, integration_config.b)

    print(f"Первообразная: F(x) = x³/3·ln(x) - x³/9")
    print(f"Точное значение: I_exact = {I_exact:.10f}")

    # Сравнение
    print("\n[5] Сравнение приближённых значений с точным:")
    abs_err_2h = abs(I_2h - I_exact)
    abs_err_h = abs(I_h - I_exact)
    abs_err_refined = abs(I_refined - I_exact)

    print(f"\n  |I_2h - I_exact|      = {abs_err_2h:.3e}")
    print(f"  |I_h - I_exact|       = {abs_err_h:.3e}")
    print(f"  |I_refined - I_exact| = {abs_err_refined:.3e}")

    print(f"\nОптимизация точности: {abs_err_refined / abs_err_2h:.1f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
