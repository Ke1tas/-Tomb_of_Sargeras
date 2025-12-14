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


# -----------------------------
# 1. Исходные данные
# -----------------------------

interpolation_config = InterpolationConfig(
    x_nodes=np.array([0.35, 0.41, 0.47, 0.51, 0.56, 0.64], dtype=float),
    y_nodes=np.array([2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310], dtype=float),
    x_eval=np.array([0.526, 0.453, 0.482, 0.552, 0.436], dtype=float)
)
integration_config = IntegrationConfig(
    a=0.35,
    b=0.64,
    epsilon=1e-6,
    max_n=400
)

# x_nodes = np.array([0.35, 0.41, 0.47, 0.51, 0.56, 0.64], dtype=float)
# y_nodes = np.array([2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310], dtype=float)
#
# x_eval = np.array([0.526, 0.453, 0.482, 0.552, 0.436], dtype=float)

# символьная переменная
x = sp.symbols('x')


# ============================================================================
# Interpolation Module
# ============================================================================

# -----------------------------
# 2. Построение многочлена Лагранжа (символьно)
# -----------------------------

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


poly = build_lagrange_polynomial(interpolation_config.x_nodes, interpolation_config.y_nodes, x)
print("Многочлен Лагранжа P(x) =", poly[1])


# -----------------------------
# 3. Вычисление значений P(x) в заданных точках
# -----------------------------
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


evaluate_tuple = evaluate_polynomial(poly[1], interpolation_config.x_eval, x)
print("\nЗначения интерполяционного многочлена в заданных точках:")
for xv, yv in zip(interpolation_config.x_eval, evaluate_tuple[1]):
    print(f"x = {xv:.3f}, P(x) = {yv:.6f}")


# -----------------------------
# 4. График P(x), узлы и предсказанные значения
# -----------------------------

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


plot_interpolation(interpolation_config.x_nodes, interpolation_config.y_nodes, interpolation_config.x_eval, evaluate_tuple[1], evaluate_tuple[0])


# ============================================================================
#  Integration Module
# ============================================================================

# -----------------------------
# 5. Оценка шага h из условия M*|b-a|*h^2/12 < eps
#    для f(x) = x^2 ln(x)
# -----------------------------

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


f = x ** 2 * sp.log(x)
# a, b = 0.35, 0.64
# eps = 1e-6

M = estimate_max_derivative(f, x, integration_config.a, integration_config.b, 2)
print("Оценка M =", M)

# неравенство: M * |b-a| * h^2 / 12 < eps
L = abs(integration_config.b - integration_config.a)
h_raw = math.sqrt(12 * integration_config.epsilon / (M * L))
print("h из неравенства (до учёта кратности 4):", h_raw)


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


# учитывать, что (b-a) делится на число шагов, кратное 4:
# N = (b-a)/h, N должно быть кратно 4
# пройдём по нескольким N, кратным 4, и выберем подходящее
N_candidates = find_optimal_steps(M, integration_config.a, integration_config.b, integration_config.epsilon, integration_config.max_n)

if N_candidates:
    N_opt, h_opt, err_opt = N_candidates
    print(f"\nВыбран N = {N_opt} (кратно 4), шаг h = {h_opt}, оценка погрешности ≈ {err_opt}")
else:
    print("\nНе найден N до 400, удовлетворяющий заданной точности eps.")


# -----------------------------
# 6. Интеграл по формуле Симпсона с шагами 2h и h
# -----------------------------
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


# функция f(x) = x^2 ln(x) как числовая
f_num = sp.lambdify(x, f, 'numpy')

# возьмём N, кратное 4 и чётное (это уже выполнено), для шага h_opt:
N_h = N_opt  # число подинтервалов для шага h
N_2h = N_h // 2  # для шага 2h

I_2h = simpson_rule(f_num, integration_config.a, integration_config.b, N_2h)
I_h = simpson_rule(f_num, integration_config.a, integration_config.b, N_h)

print(f"\nИнтеграл Симпсона с шагом 2h (N={N_2h}): I_2h = {I_2h:.10f}")
print(f"Интеграл Симпсона с шагом h  (N={N_h}): I_h  = {I_h:.10f}")


# -----------------------------
# 7. Уточнённое значение по правилу Рунге (Simpson, порядок 4)
# -----------------------------
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


# Для формулы Симпсона глобальная погрешность ~ C * h^4,
# значит уточнение по Рунге:
# I_refined = I_h + (I_h - I_2h)/(2^4 - 1) = I_h + (I_h - I_2h)/15
I_refined, err_Runge_est = apply_runge_rule(I_h, I_2h, 4)

print(f"\nУточнённое значение по Рунге: I_refined = {I_refined:.10f}")
print(f"Оценка погрешности по Рунге |I_refined - I_h| ≈ {err_Runge_est:.3e}")


# -----------------------------
# 8. Точный интеграл по Ньютону–Лейбницу
#    F(x) = x^3/3 * ln(x) - x^3/9
# -----------------------------
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


F = x ** 3 / 3 * sp.log(x) - x ** 3 / 9
I_exact = compute_exact_integral(F, x, integration_config.a, integration_config.b)

print(f"\nТочный интеграл по формуле Ньютона–Лейбница: I_exact = {I_exact:.10f}")

# -----------------------------
# 9. Сравнение значений
# -----------------------------
abs_err_2h = abs(I_2h - I_exact)
abs_err_h = abs(I_h - I_exact)
abs_err_refined = abs(I_refined - I_exact)

print("\nСравнение приближённых значений с точным:")
print(f"|I_2h      - I_exact| = {abs_err_2h:.3e}")
print(f"|I_h       - I_exact| = {abs_err_h:.3e}")
print(f"|I_refined - I_exact| = {abs_err_refined:.3e}")
