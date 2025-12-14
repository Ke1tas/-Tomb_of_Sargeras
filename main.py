import math
import numpy as np
import sympy as sp

from IntegrationConfig import IntegrationConfig
from IntegrationModule import estimate_max_derivative, find_optimal_steps, \
                              simpson_rule, apply_runge_rule, compute_exact_integral
from InterpolationConfig import InterpolationConfig
from InterpolationModule import build_lagrange_polynomial, evaluate_polynomial
from PlotInterpolation import plot_interpolation


def main():
    """Основная функция исполнения."""

    x = sp.symbols('x')

    print("=" * 70)
    print("ЗАДАЧА 1: ИНТЕРПОЛЯЦИЯ МНОГОЧЛЕНОМ ЛАГРАНЖА")
    print("=" * 70)

    interpolation_config = InterpolationConfig(
        x_nodes=np.array([0.35, 0.41, 0.47, 0.51, 0.56, 0.64], dtype=float),
        y_nodes=np.array([2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310], dtype=float),
        x_eval=np.array([0.526, 0.453, 0.482, 0.552, 0.436], dtype=float)
    )

    print("\n[1] Построение многочлена Лагранжа...")
    poly_simplified, poly_expanded = build_lagrange_polynomial(
        interpolation_config.x_nodes,
        interpolation_config.y_nodes,
        x
    )

    print(f"P(x) = {poly_expanded}")

    print("\n[2] Вычисление значений в заданных точках...")
    poly_func, y_eval = evaluate_polynomial(
        poly_expanded,
        interpolation_config.x_eval,
        x
    )

    print("\nЗначения интерполяционного многочлена:")
    for xv, yv in zip(interpolation_config.x_eval, y_eval):
        print(f"  x = {xv:.3f}, P(x) = {yv:.6f}")

    print("\n[3] Построение графика интерполяции...")
    plot_interpolation(
        interpolation_config.x_nodes,
        interpolation_config.y_nodes,
        interpolation_config.x_eval,
        y_eval,
        poly_func
    )

    print("\n" + "=" * 70)
    print("ЗАДАЧА 2: ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ (f(x) = x² ln(x))")
    print("=" * 70)

    integration_config = IntegrationConfig(
        a=0.35,
        b=0.64,
        epsilon=1e-6,
        max_n=400
    )

    f = x ** 2 * sp.log(x)
    f_num = sp.lambdify(x, f, 'numpy')

    print(f"\nФункция: f(x) = x² ln(x)")
    print(f"Интервал: [{integration_config.a}, {integration_config.b}]")
    print(f"Требуемая точность: ε = {integration_config.epsilon}")

    print("\n[1] Оценка оптимального шага...")
    max_deriv_val = estimate_max_derivative(
        f, x,
        integration_config.a,
        integration_config.b,
        deriv_order=2
    )
    length = abs(integration_config.b - integration_config.a)
    h_raw = math.sqrt(12 * integration_config.epsilon / (max_deriv_val * length))

    print(f"Вторая производная: f''(x) = 2·ln(x) + 3")
    print(f"Оценка максимума: M = {max_deriv_val:.6f}")
    print(f"Теоретический h = {h_raw:.6f}")

    result = find_optimal_steps(
        max_deriv_val,
        integration_config.a,
        integration_config.b,
        integration_config.epsilon,
        integration_config.max_n
    )

    if result is None:
        print(f"\n⚠ Не найден N до {integration_config.max_n}, "
              f"удовлетворяющий точности ε = {integration_config.epsilon}")
        return

    n_opt, h_opt, err_opt = result
    n_2h = n_opt // 2

    print(f"\nОптимальные параметры:")
    print(f"  N = {n_opt} (кратно 4)")
    print(f"  h = {h_opt:.6f}")
    print(f"  Оценка погрешности: {err_opt:.3e}")

    print("\n[2] Вычисление интегралов по формуле Симпсона...")
    i_2h = simpson_rule(f_num, integration_config.a, integration_config.b, n_2h)
    i_h = simpson_rule(f_num, integration_config.a, integration_config.b, n_opt)

    print(f"\nШаг 2h (N = {n_2h}): I_2h = {i_2h:.10f}")
    print(f"Шаг h  (N = {n_opt}): I_h  = {i_h:.10f}")

    print("\n[3] Уточнение по правилу Рунге...")
    i_refined, err_runge = apply_runge_rule(i_h, i_2h, order=4)

    print(f"Уточнённое значение: I_refined = {i_refined:.10f}")
    print(f"Оценка погрешности: {err_runge:.3e}")

    print("\n[4] Точное значение по Ньютону–Лейбницу...")
    f_antideriv = x ** 3 / 3 * sp.log(x) - x ** 3 / 9
    i_exact = compute_exact_integral(f_antideriv, x, integration_config.a, integration_config.b)

    print(f"Первообразная: F(x) = x³/3·ln(x) - x³/9")
    print(f"Точное значение: I_exact = {i_exact:.10f}")

    print("\n[5] Сравнение приближённых значений с точным:")
    abs_err_2h = abs(i_2h - i_exact)
    abs_err_h = abs(i_h - i_exact)
    abs_err_refined = abs(i_refined - i_exact)

    print(f"\n  |I_2h - I_exact|      = {abs_err_2h:.3e}")
    print(f"  |I_h - I_exact|       = {abs_err_h:.3e}")
    print(f"  |I_refined - I_exact| = {abs_err_refined:.3e}")

    print(f"\nОптимизация точности: {abs_err_refined / abs_err_2h:.1f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
