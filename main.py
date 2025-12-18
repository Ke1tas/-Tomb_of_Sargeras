import logging
from logging_config import setup_logging
import math
import numpy as np
import sympy as sp

from IntegrationConfig import IntegrationConfig
from IntegrationModule import estimate_max_derivative, find_optimal_steps, \
    simpson_rule, apply_runge_rule, compute_exact_integral
from InterpolationConfig import InterpolationConfig
from InterpolationModule import build_lagrange_polynomial, evaluate_polynomial
from PlotInterpolation import plot_interpolation


logger = logging.getLogger(__name__)

def main():
    setup_logging()
    logger.info("Запуск программы")

    x = sp.symbols('x')
    logger.debug("Создан символ x для интерполяции и интегрирования")


    logger.info("Начало блока интерполяции")
    interpolation_config = InterpolationConfig(
        x_nodes=np.array([0.35, 0.41, 0.47, 0.51, 0.56, 0.64], dtype=float),
        y_nodes=np.array([2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310], dtype=float),
        x_eval=np.array([0.526, 0.453, 0.482, 0.552, 0.436], dtype=float),
    )
    logger.debug("Создан InterpolationConfig: %s", interpolation_config)

    poly_simplified, poly_expanded = build_lagrange_polynomial(
        interpolation_config.x_nodes,
        interpolation_config.y_nodes,
        x,
    )
    logger.info("Построен полином Лагранжа степени %d", len(interpolation_config.x_nodes) - 1)

    poly_func, y_eval = evaluate_polynomial(poly_expanded, interpolation_config.x_eval, x)
    logger.debug("Выполнена оценка полинома в %d точках", len(interpolation_config.x_eval))

    for xv, yv in zip(interpolation_config.x_eval, y_eval):
        logger.debug(f" Оценка полинома в точке x = {xv:.3f}: P(x) = {yv:.6f}")

    plot_interpolation(
        interpolation_config.x_nodes,
        interpolation_config.y_nodes,
        interpolation_config.x_eval,
        y_eval,
        poly_func,
    )
    logger.info("График интерполяции построен")


    logger.info("Начало блока интегрирования")
    f = x**2 * sp.log(x)
    integration_config = IntegrationConfig(a=0.35, b=0.64)
    logger.debug("Создан IntegrationConfig: %s", integration_config)

    f_num = sp.lambdify(x, f, "numpy")
    logger.debug("Сформирована численная функция f_num")

    max_deriv_val = estimate_max_derivative(f, x, integration_config.a, integration_config.b, deriv_order=2)
    logger.info("Оценен максимум второй производной: M = %.6f", max_deriv_val)

    length = abs(integration_config.b - integration_config.a)
    h_raw = math.sqrt(12 * integration_config.epsilon / (max_deriv_val * length))
    logger.info("Найден теоретический шаг интегрирования: h = %.6f", h_raw)

    result = find_optimal_steps(
        max_deriv_val,
        integration_config.a,
        integration_config.b,
        integration_config.epsilon,
        integration_config.max_n,
    )

    if result is None:
        logger.warning(
            "Не удалось подобрать шаг: достигнут maxN=%d при epsilon=%g",
            integration_config.max_n,
            integration_config.epsilon,
        )
        return

    n_opt, h_opt, err_opt = result
    logger.info("Подобран оптимальный шаг h=%.6f, N=%d, оценка ошибки=%.3e", h_opt, n_opt, err_opt)

    i_2h = simpson_rule(f_num, integration_config.a, integration_config.b, n_opt * 2)
    i_h = simpson_rule(f_num, integration_config.a, integration_config.b, n_opt)
    logger.debug("Интеграл по правилу Симпсона: I(h)=%.10f, I(2h)=%.10f", i_h, i_2h)

    i_refined, err_runge = apply_runge_rule(i_h, i_2h, order=4)
    logger.info("Уточнённое значение интеграла по правилу Рунге: I_refined=%.10f, err=%.3e", i_refined, err_runge)

    f_antideriv = x ** 3 / 3 * sp.log(x) - x ** 3 / 9
    i_exact = compute_exact_integral(f_antideriv, x, integration_config.a, integration_config.b)
    logger.info("Точное значение интеграла по Ньютону–Лейбницу: I_exact=%.10f", i_exact)

    abs_err_2h = abs(i_2h - i_exact)
    abs_err_h = abs(i_h - i_exact)
    abs_err_refined = abs(i_refined - i_exact)

    logger.info(f"Ошибка интеграла с шагом 2h по Симпсону:  |I_2h - I_exact|      = {abs_err_2h:.3e}")
    logger.info(f"Ошибка интеграла с шагом h по Симпсону:   |I_h - I_exact|       = {abs_err_h:.3e}")
    logger.info(f"Ошибка уточненного значения интеграла:    |I_refined - I_exact| = {abs_err_refined:.3e}")

    print(f"\nОптимизация точности: {abs_err_refined / abs_err_2h:.1f}x")

    logger.info("Работа программы завершена успешно")



if __name__ == "__main__":
    main()
