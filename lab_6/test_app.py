import math
import numpy as np
import sympy as sp
import pytest

from InterpolationModule import build_lagrange_polynomial, evaluate_polynomial
from InterpolationConfig import InterpolationConfig
from IntegrationModule import (
    find_optimal_steps,
    simpson_rule,
    apply_runge_rule,
    compute_exact_integral,
)
from IntegrationConfig import IntegrationConfig


# ---------- ТЕСТЫ ИНТЕРПОЛЯЦИИ ----------

def test_build_lagrange_polynomial_exact_on_nodes():
    x = sp.symbols("x")
    x_nodes = np.array([0.0, 1.0, 2.0], dtype=float)
    y_nodes = np.array([1.0, 3.0, 2.0], dtype=float)

    poly_simplified, poly_expanded = build_lagrange_polynomial(x_nodes, y_nodes, x)

    poly_func, _ = evaluate_polynomial(poly_expanded, x_nodes, x)
    y_interp = poly_func(x_nodes)

    assert np.allclose(y_interp, y_nodes, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "x_nodes,y_nodes",
    [
        (np.array([0.0], dtype=float), np.array([1.0], dtype=float)),
        (np.array([], dtype=float), np.array([], dtype=float)),
    ],
)
def test_build_lagrange_polynomial_edge_small_nodes(x_nodes, y_nodes):
    x = sp.symbols("x")
    poly_simplified, poly_expanded = build_lagrange_polynomial(x_nodes, y_nodes, x)
    poly_func, _ = evaluate_polynomial(poly_expanded, x_nodes, x)
    _ = poly_func(x_nodes)
    assert True


def test_interpolation_config_validation():
    x_nodes = np.array([0.0, 1.0], dtype=float)
    y_nodes = np.array([1.0], dtype=float)
    x_eval = np.array([0.5], dtype=float)

    with pytest.raises(ValueError):
        InterpolationConfig(x_nodes=x_nodes, y_nodes=y_nodes, x_eval=x_eval)


# ---------- ТЕСТЫ ИНТЕГРИРОВАНИЯ ----------

def test_simpson_rule_polynomial_quadratic():
    f_num = lambda x: x ** 2
    a, b = 0.0, 1.0
    exact = 1.0 / 3.0

    for n in (2, 4, 10):
        if n % 2 == 0:
            approx = simpson_rule(f_num, a, b, n)
            assert math.isclose(approx, exact, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("intervals_num", [1, 3, 5])
def test_simpson_rule_raises_on_odd_intervals(intervals_num):
    f_num = lambda x: x
    with pytest.raises(ValueError):
        simpson_rule(f_num, 0.0, 1.0, intervals_num)


def test_find_optimal_steps_with_known_derivative(monkeypatch):
    a, b = 0.0, 1.0
    epsilon = 1e-3
    max_deriv_val = 2.0

    result = find_optimal_steps(max_deriv_val, a, b, epsilon, max_n=1000)
    assert result is not None
    N, h, err_est = result
    assert N % 4 == 0
    assert err_est < epsilon
    assert math.isclose(h, (b - a) / N, rel_tol=1e-15)


def test_apply_runge_rule_and_exact_integral():
    x = sp.symbols("x")
    f = x ** 2
    f_num = sp.lambdify(x, f, "numpy")

    a, b = 0.0, 1.0
    i_2h = simpson_rule(f_num, a, b, 4)   # N=4
    i_h = simpson_rule(f_num, a, b, 8)    # N=8

    i_refined, err_runge = apply_runge_rule(i_h, i_2h, order=4)

    F = x ** 3 / 3
    i_exact = compute_exact_integral(F, x, a, b)

    assert math.isclose(i_exact, 1.0 / 3.0, rel_tol=1e-12)
    assert abs(i_refined - i_exact) <= abs(i_h - i_exact)
    assert err_runge >= 0.0


# ---------- ТЕСТЫ КОНФИГУРАЦИИ ИНТЕГРИРОВАНИЯ ----------

@pytest.mark.parametrize(
    "a,b,epsilon",
    [
        (0.0, 1.0, 1e-6),
        (0.0, 2.0, 1e-3),
    ],
)
def test_integration_config_valid(a, b, epsilon):
    cfg = IntegrationConfig(a=a, b=b, epsilon=epsilon)
    assert cfg.a == a
    assert cfg.b == b
    assert math.isclose(cfg.epsilon, epsilon)


def test_integration_config_invalid_bounds_and_epsilon():
    with pytest.raises(ValueError):
        IntegrationConfig(a=1.0, b=1.0, epsilon=1e-6)

    with pytest.raises(ValueError):
        IntegrationConfig(a=0.0, b=1.0, epsilon=0.0)
