import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------
# 1. Исходные данные
# -----------------------------
x_nodes = np.array([0.35, 0.41, 0.47, 0.51, 0.56, 0.64], dtype=float)
y_nodes = np.array([2.73951, 2.30080, 1.96864, 1.78776, 1.59502, 1.34310], dtype=float)

x_eval = np.array([0.526, 0.453, 0.482, 0.552, 0.436], dtype=float)

# символьная переменная
x = sp.symbols('x')

# -----------------------------
# 2. Построение многочлена Лагранжа (символьно)
# -----------------------------
def lagrange_poly(xs, ys, sym_x):
    n = len(xs)
    P = 0
    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (sym_x - xs[j]) / (xs[i] - xs[j])
        P += ys[i] * Li
    return sp.simplify(P)

P = lagrange_poly(x_nodes, y_nodes, x)        # формула Лагранжа
P_expanded = sp.expand(P)                     # приведённый многочлен
print("Многочлен Лагранжа P(x) =", P_expanded)

# -----------------------------
# 3. Вычисление значений P(x) в заданных точках
# -----------------------------
P_func = sp.lambdify(x, P_expanded, 'numpy')  # числовая функция
y_eval = P_func(x_eval)

print("\nЗначения интерполяционного многочлена в заданных точках:")
for xv, yv in zip(x_eval, y_eval):
    print(f"x = {xv:.3f}, P(x) = {yv:.6f}")

# -----------------------------
# 4. График P(x), узлы и предсказанные значения
# -----------------------------
x_plot = np.linspace(0.35, 0.64, 200)
y_plot = P_func(x_plot)

plt.figure(figsize=(7, 4))
plt.plot(x_plot, y_plot, 'k-', linewidth=1.5)             # кривая P(x)
plt.plot(x_nodes, y_nodes, 'bo', markersize=6)            # узлы
plt.plot(x_eval, y_eval, 'rs', markersize=6)              # предсказанные значения
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title('Интерполяционный многочлен Лагранжа')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Оценка шага h из условия M*|b-a|*h^2/12 < eps
#    для f(x) = x^2 ln(x)
# -----------------------------
a, b = 0.35, 0.64
eps = 1e-6

f = x**2 * sp.log(x)
f2 = sp.diff(f, x, 2)     # f''(x) = 2 ln(x) + 3
print("\nВторая производная f''(x) =", f2)

f2_func = sp.lambdify(x, f2, 'numpy')

# искать максимум |f''(x)| на [a,b] по сетке
xx_test = np.linspace(a, b, 1000)
M = np.max(np.abs(f2_func(xx_test)))
print("Оценка M =", M)

# неравенство: M * |b-a| * h^2 / 12 < eps
L = abs(b - a)
h_raw = math.sqrt(12 * eps / (M * L))
print("h из неравенства (до учёта кратности 4):", h_raw)

# учитывать, что (b-a) делится на число шагов, кратное 4:
# N = (b-a)/h, N должно быть кратно 4
# пройдём по нескольким N, кратным 4, и выберем подходящее
N_candidates = []
for N in range(4, 401, 4):        # до 400 частей, шаг кратен 4
    h_candidate = (b - a) / N
    err_est = M * L * h_candidate**2 / 12
    if err_est < eps:
        N_candidates.append((N, h_candidate, err_est))

if N_candidates:
    N_opt, h_opt, err_opt = N_candidates[0]   # первый удовлетворяющий
    print(f"\nВыбран N = {N_opt} (кратно 4), шаг h = {h_opt}, оценка погрешности ≈ {err_opt}")
else:
    print("\nНе найден N до 400, удовлетворяющий заданной точности eps.")

# -----------------------------
# 6. Интеграл по формуле Симпсона с шагами 2h и h
# -----------------------------
def simpson(f_num, a, b, N):
    """
    Композитная формула Симпсона.
    N - число подинтервалов (должно быть чётным).
    """
    if N % 2 != 0:
        raise ValueError("N должно быть чётным для формулы Симпсона.")
    h = (b - a) / N
    x_vals = np.linspace(a, b, N + 1)
    y_vals = f_num(x_vals)
    S = h/3 * (y_vals[0]
               + 4 * np.sum(y_vals[1:-1:2])
               + 2 * np.sum(y_vals[2:-1:2])
               + y_vals[-1])
    return S

# функция f(x) = x^2 ln(x) как числовая
f_num = sp.lambdify(x, f, 'numpy')

# возьмём N, кратное 4 и чётное (это уже выполнено), для шага h_opt:
N_h = N_opt          # число подинтервалов для шага h
N_2h = N_h // 2      # для шага 2h

I_2h = simpson(f_num, a, b, N_2h)
I_h = simpson(f_num, a, b, N_h)

print(f"\nИнтеграл Симпсона с шагом 2h (N={N_2h}): I_2h = {I_2h:.10f}")
print(f"Интеграл Симпсона с шагом h  (N={N_h}): I_h  = {I_h:.10f}")

# -----------------------------
# 7. Уточнённое значение по правилу Рунге (Simpson, порядок 4)
# -----------------------------
# Для формулы Симпсона глобальная погрешность ~ C * h^4,
# значит уточнение по Рунге:
# I_refined = I_h + (I_h - I_2h)/(2^4 - 1) = I_h + (I_h - I_2h)/15
I_refined = I_h + (I_h - I_2h) / (2**4 - 1)
err_Runge_est = abs(I_refined - I_h)

print(f"\nУточнённое значение по Рунге: I_refined = {I_refined:.10f}")
print(f"Оценка погрешности по Рунге |I_refined - I_h| ≈ {err_Runge_est:.3e}")

# -----------------------------
# 8. Точный интеграл по Ньютону–Лейбницу
#    F(x) = x^3/3 * ln(x) - x^3/9
# -----------------------------
F = x**3/3 * sp.log(x) - x**3/9
I_exact = sp.N(F.subs(x, b) - F.subs(x, a))

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
