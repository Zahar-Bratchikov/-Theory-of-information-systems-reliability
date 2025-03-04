#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon, triang
from scipy.integrate import quad
from scipy.optimize import brentq

# -----------------------------
# Заданные параметры для элементов системы:
# 1. Gamma-распределение: Г(9,67)
#    - Форма: k1 = 9
#    - Масштаб: theta1 = 67
# 2. Экспоненциальное распределение: Exp(1.5e-4)
#    - λ = 1.5e-4, scale = 1/λ
# 3. Распределение Симпсона: S(23,1000)
#    - Границы: a3 = 23, b3 = 1000, используем симметричное треугольное распределение с c=0.5

# Gamma-распределение
k1, theta1 = 9, 67
dist1 = gamma(a=k1, scale=theta1)

# Экспоненциальное распределение
lmbd2 = 1.5e-4
dist2 = expon(scale=1/lmbd2)

# Симпсоновское распределение (как симметричное треугольное распределение)
a3, b3 = 23, 1000
dist3 = triang(c=0.5, loc=a3, scale=(b3 - a3))

# -----------------------------
# Функции для расчёта характеристик системы
# -----------------------------

# Плотность распределения времени до отказа системы f(t)
# Для системы, соединённой последовательно, f(t) = сумма( f_i(t) * произведение(P_j(t)) ), j ≠ i,
# где P_i(t) = вероятность безотказной работы i-го элемента = survival function.
def f(t):
    f1 = dist1.pdf(t)
    f2 = dist2.pdf(t)
    f3 = dist3.pdf(t)
    P1 = dist1.sf(t)
    P2 = dist2.sf(t)
    P3 = dist3.sf(t)
    return f1 * P2 * P3 + f2 * P1 * P3 + f3 * P1 * P2

# Вероятность безотказной работы системы P(t) = произведение(P_i(t)) для всех элементов
def P(t):
    return dist1.sf(t) * dist2.sf(t) * dist3.sf(t)

# Интенсивность отказов системы λ(t) = f(t)/P(t)
def lambda_t(t):
    P_val = P(t)
    return f(t) / P_val if P_val > 0 else 0

# Средняя наработка до отказа системы (T_mid) = ∫[0,∞] P(t) dt
def T_mid(limit=20000):
    val, _ = quad(P, 0, limit)
    return val

# Второй момент для вычисления дисперсии: E[T^2] = ∫[0,∞] 2t P(t) dt
def moment2(limit=20000):
    val, _ = quad(lambda t: 2 * t * P(t), 0, limit)
    return val

# Дисперсия системы D = E[T^2] - (T_mid)^2
def D(limit=20000):
    Tmid = T_mid(limit)
    return moment2(limit) - Tmid**2

# Среднее квадратическое отклонение системы sigma
def sigma(limit=20000):
    var = D(limit)
    return np.sqrt(var) if var > 0 else 0

# Гамма-процентная наработка до отказа системы T_gamma:
# Находим такое t, что F(t)=1-P(t)=gamma/100.
def T_gamma(gamma, t_min=0, t_max=20000):
    target = gamma / 100  # требуемая вероятность отказа
    func = lambda t: (1 - P(t)) - target
    try:
        return brentq(func, t_min, t_max)
    except ValueError:
        return np.nan

# -----------------------------
# Вывод численных характеристик в консоль
# -----------------------------
print("System Reliability Characteristics:")
print("1. Вероятность безотказной работы P(t) при t = 0: {:.6f}".format(P(0)))
print("2. Средняя наработка до отказа T_mid: {:.2f}".format(T_mid()))
print("3. Дисперсия D: {:.2f}".format(D()))
print("   Среднее квадратическое отклонение sigma: {:.2f}".format(sigma()))
example_t = 500
print("4. Интенсивность отказов λ(t) при t = {}: {:.6f}".format(example_t, lambda_t(example_t)))
print("5. Плотность распределения f(t) при t = {}: {:.6f}".format(example_t, f(example_t)))
print("6. Гамма-процентная наработка T_gamma:")
for gamma_percent in range(0, 101, 10):
    print("   γ = {:3d}%  →  T_gamma = {:.2f}".format(gamma_percent, T_gamma(gamma_percent)))

# -----------------------------
# Построение графиков в одном окне
# -----------------------------
t_values = np.linspace(0, 1500, 1000)
f_values = [f(t) for t in t_values]         # f(t)
P_values = [P(t) for t in t_values]           # P(t)
lambda_values = [lambda_t(t) for t in t_values]  # λ(t)

# Гамма-процентная наработка T_gamma для γ = 0,10,...,100
T_gamma_values = [T_gamma(g) for g in range(0, 101, 10)]

plt.figure(figsize=(12, 8))

# График 1: Плотность распределения f(t)
plt.subplot(2, 2, 1)
plt.plot(t_values, f_values, label="f(t)", color="tab:blue")
plt.xlabel("Время t")
plt.ylabel("f(t)")
plt.title("5. Плотность распределения времени до отказа f(t)")
plt.legend()
plt.grid()

# График 2: Вероятность безотказной работы P(t)
plt.subplot(2, 2, 2)
plt.plot(t_values, P_values, label="P(t)", color="tab:green")
plt.xlabel("Время t")
plt.ylabel("P(t)")
plt.title("1. Вероятность безотказной работы P(t)")
plt.legend()
plt.grid()

# График 3: Интенсивность отказов λ(t)
plt.subplot(2, 2, 3)
plt.plot(t_values, lambda_values, label="λ(t)", color="tab:red")
plt.xlabel("Время t")
plt.ylabel("λ(t)")
plt.title("4. Интенсивность отказов λ(t)")
plt.legend()
plt.grid()

# График 4: Гамма-процентная наработка T_gamma
plt.subplot(2, 2, 4)
plt.plot(range(0, 101, 10), T_gamma_values, "ko-", label="T_gamma")
plt.xlabel("γ, %")
plt.ylabel("Время t")
plt.title("6. Гамма-процентная наработка T_gamma")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()