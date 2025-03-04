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
# 3. Распределение Симпсона: S(23, 1000)
#    - Границы: a3 = 23, b3 = 1000 (симметричное треугольное распределение, c=0.5)
# -----------------------------

# Gamma-распределение
k1, theta1 = 9, 67
dist1 = gamma(a=k1, scale=theta1)

# Экспоненциальное распределение
lmbd2 = 1.5e-4
dist2 = expon(scale=1/lmbd2)

# Симпсоновское распределение (как симметричное треугольное распределение)
a3, b3 = 23, 1000
dist3 = triang(c=0.5, loc=a3, scale=(b3 - a3))

# Функция плотности распределения времени до отказа системы f(t)
def f(t):
    f1 = dist1.pdf(t)
    f2 = dist2.pdf(t)
    f3 = dist3.pdf(t)
    P1 = dist1.sf(t)
    P2 = dist2.sf(t)
    P3 = dist3.sf(t)
    return f1 * P2 * P3 + f2 * P1 * P3 + f3 * P1 * P2

# Вероятность безотказной работы системы P(t)
def P(t):
    return dist1.sf(t) * dist2.sf(t) * dist3.sf(t)

# Интенсивность отказов системы λ(t) = f(t)/P(t)
def lambda_t(t):
    P_val = P(t)
    return f(t) / P_val if P_val > 0 else 0

# Гамма-процентная наработка системы T_gamma:
# Ищем такое t, что 1 - P(t) = γ/100
def T_gamma(gamma, t_min=0, t_max=20000):
    target = gamma / 100
    func = lambda t: (1 - P(t)) - target
    try:
        return brentq(func, t_min, t_max)
    except ValueError:
        return np.nan

# Средняя наработка до отказа системы (T_mid) = ∫₀∞ P(t) dt
def T_mid(limit=20000):
    val, _ = quad(P, 0, limit)
    return val

# Второй момент для вычисления дисперсии
def moment2(limit=20000):
    val, _ = quad(lambda t: 2 * t * P(t), 0, limit)
    return val

# Дисперсия системы D = E[T^2] - (T_mid)^2
def D(limit=20000):
    Tmid = T_mid(limit)
    return moment2(limit) - Tmid**2

# Среднее квадратическое отклонение sigma
def sigma(limit=20000):
    var = D(limit)
    return np.sqrt(var) if var > 0 else 0

# Функция для построения графиков:
# Строим графики для:
#   - Плотности распределения f(t),
#   - Вероятности безотказной работы P(t),
#   - Гамма-процентной наработки T_gamma,
#   - Интенсивности отказов λ(t)
def plot_graphs():
    t_values = np.linspace(0, 1500, 1000)
    f_values = [f(t) for t in t_values]
    P_values = [P(t) for t in t_values]
    lambda_values = [lambda_t(t) for t in t_values]
    gamma_labels = list(range(0, 101, 10))
    T_gamma_values = [T_gamma(g) for g in gamma_labels]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: Плотность распределения f(t)
    axs[0, 0].plot(t_values, f_values, label="f(t)", color="tab:blue")
    axs[0, 0].set_xlabel("Время t")
    axs[0, 0].set_ylabel("f(t)")
    axs[0, 0].set_title("Плотность распределения f(t)")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # График 2: Вероятность безотказной работы P(t)
    axs[0, 1].plot(t_values, P_values, label="P(t)", color="tab:green")
    axs[0, 1].set_xlabel("Время t")
    axs[0, 1].set_ylabel("P(t)")
    axs[0, 1].set_title("Вероятность безотказной работы P(t)")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # График 3: Гамма-процентная наработка T_gamma
    axs[1, 0].plot(gamma_labels, T_gamma_values, "ko-", label="T_gamma")
    axs[1, 0].set_xlabel("γ, %")
    axs[1, 0].set_ylabel("Время T_gamma")
    axs[1, 0].set_title("Гамма-процентная наработка T_gamma")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # График 4: Интенсивность отказов λ(t)
    axs[1, 1].plot(t_values, lambda_values, label="λ(t)", color="tab:red")
    axs[1, 1].set_xlabel("Время t")
    axs[1, 1].set_ylabel("λ(t)")
    axs[1, 1].set_title("Интенсивность отказов λ(t)")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()
    plt.show()

# Вывод в консоль численных характеристик для параметров, для которых не строятся графики
print("System Reliability Numerical Characteristics:")
print("Средняя наработка до отказа T_mid: {:.2f}".format(T_mid()))
print("Дисперсия D: {:.2f}".format(D()))
print("Среднее квадратическое отклонение sigma: {:.2f}".format(sigma()))
# Для f(t), P(t), T_gamma и λ(t) строятся графики – консольный вывод не производится.

# Построение графиков для f(t), P(t), T_gamma и λ(t)
plot_graphs()