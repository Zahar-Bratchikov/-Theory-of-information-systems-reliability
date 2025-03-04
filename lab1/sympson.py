#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Реализация численного интегрирования методом Симпсона
def simpson_integrate(func, a, b, n=1000):
    """
    Вычисляет определённый интеграл функции func на отрезке [a, b] методом Симпсона.
    Параметры:
      func — функция для интегрирования, принимающая один аргумент t;
      a, b — пределы интегрирования;
      n — количество разбиений (должно быть чётным).
    """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        t = a + i * h
        if i % 2 == 0:
            s += 2 * func(t)
        else:
            s += 4 * func(t)
    return s * h / 3

# Функция плотности распределения времени до отказа по закону распределения Симпсона, обозначаемая как f(t)
def f(t, a, b):
    if a <= t <= b:
        return (2 / (b - a)) - (2 / (b - a)**2) * abs(a + b - 2*t)
    else:
        return 0

# Вычисление момента порядка k: m_k = ∫[a, b] t^k * f(t) dt
def m_k(k, a, b):
    integrand = lambda t: (t**k) * f(t, a, b)
    return simpson_integrate(integrand, a, b, n=1000)

# Средняя наработка до отказа (T_mid) – m_1
def T_mid(a, b):
    return m_k(1, a, b)

# Дисперсия (D) = m_2 - (T_mid)^2
def D(a, b):
    m2 = m_k(2, a, b)
    return m2 - T_mid(a, b)**2

# Среднее квадратическое отклонение (sigma) — корень из D
def sigma(a, b):
    return np.sqrt(D(a, b))

# Вероятность безотказной работы P(t) = ∫[t, b] f(x) dx
def P(t, a, b):
    lower = t if t > a else a
    return simpson_integrate(lambda x: f(x, a, b), lower, b, n=1000)

# Интенсивность отказов λ(t) = f(t)/P(t)
def lambda_t(t, a, b):
    P_val = P(t, a, b)
    return f(t, a, b) / P_val if P_val > 0 else 0

# Гамма-процентная наработка до отказа T_gamma:
# Определяем такое t, что 1 - ∫[a, t] f(x) dx = (γ/100)
def T_gamma(gamma, a, b, tol=1e-4):
    target = gamma / 100  # требуемая доля отказов
    low = a
    high = b
    while high - low > tol:
        mid = (low + high) / 2
        F_mid = simpson_integrate(lambda x: f(x, a, b), a, mid, n=1000)
        survival = 1 - F_mid  # вероятность безотказной работы до mid
        if survival > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# Функция для построения графиков характеристик:
# 1. P(t): Вероятность безотказной работы
# 5. f(t): Плотность распределения времени до отказа
# 4. λ(t): Интенсивность отказов
# 6. T_gamma: Гамма-процентная наработка до отказа
def plot_graphs(a, b):
    t_values = np.linspace(a, b, 1000)
    f_values = [f(t, a, b) for t in t_values]              # f(t)
    P_values = [P(t, a, b) for t in t_values]                # P(t)
    lambda_values = [lambda_t(t, a, b) for t in t_values]    # λ(t)
    T_gamma_values = [T_gamma(g, a, b) for g in range(0, 101, 10)]  # T_gamma при γ=0,10,...,100

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

# Заданные параметры для распределения Симпсона: S(23, 1000)
a, b = 23, 1000

# Вывод численных характеристик в консоль
print("Simpson Distribution Characteristics (метод численного интегрирования Симпсона):")
print("1. Вероятность безотказной работы P(t) при t = a (t = {}): {:.6f}".format(a, P(a, a, b)))
print("2. Средняя наработка до отказа T_mid: {:.2f}".format(T_mid(a, b)))
print("3. Дисперсия D: {:.2f}".format(D(a, b)))
print("   Среднее квадратическое отклонение sigma: {:.2f}".format(sigma(a, b)))
example_t = 500
print("4. Интенсивность отказов λ(t) при t = {}: {:.6f}".format(example_t, lambda_t(example_t, a, b)))
print("5. Плотность распределения f(t) при t = {}: {:.6f}".format(example_t, f(example_t, a, b)))
print("6. Гамма-процентная наработка T_gamma:")
for gamma in range(0, 101, 10):
    print("   γ = {:3d}%  →  T_gamma = {:.2f}".format(gamma, T_gamma(gamma, a, b)))

# Построение графиков
plot_graphs(a, b)