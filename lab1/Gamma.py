#!/usr/bin/env python3
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Заданные параметры Гамма-распределения: Г(9,67)
k = 9         # Форма
theta = 67    # Масштаб

# Временная сетка для вычислений и построения графиков
t = np.linspace(0, 5000, 1000)

# 5. Плотность распределения времени до отказа f(t)
f_t = stats.gamma.pdf(t, k, scale=theta)

# 1. Вероятность безотказной работы P(t) = 1 - F(t)
P_t = stats.gamma.sf(t, k, scale=theta)

# 4. Интенсивность отказов λ(t) = f(t) / P(t)
lambda_t = np.divide(f_t, P_t, out=np.zeros_like(f_t), where=P_t>0)

# 6. Гамма-процентная наработка до отказа T_gamma
# По условию: 1 - F(T_gamma) = γ/100  =>  T_gamma = stats.gamma.ppf(1 - (γ/100), k, scale=theta)
gamma_percentages = np.linspace(0, 100, 11)  # от 0% до 100% с шагом 10%
T_gamma = np.array([stats.gamma.ppf(1 - (g / 100), k, scale=theta) for g in gamma_percentages])

# Расчёт основных характеристик:
# 2. Средняя наработка до отказа T_mid = E[T] = k * theta
T_mid = k * theta

# 3. Дисперсия D = k * theta^2, и стандартное отклонение sigma = sqrt(D)
D_t = k * theta**2
sigma = np.sqrt(D_t)

# Визуализация графиков
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Плотность распределения времени до отказа f(t)
ax[0, 0].plot(t, f_t, label="f(t)", color="tab:blue")
ax[0, 0].set_title("5. Плотность распределения времени до отказа f(t)")
ax[0, 0].set_xlabel("Время t")
ax[0, 0].set_ylabel("f(t)")
ax[0, 0].legend()
ax[0, 0].grid()

# График 2: Гамма-процентная наработка T_gamma
ax[0, 1].plot(gamma_percentages, T_gamma, marker='o', linestyle='-', color='blue', label="T_gamma")
ax[0, 1].set_title("6. Гамма-процентная наработка T_gamma")
ax[0, 1].set_xlabel("γ, %")
ax[0, 1].set_ylabel("Время T_gamma")
ax[0, 1].legend()
ax[0, 1].grid()

# График 3: Вероятность безотказной работы P(t)
ax[1, 0].plot(t, P_t, label="P(t)", color="purple")
ax[1, 0].set_title("1. Вероятность безотказной работы P(t)")
ax[1, 0].set_xlabel("Время t")
ax[1, 0].set_ylabel("P(t)")
ax[1, 0].legend()
ax[1, 0].grid()

# График 4: Интенсивность отказов λ(t)
ax[1, 1].plot(t, lambda_t, label="λ(t)", color="red")
ax[1, 1].set_title("4. Интенсивность отказов λ(t)")
ax[1, 1].set_xlabel("Время t")
ax[1, 1].set_ylabel("λ(t)")
ax[1, 1].legend()
ax[1, 1].grid()

plt.tight_layout()
plt.show()

# Вывод вычисленных характеристик в консоль
print("Gamma Distribution Reliability Characteristics (Г(9,67)):")
print("1. Вероятность безотказной работы P(t) при t = 0: {:.6f}".format(stats.gamma.sf(0, k, scale=theta)))
print("2. Средняя наработка до отказа (T_mid): {:.2f}".format(T_mid))
print("3. Дисперсия D: {:.2f}".format(D_t))
print("   Среднее квадратическое отклонение sigma: {:.2f}".format(sigma))
sample_t = 1000
print("4. Интенсивность отказов λ(t) при t = {}: {:.6f}".format(sample_t,
      stats.gamma.pdf(sample_t, k, scale=theta) / stats.gamma.sf(sample_t, k, scale=theta)))
print("5. Плотность распределения f(t) при t = {}: {:.6f}".format(sample_t, stats.gamma.pdf(sample_t, k, scale=theta)))
print("6. Гамма-процентная наработка T_gamma:")
for g, t_val in zip(gamma_percentages, T_gamma):
    print("   γ = {:3.0f}%  →  T_gamma = {:.2f}".format(g, t_val))