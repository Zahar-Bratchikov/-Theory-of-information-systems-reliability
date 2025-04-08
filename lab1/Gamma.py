import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Заданные параметры Гамма-распределения: Г(9,67)
k = 9       # Форма
theta = 67  # Масштаб

# Временная сетка для построения графиков
t = np.linspace(0, 5000, 1000)

# 5. Плотность распределения времени до отказа f(t)
f_t = stats.gamma.pdf(t, k, scale=theta)

# 1. Вероятность безотказной работы P(t) = 1 - F(t)
P_t = stats.gamma.sf(t, k, scale=theta)

# 4. Интенсивность отказов λ(t) = f(t) / P(t)
lambda_t = np.divide(f_t, P_t, out=np.zeros_like(f_t), where=P_t > 0)

# 6. Гамма-процентная наработка T_gamma:
# По условию: 1 - F(T_gamma) = γ/100  =>  T_gamma = stats.gamma.ppf(1 - (γ/100), k, scale=theta)
gamma_percentages = np.linspace(0, 100, 11)  # от 0% до 100% с шагом 10%
T_gamma = np.array([stats.gamma.ppf(1 - (g / 100), k, scale=theta) for g in gamma_percentages])

# Основные характеристики (численные значения выводятся в консоль только для параметров без графиков)
T_mid = stats.gamma.mean(k, scale=theta)   # Математическое ожидание
D_t = stats.gamma.var(k, scale=theta)     # Дисперсия
sigma = stats.gamma.std(k, scale=theta) # Среднеквадратическое отклонение
#Интенсивность отказов в момент времени t = 1000
sample_t = 1000
lambda_val = stats.gamma.pdf(sample_t, k, scale=theta) / stats.gamma.sf(sample_t, k, scale=theta)

# Построение графиков для:
#   - Плотности распределения f(t),
#   - Вероятности безотказной работы P(t),
#   - Гамма-процентной наработки T_gamma,
#   - Интенсивности отказов λ(t)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Плотность распределения f(t)
axs[0, 0].plot(t, f_t, label="f(t)", color="tab:blue")
axs[0, 0].set_xlabel("Время t")
axs[0, 0].set_ylabel("f(t)")
axs[0, 0].set_title("Плотность распределения f(t)")
axs[0, 0].legend()
axs[0, 0].grid()

# График 2: Вероятность безотказной работы P(t)
axs[0, 1].plot(t, P_t, label="P(t)", color="tab:green")
axs[0, 1].set_xlabel("Время t")
axs[0, 1].set_ylabel("P(t)")
axs[0, 1].set_title("Вероятность безотказной работы P(t)")
axs[0, 1].legend()
axs[0, 1].grid()

# График 3: Гамма-процентная наработка T_gamma
axs[1, 0].plot(gamma_percentages, T_gamma, marker='o', linestyle='-', color='blue', label="T_gamma")
axs[1, 0].set_xlabel("γ, %")
axs[1, 0].set_ylabel("Время T_gamma")
axs[1, 0].set_title("Гамма-процентная наработка T_gamma")
axs[1, 0].legend()
axs[1, 0].grid()

# График 4: Интенсивность отказов λ(t)
axs[1, 1].plot(t, lambda_t, label="λ(t)", color="tab:red")
axs[1, 1].set_xlabel("Время t")
axs[1, 1].set_ylabel("λ(t)")
axs[1, 1].set_title("Интенсивность отказов λ(t)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()

# Вывод в консоль численных характеристик для параметров, для которых графики не строятся
print("Gamma Distribution Numerical Characteristics (Г(9,67)):")
print("Средняя наработка до отказа T_mid: {:.2f}".format(T_mid))
print("Дисперсия D: {:.2f}".format(D_t))
print("Среднее квадратическое отклонение sigma: {:.2f}" .format(sigma))
print("Интенсивность отказов λ(t) при t = {}: {:.6f}".format(sample_t, lambda_val))
