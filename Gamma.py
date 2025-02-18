import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Заданные параметры гамма-распределения
k = 3   # форма
theta = 2  # масштаб

# Функция плотности вероятности (PDF)
t = np.linspace(0, 20, 1000)
pdf = stats.gamma.pdf(t, k, scale=theta)

# Вероятность безотказной работы (Reliability Function)
P_t = stats.gamma.sf(t, k, scale=theta)

# Интенсивность отказов
lambda_t = pdf / P_t

# Вычисление гамма-процентной наработки
gamma_values = np.linspace(0, 1, 11)  # от 0 до 100%
T_gamma = stats.gamma.ppf(gamma_values, k, scale=theta)

# Вычисление T_m, дисперсии и стандартного отклонения
T_m = k * theta  # Средняя наработка до отказа (ср время)
D_t = k * theta ** 2  # Дисперсия
std_dev = np.sqrt(D_t)  # Среднее квадратическое отклонение

# Визуализация
fig, ax = plt.subplots(4, 1, figsize=(8, 20))

ax[0].plot(t, pdf, label='Плотность вероятности (PDF)')
ax[0].set_title('Плотность распределения Парето')
ax[0].set_xlabel('Время')
ax[0].set_ylabel('Плотность')
ax[0].legend()
ax[0].grid()

ax[1].plot(gamma_values * 100, T_gamma, marker='o', linestyle='-', color='blue', label='Гамма-процентная наработка')
ax[1].set_title('Гамма-процентная наработка до отказа')
ax[1].set_xlabel('Процент (γ)')
ax[1].set_ylabel('Время Tγ')
ax[1].legend()
ax[1].grid()

ax[2].plot(t, P_t, label='Вероятность безотказной работы', color='purple')
ax[2].set_title('Вероятность безотказной работы')
ax[2].set_xlabel('Время')
ax[2].set_ylabel('P(T > t)')
ax[2].legend()
ax[2].grid()

ax[3].plot(t, lambda_t, label='Интенсивность отказов', color='red')
ax[3].set_title('График интенсивности отказов')
ax[3].set_xlabel('Время')
ax[3].set_ylabel('λ(t)')
ax[3].legend()
ax[3].grid()

plt.tight_layout()
plt.show()

# Вывод расчетных значений
print("Средняя наработка до отказа (MTTF):", T_m)
print("Дисперсия времени безотказной работы:", D_t)
print("Среднее квадратическое отклонение:", std_dev)


