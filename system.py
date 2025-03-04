#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon, triang
from scipy.integrate import quad
from scipy.optimize import brentq

# -----------------------------
# Параметры распределений для элементов:
# -----------------------------
# Первый элемент: Gamma(9, 67) -> k=9, theta=67
k1, theta1 = 9, 67
dist1 = gamma(a=k1, scale=theta1)

# Второй элемент: Exp(1.5e-4) -> lambda = 1.5e-4, scale = 1/lambda
lmbd2 = 1.5e-4
dist2 = expon(scale=1/lmbd2)

# Третий элемент: Треугольное распределение S(23, 1000)
# Вычисляем параметры для симметричного треугольного распределения
a3, b3 = 23, 1000
mode3 = (a3 + b3) / 2
c3 = (mode3 - a3) / (b3 - a3)  # для симметричного распределения должно быть 0.5
dist3 = triang(c=c3, loc=a3, scale=(b3 - a3))

# -----------------------------
# Функции для расчёта показателей системы
# -----------------------------
def reliability(dist, t):
    """Вероятность безотказной работы R(t) = 1 - F(t)"""
    return dist.sf(t)  # Survival function = 1 - CDF

def system_reliability(t):
    """Системная вероятность безотказной работы для последовательно соединённых элементов"""
    R1 = reliability(dist1, t)
    R2 = reliability(dist2, t)
    R3 = reliability(dist3, t)
    return R1 * R2 * R3

def system_pdf(t):
    """
    Плотность вероятности времени отказа системы,
    где f_sys(t) = f1 * R2 * R3 + f2 * R1 * R3 + f3 * R1 * R2.
    """
    f1 = dist1.pdf(t)
    f2 = dist2.pdf(t)
    f3 = dist3.pdf(t)
    R1 = reliability(dist1, t)
    R2 = reliability(dist2, t)
    R3 = reliability(dist3, t)
    return f1 * R2 * R3 + f2 * R1 * R3 + f3 * R1 * R2

def system_hazard(t):
    """Интенсивность отказов системы: h(t) = f_sys(t) / R_sys(t)"""
    R_sys = system_reliability(t)
    f_sys = system_pdf(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(R_sys > 0, f_sys / R_sys, 0)
    return h

def system_mttf():
    """Средняя наработка до отказа системы (MTTF): интеграл от R_sys(t) dt"""
    mttf, err = quad(system_reliability, 0, 2000)
    return mttf

def system_ppf(q, t_min=0, t_max=2000):
    """
    Находит такое время t, что F_sys(t)=q,
    где F_sys(t)=1-R_sys(t). Используется метод Брента.
    """
    func = lambda t: (1 - system_reliability(t)) - q
    try:
        return brentq(func, t_min, t_max)
    except ValueError:
        return np.nan

# -----------------------------
# Основные расчёты для системы
# -----------------------------
mttf_sys = system_mttf()
print(f"Средняя наработка до отказа системы (MTTF): {mttf_sys:.2f}")

# Расчет гамма-процентных наработок: γ = 0, 10, 20, ..., 100
gammas = np.arange(0, 110, 10) / 100.0  # 0.0, 0.1, ..., 1.0
gamma_quantiles = np.array([system_ppf(q) for q in gammas])

print("\nГамма-процентные наработки (квантили) для системы:")
print("Гамма (%)\tВремя до отказа")
for gamma_val, t_val in zip(gammas * 100, gamma_quantiles):
    print(f"{gamma_val:8.0f}%\t{t_val:10.2f}")

# -----------------------------
# Построение графиков в одном окне
# -----------------------------
# Определяем временную сетку для построения графиков
t = np.linspace(0, 1500, 1000)

# Расчет показателей системы на временной сетке
R_sys = system_reliability(t)
f_sys = system_pdf(t)
h_sys = system_hazard(t)

# Настраиваем один рисунок с несколькими подграфиками (2 строки x 2 столбца)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Показатели надежности системы", fontsize=16)

# График 1: Надежность системы (вероятность безотказной работы)
axs[0, 0].plot(t, R_sys, color="blue")
axs[0, 0].set_title("Надежность R_sys(t)")
axs[0, 0].set_xlabel("Время")
axs[0, 0].set_ylabel("R_sys(t)")
axs[0, 0].grid(True)

# График 2: Плотность вероятности (PDF)
axs[0, 1].plot(t, f_sys, color="red")
axs[0, 1].set_title("Плотность вероятности f_sys(t)")
axs[0, 1].set_xlabel("Время")
axs[0, 1].set_ylabel("f_sys(t)")
axs[0, 1].grid(True)

# График 3: Интенсивность отказов
axs[1, 0].plot(t, h_sys, color="green")
axs[1, 0].set_title("Интенсивность отказов h_sys(t)")
axs[1, 0].set_xlabel("Время")
axs[1, 0].set_ylabel("h_sys(t)")
axs[1, 0].grid(True)

# График 4: Гамма-процентные наработки (квантили)
width = 4  # ширина столбцов
axs[1, 1].bar(gammas * 100, gamma_quantiles, width=8, color="purple", alpha=0.7)
axs[1, 1].plot(gammas * 100, gamma_quantiles, 'ko-', label="Квантили")
axs[1, 1].set_title("Гамма-процентные наработки")
axs[1, 1].set_xlabel("γ, %")
axs[1, 1].set_ylabel("Время до отказа")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()