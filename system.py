#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon, triang
from scipy.integrate import quad
from scipy.optimize import brentq

# -----------------------------
# Параметры распределений для элементов:
# -----------------------------
# Первый элемент: Gamma(9, 67) -> k=9, θ=67
k1, theta1 = 9, 67
dist1 = gamma(a=k1, scale=theta1)

# Второй элемент: Exp(1.5e-4) -> λ = 1.5e-4, scale = 1/λ
lmbd2 = 1.5e-4
dist2 = expon(scale=1/lmbd2)

# Третий элемент: Треугольное распределение S(23, 1000)
# Для симметричного треугольного распределения mode = (a+b)/2
a3, b3 = 23, 1000
mode3 = (a3 + b3) / 2
c3 = (mode3 - a3) / (b3 - a3)  # для симметричного распределения должно быть 0.5
dist3 = triang(c=c3, loc=a3, scale=(b3 - a3))

# -----------------------------
# Функции для расчёта показателей системы
# -----------------------------
def reliability(dist, t):
    """Вероятность безотказной работы R(t) = 1 - F(t) для данного распределения."""
    return dist.sf(t)

def system_reliability(t):
    """
    Системная вероятность безотказной работы для последовательно соединённых элементов.
    R_sys(t) = R1(t) * R2(t) * R3(t)
    """
    return reliability(dist1, t) * reliability(dist2, t) * reliability(dist3, t)

def system_pdf(t):
    """
    Плотность распределения времени до отказа системы.
    f_sys(t) = f1(t)*R2(t)*R3(t) + f2(t)*R1(t)*R3(t) + f3(t)*R1(t)*R2(t)
    """
    f1 = dist1.pdf(t)
    f2 = dist2.pdf(t)
    f3 = dist3.pdf(t)
    R1 = reliability(dist1, t)
    R2 = reliability(dist2, t)
    R3 = reliability(dist3, t)
    return f1 * R2 * R3 + f2 * R1 * R3 + f3 * R1 * R2

def system_hazard(t):
    """Интенсивность отказов системы: h_sys(t) = f_sys(t) / R_sys(t)"""
    R_sys = system_reliability(t)
    f_sys = system_pdf(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(R_sys > 0, f_sys / R_sys, 0)
    return h

def system_mttf(integral_limit=2000):
    """
    Средняя наработка до отказа (MTTF) системы:
    MTTF = ∫₀∞ R_sys(t) dt.
    """
    mttf, _ = quad(system_reliability, 0, integral_limit)
    return mttf

def system_moment2(integral_limit=2000):
    """
    Второй момент времени безотказной работы системы:
    E[T²] = ∫₀∞ 2t R_sys(t) dt.
    """
    moment2, _ = quad(lambda t: 2 * t * system_reliability(t), 0, integral_limit)
    return moment2

def system_ppf(q, t_min=0, t_max=2000):
    """
    Гамма-процентная наработка до отказа: находит такое время t,
    что F_sys(t) = q, где F_sys(t) = 1 - R_sys(t).
    Используется метод Брента.
    """
    func = lambda t: (1 - system_reliability(t)) - q
    try:
        return brentq(func, t_min, t_max)
    except ValueError:
        return np.nan

# -----------------------------
# Основные расчёты для системы
# -----------------------------
# 1. Вероятность безотказной работы приведена в графике.
# 2. Средняя наработка до отказа (MTTF)
mttf_sys = system_mttf()

# 3. Среднее квадратическое отклонение и дисперсия времени безотказной работы
E_T2 = system_moment2()
variance_sys = E_T2 - mttf_sys**2
std_sys = np.sqrt(variance_sys) if variance_sys > 0 else 0

# 4. Интенсивность отказов представлена в графике.
# 5. Плотность распределения времени до отказа представлена в графике.
# 6. Гамма-процентная наработка до отказа (γ = 0, 10, ..., 100)
gammas = np.arange(0, 110, 10) / 100.0  # 0.0, 0.1, ..., 1.0
gamma_quantiles = np.array([system_ppf(q) for q in gammas])

# Вывод численных значений
print("Результаты для системы:")
print(f"1. Вероятность безотказной работы R_sys(0): {system_reliability(0):.2f} (при t=0 всегда 1)")
print(f"2. Средняя наработка до отказа (MTTF): {mttf_sys:.2f}")
print(f"3. Дисперсия времени безотказной работы: {variance_sys:.2f}")
print(f"   Среднее квадратическое отклонение: {std_sys:.2f}")
print(f"4. Интенсивность отказов h_sys(t) представлена графиком.")
print(f"5. Плотность распределения времени до отказа f_sys(t) представлена графиком.")
print("6. Гамма-процентная наработка до отказа (квантили):")
print("   Гамма (%)\tВремя до отказа")
for gamma_val, t_val in zip(gammas * 100, gamma_quantiles):
    print(f"   {gamma_val:8.0f}%\t{t_val:10.2f}")

# -----------------------------
# Построение графиков в одном окне
# -----------------------------
t = np.linspace(0, 1500, 1000)  # Временная сетка для графиков

# Вычисление показателей системы
R_sys = system_reliability(t)
f_sys = system_pdf(t)
h_sys = system_hazard(t)

# Создаем фигуру с 2 строками и 2 столбцами графиков
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Показатели надежности системы", fontsize=16)

# График 1. Вероятность безотказной работы
axs[0, 0].plot(t, R_sys, color="blue")
axs[0, 0].set_title("1. Вероятность безотказной работы (R_sys(t))")
axs[0, 0].set_xlabel("Время")
axs[0, 0].set_ylabel("R_sys(t)")
axs[0, 0].grid(True)

# График 2. Плотность распределения времени до отказа
axs[0, 1].plot(t, f_sys, color="red")
axs[0, 1].set_title("5. Плотность распределения времени до отказа (f_sys(t))")
axs[0, 1].set_xlabel("Время")
axs[0, 1].set_ylabel("f_sys(t)")
axs[0, 1].grid(True)

# График 3. Интенсивность отказов
axs[1, 0].plot(t, h_sys, color="green")
axs[1, 0].set_title("4. Интенсивность отказов (h_sys(t))")
axs[1, 0].set_xlabel("Время")
axs[1, 0].set_ylabel("h_sys(t)")
axs[1, 0].grid(True)

# График 4. Гамма-процентная наработка до отказа (квантили)
width = 8  # ширина столбцов
axs[1, 1].bar(gammas * 100, gamma_quantiles, width=width, color="purple", alpha=0.7)
axs[1, 1].plot(gammas * 100, gamma_quantiles, 'ko-', label="Квантили")
axs[1, 1].set_title("6. Гамма-процентная наработка до отказа")
axs[1, 1].set_xlabel("γ, %")
axs[1, 1].set_ylabel("Время до отказа")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()