
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

# Функция плотности распределения времени до отказа по закону распределения Симпсона, f(t)
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

# Гамма-процентная наработка T_gamma:
# Находим такое t, что 1 - ∫[a, t] f(x) dx = γ/100
def T_gamma(gamma, a, b, tol=1e-4):
    target = gamma / 100  # требуемая доля отказов
    low = a
    high = b
    while high - low > tol:
        mid = (low + high) / 2
        F_mid = simpson_integrate(lambda x: f(x, a, b), a, mid, n=1000)
        survival = 1 - F_mid  # вероятность безотказной работы
        if survival > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# Функция для построения графиков:
# Строим графики для:
#   - Плотности распределения f(t),
#   - Вероятности безотказной работы P(t),
#   - Гамма-процентной наработки T_gamma,
#   - Интенсивности отказов λ(t)
def plot_graphs(a, b):
    t_values = np.linspace(a, b, 1000)
    f_values = [f(t, a, b) for t in t_values]
    P_values = [P(t, a, b) for t in t_values]
    lambda_values = [lambda_t(t, a, b) for t in t_values]
    gamma_labels = list(range(0, 101, 10))
    T_gamma_values = [T_gamma(g, a, b) for g in gamma_labels]

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

# Заданные параметры распределения Симпсона: S(23, 1000)
a, b = 23, 1000

# Вывод в консоль численных характеристик, для которых графики не строятся
print("Simpson Distribution Numerical Characteristics:")
print("Средняя наработка до отказа T_mid: {:.2f}".format(T_mid(a, b)))
print("Дисперсия D: {:.2f}".format(D(a, b)))
print("Среднее квадратическое отклонение sigma: {:.2f}".format(sigma(a, b)))
# Для параметров f(t), P(t), T_gamma и λ(t) строятся графики – консольный вывод не производится.

# Построение графиков для f(t), P(t), T_gamma и λ(t)
plot_graphs(a, b)