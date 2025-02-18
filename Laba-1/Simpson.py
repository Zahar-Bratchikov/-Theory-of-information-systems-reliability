import matplotlib.pyplot as plt
import numpy as np

# Функция плотности распределения Симпсона (PDF)
def simpson_pdf(t, a, b):
    if a <= t <= b:
        return (2 / (b - a)) - (2 / (b - a) ** 2) * abs(a + b - 2 * t)
    else:
        return 0

# Средняя наработка до отказа (MTTF) - математическое ожидание
def mean(a, b):
    return (a + b) / 2

# Дисперсия времени до отказа
def variance(a, b):
    return (b - a) ** 2 / 8

# Среднеквадратическое отклонение (СКО) - корень из дисперсии
def std_dev(a, b):
    return variance(a, b) ** 0.5

# Численное интегрирование методом прямоугольников
def integrate(f, start, end, steps=1000):
    step = (end - start) / steps
    area = 0
    for i in range(steps):
        x = start + i * step
        area += f(x) * step
    return area

# Вероятность безотказной работы (R(t)) - интеграл плотности вероятности от t до b
def reliability(t, a, b):
    return integrate(lambda x: simpson_pdf(x, a, b), t, b)

# Интенсивность отказов (λ(t)) - отношение плотности вероятности к вероятности безотказной работы
def failure_rate(t, a, b):
    f_t = simpson_pdf(t, a, b)  # Плотность вероятности f(t)
    r_t = reliability(t, a, b)  # Вероятность безотказной работы R(t)
    return f_t / r_t if r_t > 0 else 0

# Гамма-процентная наработка - значение времени, при котором наработка достигает γ% отказов
def gamma_quantile(gamma, a, b):
    target = gamma / 100  # Преобразуем процент в долю
    t = a  # Начинаем с нижней границы
    step = (b - a) / 10000  # Шаг для перебора значений
    accumulated = 0  # Накопленная вероятность

    # Численно находим момент времени t, при котором достигается заданная вероятность
    while t <= b:
        accumulated += simpson_pdf(t, a, b) * step
        if accumulated >= target:
            return t
        t += step

    return b  # Если не нашли, возвращаем верхнюю границу

# Функция для построения графиков плотности, вероятности, интенсивности и гамма-наработки
def plot_graphs(a, b):
    t_values = np.linspace(a, b, 1000)  # Массив значений времени от a до b

    # Вычисляем значения для графиков
    pdf_values = [simpson_pdf(t, a, b) for t in t_values]  # Плотность f(t)
    reliability_values = [reliability(t, a, b) for t in t_values]  # Вероятность R(t)
    failure_rate_values = [failure_rate(t, a, b) for t in t_values]  # Интенсивность λ(t)

    # Гамма-процентная наработка для γ от 0 до 100 с шагом 10
    gamma_values = [gamma_quantile(g, a, b) for g in range(0, 101, 10)]

    # Создаем область для графиков
    plt.figure(figsize=(12, 8))

    # График 1: Плотность распределения f(t)
    plt.subplot(2, 2, 1)
    plt.plot(t_values, pdf_values, label='Плотность распределения')
    plt.xlabel('Время t')
    plt.ylabel('f(t)')
    plt.legend()

    # График 2: Вероятность безотказной работы R(t)
    plt.subplot(2, 2, 2)
    plt.plot(t_values, reliability_values, label='Вероятность безотказной работы')
    plt.xlabel('Время t')
    plt.ylabel('R(t)')
    plt.legend()

    # График 3: Интенсивность отказов λ(t)
    plt.subplot(2, 2, 3)
    plt.plot(t_values, failure_rate_values, label='Интенсивность отказов')
    plt.xlabel('Время t')
    plt.ylabel('λ(t)')
    plt.legend()

    # График 4: Гамма-процентная наработка t_γ
    plt.subplot(2, 2, 4)
    plt.plot(range(0, 101, 10), gamma_values, label='Гамма-процентная наработка')
    plt.xlabel('γ (%)')
    plt.ylabel('Время t')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Задаем параметры распределения Симпсона (границы a и b)
a, b = 23, 1000

# Выводим основные числовые характеристики
print("Среднее время безотказной работы (MTTF):", mean(a, b))
print("Дисперсия:", variance(a, b))
print("Среднеквадратическое отклонение:", std_dev(a, b))

# Пример: интенсивность отказов при t = 500
t = 500
print(f"Интенсивность отказов при t={t}:", failure_rate(t, a, b))

# Построение графиков
plot_graphs(a, b)
