"""
- Вероятность безотказной работы P(t)
- Плотность распределения отказов f(t)
- Интенсивность отказов λ(t)
- Среднее время до отказа Tср
- Дисперсию и среднеквадратичное отклонение
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_reliability_metrics(n, delta_t, t_total):
    """
    Рассчитывает основные метрики надежности системы.
    
    Параметры:
    n (np.array): массив количества отказов в каждом временном интервале
    delta_t (int): длительность временного интервала
    t_total (int): общее время наблюдения
    
    Возвращает:
    tuple: (t, N_t, P_t, f_t, lambda_t, T_avg, D, sigma)
    """
    N0 = np.sum(n)
    t = np.arange(0, t_total, delta_t) + delta_t / 2
    
    # Расчет количества работающих элементов
    N_t = N0 - np.cumsum(n) + n
    
    # Расчет основных метрик надежности
    P_t = N_t / N0
    f_t = n / (N0 * delta_t)
    lambda_t = np.where(N_t != 0, n / (N_t * delta_t), np.nan)
    
    # Расчет статистических характеристик
    ti_expanded = np.repeat(t, n)
    T_avg = np.mean(ti_expanded)
    D = np.var(ti_expanded, ddof=1)
    sigma = np.sqrt(D)
    
    return t, N_t, P_t, f_t, lambda_t, T_avg, D, sigma

def print_results(t, n, N_t, P_t, lambda_t, f_t, T_avg):
    """Выводит результаты расчетов в табличном формате."""
    print("\nРезультаты расчетов:")
    print("t (мин) | n(t) | N(t) | P(t)   | Tср     | λ(t)     | f(t)")
    print("-" * 60)
    
    for i in range(len(t)):
        lam = f"{lambda_t[i]:.4f}" if not np.isnan(lambda_t[i]) else " — "
        print(f"{t[i]:>6} | {n[i]:>4} | {N_t[i]:>4} | {P_t[i]:.4f} | {T_avg:>7.4f} | {lam:>8} | {f_t[i]:.4f}")

def plot_reliability_metrics(t, P_t, f_t, lambda_t):
    """Строит графики основных метрик надежности."""
    plt.figure(figsize=(14, 10))
    
    # График вероятности безотказной работы
    plt.subplot(3, 1, 1)
    plt.plot(t, P_t, marker='o', color='green')
    plt.title("Вероятность безотказной работы P(t)")
    plt.xlabel("Время (мин)")
    plt.ylabel("P(t)")
    plt.grid(True)
    
    # График плотности распределения отказов
    plt.subplot(3, 1, 2)
    plt.bar(t, f_t, width=8, color='blue', alpha=0.6)
    plt.title("Плотность распределения времени до отказа f(t)")
    plt.xlabel("Время (мин)")
    plt.ylabel("f(t)")
    plt.grid(True)
    
    # График интенсивности отказов
    plt.subplot(3, 1, 3)
    plt.plot(t, lambda_t, marker='x', linestyle='--', color='red')
    plt.title("Интенсивность отказов λ(t)")
    plt.xlabel("Время (мин)")
    plt.ylabel("λ(t)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Основная функция программы."""
    # Исходные данные
    n = np.array([463, 476, 452, 359, 80, 296, 195, 316, 148, 434])
    delta_t = 10
    t_total = 100
    
    # Расчет метрик надежности
    t, N_t, P_t, f_t, lambda_t, T_avg, D, sigma = calculate_reliability_metrics(n, delta_t, t_total)
    
    # Вывод результатов
    print_results(t, n, N_t, P_t, lambda_t, f_t, T_avg)
    
    print("\n--- Итоговые статистические характеристики ---")
    print(f"Среднее время до отказа Tср = {T_avg:.4f} мин")
    print(f"Дисперсия D = {D:.4f}")
    print(f"Среднеквадратичное отклонение σ = {sigma:.4f}")
    
    # Построение графиков
    plot_reliability_metrics(t, P_t, f_t, lambda_t)

if __name__ == "__main__":
    main()
