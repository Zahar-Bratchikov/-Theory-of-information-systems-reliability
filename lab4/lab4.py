from math import comb
from fractions import Fraction

# Исходные данные
P = 0.97
k = 4
P_switch = [0.99, 0.97, 0.95, 0.93]
k_i = [4, 2, 2, 2, 5]
m1 = Fraction(4, 5)
m2 = Fraction(7, 5)

# 1. Без резервирования (последовательное соединение)
P1 = P ** 5

# 2. Общее резервирование с постоянно включённым резервом (параллельно)
def parallel_system(P_elem, k_total):
    # Вероятность отказа всех k+1 элементов
    return 1 - ((1 - (P_elem ** 5)) ** (k_total + 1))

P2_elem = parallel_system(P, k)
P2 = P2_elem

# 3. Общее резервирование с замещением (с переключателями)
def redundant_with_switches(P_elem, P_switches):
    P_fail = 1 - P_elem
    P_total = P_elem  # система работает, если основной работает

    fail_chain = 1  # вероятность, что все предыдущие резервы не сработали

    for Pswitch in P_switches:
        success = P_elem * Pswitch
        P_total += P_fail * fail_chain * success
        fail_chain *= (1 - success)

    return P_total


    return P_total


def prod(arr):
    result = 1
    for x in arr:
        result *= x
    return result

P3_elem = redundant_with_switches(P, P_switch)
P3 = P3_elem ** 5

# 4. Раздельное резервирование с постоянным включением
def parallel_redundancy(P_elem, k_i):
    return 1 - (1 - P_elem) ** (k_i + 1)

P4 = 1
for ki in k_i:
    P4 *= parallel_redundancy(P, ki)

# 5. Резервирование с дробной кратностью
# Формула: P_sys = 1 - C(n+m, m+1)*(1-P)^(m+1)
def fractional_redundancy(P_elem, m):
    n = m.numerator
    d = m.denominator
    total = 0
    for i in range(n + 1):
        total += comb(n + d, i) * (P_elem ** (n + d - i)) * ((1 - P_elem) ** i)
    return total

P5_1 = fractional_redundancy(P, m1)
P5_2 = fractional_redundancy(P, m2)

# Выигрыш
def gain(Px):
    return Px - P1

# Вывод результатов
print(f"1. Без резервирования: P = {P1:.10f}")
print(f"2. Общее резервирование (постоянно включено, k={k}): P = {P2:.10f}, выигрыш = {gain(P2):.10f}")
print(f"3. Общее резервирование с замещением и переключателями: P = {P3:.10f}, выигрыш = {gain(P3):.10f}")
print(f"4. Раздельное резервирование (по элементам): P = {P4:.10f}, выигрыш = {gain(P4):.10f}")
print(f"5. Дробная кратность m1=4/5: P = {P5_1:.20f}, выигрыш = {gain(P5_1):.10f}")
print(f"5. Дробная кратность m2=7/5: P = {P5_2:.20f}, выигрыш = {gain(P5_2):.10f}")
