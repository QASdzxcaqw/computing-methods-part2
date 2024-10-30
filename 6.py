import numpy as np

# Функция для вычисления пределов для собственных значений по теореме Гершгорина
def gershgorin_bounds(A):
    bounds = []
    for i in range(A.shape[0]):
        radius = np.sum(np.abs(A[i])) - np.abs(A[i, i])
        bounds.append((A[i, i] - radius, A[i, i] + radius))
    return bounds

# Метод Якоби
def jacobi_method(A, epsilon, strategy="max_offdiag", max_iterations=1000):
    n = A.shape[0]
    A = A.copy()  # Рабочая копия матрицы
    U = np.eye(n)  # Матрица поворотов
    iterations = 0

    def max_offdiag_element(A):
        # Находит наибольший по модулю внедиагональный элемент
        max_val, k, l = 0, 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if np.abs(A[i, j]) > max_val:
                    max_val = np.abs(A[i, j])
                    k, l = i, j
        return k, l

    for _ in range(max_iterations):
        if strategy == "max_offdiag":
            k, l = max_offdiag_element(A)
        elif strategy == "cyclic":
            k, l = (iterations % (n * (n - 1) // 2)) // n, (iterations % (n * (n - 1) // 2)) % n
            if k >= l:
                k, l = k, (l + 1) % n  # смещаем индексы для выборки только внедиагональных элементов

        if np.abs(A[k, l]) < epsilon:
            break

        # Вычисляем угол поворота
        if A[k, k] == A[l, l]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[k, l] / (A[k, k] - A[l, l]))

        cos, sin = np.cos(theta), np.sin(theta)

        # Поворот матрицы A
        for i in range(n):
            if i != k and i != l:
                A_ik, A_il = A[i, k], A[i, l]
                A[i, k] = A[k, i] = cos * A_ik - sin * A_il
                A[i, l] = A[l, i] = sin * A_ik + cos * A_il
        A_kk, A_ll, A_kl = A[k, k], A[l, l], A[k, l]
        A[k, k] = cos**2 * A_kk + sin**2 * A_ll - 2 * sin * cos * A_kl
        A[l, l] = sin**2 * A_kk + cos**2 * A_ll + 2 * sin * cos * A_kl
        A[k, l] = A[l, k] = 0

        # Поворот матрицы собственных векторов
        for i in range(n):
            U_ik, U_il = U[i, k], U[i, l]
            U[i, k] = cos * U_ik - sin * U_il
            U[i, l] = sin * U_ik + cos * U_il

        iterations += 1
        if np.max(np.abs(np.triu(A, 1))) < epsilon:
            break

    eigenvalues = np.diag(A)
    return eigenvalues, U, iterations

# Функция для тестирования
def test_jacobi_method(n, epsilon_values, strategy):
    # Создаем матрицу Гильберта порядка n
    A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
    bounds = gershgorin_bounds(A)

    for epsilon in epsilon_values:
        print(f"\nТочность ε = {epsilon}, стратегия = {strategy}")

        eigenvalues, eigenvectors, iterations = jacobi_method(A, epsilon, strategy=strategy)
        print(f"Найденные собственные значения: {eigenvalues}")
        print(f"Количество итераций: {iterations}")

        # Проверка теоремы Гершгорина
        for i, eig in enumerate(eigenvalues):
            within_bounds = any(lower <= eig <= upper for (lower, upper) in bounds)
            print(f"Собственное значение {eig} {'входит' if within_bounds else 'НЕ входит'} в пределы Гершгорина")

# Запуск тестов
if __name__ == "__main__":
    n = 5  # для тестов можно увеличить до 10, 15 или более для больших порядков
    epsilon_values = [1e-2, 1e-4, 1e-6]

    # Тестируем обе стратегии
    for strategy in ["max_offdiag", "cyclic"]:
        test_jacobi_method(n, epsilon_values, strategy)
