import numpy as np
import scipy
import scipy.sparse as sp
from scipy.signal import hilbert


# Функция для создания симметричной разреженной матрицы с диагональным преобладанием
def create_sparse_diagonally_dominant_matrix(size):
    # Генерируем случайную разреженную матрицу
    A = sp.random(size, size, density=0.01, format="csr")
    A = A + A.T  # Делает матрицу симметричной

    # Добавляем диагональное преобладание
    for i in range(size):
        A[i, i] = np.sum(np.abs(A[i])) + 1
    return A.toarray()  # Приводим к формату массива NumPy


def iterational_method(alpha,beta,x0,epsilon):
    num_of_iters = 1
    x1 = alpha @ x0 + beta
    while np.linalg.norm(x1 - x0)>epsilon and num_of_iters < 1000:
        x0 = x1
        x1 = alpha @ x0 + beta
        num_of_iters += 1
    return x1, num_of_iters


# Метод простой итерации
def simple_iteration_method(a, b, epsilon, x0):
    alpha = np.zeros([a.shape[0], a.shape[1]])
    beta = np.zeros(b.shape[0])
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if i != j:
                alpha[i, j] = -a[i, j] / a[i, i]
                beta[i] = b[i] / a[i, i]
    return iterational_method(alpha, beta, x0, epsilon)


# Метод Зейделя
def seidel_method(a, b, epsilon, x0):
    n, m = a.shape[0], a.shape[1]
    l, r, d = [np.zeros([n, m]) for _ in range(3)]
    for i in range(n):
        for j in range(m):
            if i > j:
                l[i, j] = a[i, j]
            elif i < j:
                r[i, j] = a[i, j]
            else:
                d[i, j] = a[i, j]
    beta = np.linalg.inv(d + l)
    return iterational_method(-beta @ r, beta @ x0, x0, epsilon)


# Тестирование работы методов
def test_methods( epsilon_values):
    a = scipy.linalg.hilbert(3)
    b = np.random.rand(3)
    x = np.linalg.solve(a, b)
    for epsilon in epsilon_values:
        print(f"\nТочность ε = {epsilon}")

        # Метод простой итерации
        solution_simple, iterations_simple = simple_iteration_method(a, b, epsilon, b)
        print(f"Метод простой итерации: количество итераций = {iterations_simple}, погрешность = {np.linalg.norm(solution_simple - x)}")

        # Метод Зейделя
        solution_seidel, iterations_seidel = seidel_method(a, b, epsilon, b)
        print(f"Метод Зейделя: количество итераций = {iterations_seidel}, погрешность = {np.linalg.norm(solution_seidel - x)}")


# Запуск тестов
if __name__ == "__main__":
    size = 200
    epsilon_values = [1e-2, 1e-4, 1e-6]
    test_methods(epsilon_values)
