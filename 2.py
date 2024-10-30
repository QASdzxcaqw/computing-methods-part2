import numpy as np
from slae_analysis import spectral_condition_number

# Функция для поворота элементов матрицы по методу Гивенса
def givens_rotation(matrix, b):
    n = len(b)
    Q = np.eye(n)
    R = matrix.copy()

    for j in range(n - 1):
        for i in range(j + 1, n):
            if R[i, j] != 0:
                r = np.hypot(R[j, j], R[i, j])
                c = R[j, j] / r
                s = -R[i, j] / r

                G = np.eye(n)
                G[j, j] = c
                G[j, i] = -s
                G[i, j] = s
                G[i, i] = c

                R = G @ R
                Q = Q @ G.T

    # Q^T * b - преобразуем вектор правой части
    Qt_b = Q.T @ b
    return R, Qt_b

# Решение верхнетреугольной матрицы методом обратной подстановки
def back_substitution(R, Qt_b):
    n = len(Qt_b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (Qt_b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]

    return x

# Полный метод решения СЛАУ методом Гивенса
def solve_using_givens(matrix, b):
    R, Qt_b = givens_rotation(matrix, b)
    x = back_substitution(R, Qt_b)
    return x



# Проверка на различных матрицах
def test_solution(matrix, b):
    # Решение методом Гивенса
    x_givens = solve_using_givens(matrix, b)

    # Решение встроенным методом для сравнения
    x_builtin = np.linalg.solve(matrix, b)

    # Проверка точности
    print("Решение методом Гивенса:", x_givens)
    print("Решение встроенным методом:", x_builtin)
    print("Погрешность:", np.linalg.norm(x_givens - x_builtin))

    # Числа обусловленности
    cond_original = spectral_condition_number(matrix)
    cond_R = spectral_condition_number(givens_rotation(matrix, b)[0])  # Верхнетреугольная матрица

    print(f"Число обусловленности исходной матрицы: {cond_original}")
    print(f"Число обусловленности матрицы R: {cond_R}")
    print()


# Тесты на разных матрицах
if __name__ == "__main__":
    # Хорошо обусловленная матрица
    A1 = np.array([
        [4, 1, 2],
        [1, 5, 1],
        [2, 1, 3]
    ])
    b1 = np.array([1, 2, 3])
    print("Тест на хорошо обусловленной матрице")
    test_solution(A1, b1)

    # Плохо обусловленная матрица
    A2 = np.array([
        [1, 1, 1],
        [1, 1 + 1e-10, 1],
        [1, 1, 1 + 1e-10]
    ])
    b2 = np.array([3, 3 + 1e-10, 3 + 2 * 1e-10])
    print("Тест на плохо обусловленной матрице")
    test_solution(A2, b2)

    # Матрица Гильберта (очень плохо обусловленная)
    n = 3
    A3 = np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])
    b3 = np.ones(n)
    print("Тест на матрице Гильберта")
    test_solution(A3, b3)
