import numpy as np

# Функция для вычисления собственных значений матрицы
def eigenvalues(matrix):
    return np.linalg.eigvals(matrix)

# Число обусловленности по спектральному критерию
def spectral_condition_number(matrix):
    return np.linalg.norm(matrix)*np.linalg.norm(np.linalg.inv(matrix))

# Число обусловленности по критерию Ортеги
def ortega_condition_number(matrix):
    vol = 1
    for n in range(matrix.shape[0]):
        vol *= np.linalg.norm(matrix[n])
    return vol / abs(np.linalg.det(matrix))

# Число обусловленности по угловому критерию
def angular_condition_number(matrix):
    C = np.linalg.inv(matrix)
    return max([np.linalg.norm(a_n) * np.linalg.norm(c_n) for a_n, c_n in zip(matrix, np.transpose(C))])

# Функция для решения СЛАУ и вычисления погрешности
def solve_and_evaluate_error(matrix, b, perturbation):
    x = np.linalg.solve(matrix, b)
    b_perturbed = b + perturbation
    matrix_perturbed = matrix + perturbation
    x_perturbed = np.linalg.solve(matrix_perturbed, b_perturbed)
    return np.linalg.norm(x - x_perturbed)

# Генерация тестов для разных матриц
def generate_tests():
    tests = []

    # Матрицы Гильберта
    for n in range(3, 11):
        hilbert_matrix = np.linalg.inv(np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]))
        b = np.random.rand(n)
        tests.append((hilbert_matrix, b))

    # Матрицы из методички Пакулиной (пример для теста)
    matrix_pakulina = np.array([[-401.64, 200.12],
               [21200.72,  -601.76]])
    b_pakulina = np.array([10, 6])
    tests.append((matrix_pakulina, b_pakulina))

    # Трёхдиагональная матрица с диагональным преобладанием
    n = 5
    tridiag_matrix = np.diag([4] * n) + np.diag([-1] * (n - 1), -1) + np.diag([-1] * (n - 1), 1)
    b_tridiag = np.random.rand(n)
    tests.append((tridiag_matrix, b_tridiag))

    return tests

# Основной цикл для тестирования
if __name__ == "__main__":
    tests = generate_tests()
    perturbations = [10**-i for i in range(2, 11)]

    for i, (matrix, b) in enumerate(tests):
        print(f"Тест {i + 1}")

        # Вычисление числа обусловленности
        spectral = spectral_condition_number(matrix)
        ortega = ortega_condition_number(matrix)
        angular = angular_condition_number(matrix)

        print(f"Спектральный критерий: {spectral}")
        print(f"Критерий Ортеги: {ortega}")
        print(f"Угловой критерий: {angular}")

        # Вычисление погрешности при варьировании правой части
        for perturbation in perturbations:
            error = solve_and_evaluate_error(matrix, b, perturbation)
            print(f"Погрешность при возмущении {perturbation}: {error}")
        print()
