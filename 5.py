import numpy as np

# Степенной метод
def power_method(a, epsilon, x0):
    if x0 is None:
        x0 = np.random.uniform(-1, 1, size=a.shape[1])
    x1 = a @ x0
    num_of_iters = 1
    lambda0 = x1[0] / x0[0]
    while True:
        x0, x1 = x1, a @ x1
        lambda1 = x1[0] / x0[0]
        if abs(lambda1 - lambda0) < epsilon or num_of_iters > 5000:
            break
        lambda0 = lambda1
        num_of_iters += 1
    return abs(lambda1), num_of_iters

# Метод скалярных произведений
def scalar_product_method(a, epsilon, x0):
    if x0 is None:
        x0 = np.random.uniform(-1, 1, size=a.shape[1])
    num_of_iters = 1
    x1 = a @ x0
    y0 = np.copy(x0)
    a_T = np.transpose(a)
    y1 = a_T @ x0
    lambda0 = np.dot(x1, y1) / np.dot(x0, y0)
    while True:
        x0, x1 = x1, a @ x1
        y0, y1 = y1, a_T @ y1
        lambda1 = np.dot(x1, y1) / np.dot(x0, y1)
        if abs(lambda1 - lambda0) < epsilon or num_of_iters > 5000:
            break
        lambda0 = lambda1
        num_of_iters += 1
    return abs(lambda1), num_of_iters

# Тестирование и сравнение методов
def test_methods(A, epsilon_values):
    x0 = np.ones(A.shape[1])
    for epsilon in epsilon_values:
        print(f"\nТочность ε = {epsilon}")

        # Степенной метод
        lambda_power, iterations_power = power_method(A, epsilon, x0)
        print(f"Степенной метод: собственное значение = {lambda_power}, количество итераций = {iterations_power}")

        # Метод скалярных произведений
        lambda_scalar, iterations_scalar = scalar_product_method(A, epsilon, x0)
        print(f"Метод скалярных произведений: собственное значение = {lambda_scalar}, количество итераций = {iterations_scalar}")

        # Проверка с библиотечной функцией для сравнения
        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_eigenvalue_index = np.argmax(np.abs(eigenvalues))
        max_eigenvalue = eigenvalues[max_eigenvalue_index]
        corresponding_eigenvector = eigenvectors[:, max_eigenvalue_index]

        print(f"Максимальное собственное значение (библиотека): {max_eigenvalue}")
        print(f"Собственный вектор (библиотека): {corresponding_eigenvector}\n")

# Пример теста
if __name__ == "__main__":
    # Пример: Матрица Гильберта 5x5
    n = 4
    A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
    epsilon_values = [1e-2, 1e-4, 1e-6]

    test_methods(A, epsilon_values)
