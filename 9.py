import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Задаем параметры
L = 10.0  # Длина стержня
T = 1.0  # Время
alpha = 0.1  # Коэффициент теплопроводности
Nx = 50  # Число пространственных узлов
Nt = 200  # Число временных шагов
dx = L / (Nx - 1)  # Шаг по пространству
dt = T / Nt  # Шаг по времени


# Инициализация начального распределения температуры
def initial_condition(Nx):
    u = np.zeros(Nx)
    for i in range(Nx):
        u[i] = np.sin(np.pi * i * dx / L)  # Начальное распределение
    return u


# Явная схема
def explicit_scheme(Nx, Nt, dt):
    u = np.zeros((Nt, Nx))
    u[0, :] = initial_condition(Nx)

    for n in range(1, Nt):
        for i in range(1, Nx - 1):
            u[n, i] = u[n - 1, i] + alpha * dt / dx ** 2 * (u[n - 1, i + 1] - 2 * u[n - 1, i] + u[n - 1, i - 1])

        # Условие для границ
        u[n, 0] = u[n, -1] = 0  # Дирихлеевские условия

    return u


# Неявная схема
def implicit_scheme(Nx, Nt, dt):
    u = np.zeros((Nt, Nx))
    u[0, :] = initial_condition(Nx)

    # Коэффициенты матрицы
    A = np.zeros((Nx - 2, Nx - 2))
    for i in range(Nx - 2):
        if i > 0:
            A[i, i - 1] = -alpha * dt / dx ** 2
        A[i, i] = 1 + 2 * alpha * dt / dx ** 2
        if i < Nx - 3:
            A[i, i + 1] = -alpha * dt / dx ** 2

    for n in range(1, Nt):
        b = u[n - 1, 1:-1].copy()
        b[0] += alpha * dt / dx ** 2 * u[n - 1, 0]  # Левое граничное условие
        b[-1] += alpha * dt / dx ** 2 * u[n - 1, -1]  # Правое граничное условие
        u[n, 1:-1] = np.linalg.solve(A, b)

        # Условие для границ
        u[n, 0] = u[n, -1] = 0  # Дирихлеевские условия

    return u


# Поведение решения при несоблюдении условий устойчивости
def unstable_behavior():
    dt_values = [0.1, 0.5, 1.0]  # Значения dt, которые могут быть неустойчивыми
    plt.figure(figsize=(15, 10))

    for dt in dt_values:
        Nt = int(T / dt)
        u = explicit_scheme(Nx, Nt)

        # Визуализация
        X = np.linspace(0, L, Nx)
        T = np.linspace(0, T, Nt)
        X, T = np.meshgrid(X, T)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, T, u, cmap='viridis')
        ax.set_title(f'Explicit Scheme (dt = {dt})')
        ax.set_xlabel('Space')
        ax.set_ylabel('Time')
        ax.set_zlabel('Temperature')
        plt.show()


# Основная функция
def main(T=None):
    # Неявная схема
    u_implicit = implicit_scheme(Nx, Nt)
    X = np.linspace(0, L, Nx)
    T = np.linspace(0, T, Nt)
    X, T = np.meshgrid(X, T)

    # Визуализация неявной схемы
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u_implicit, cmap='viridis')
    ax.set_title('Implicit Scheme')
    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Temperature')
    plt.show()

    # Поведение явной схемы при несоблюдении условий устойчивости
    unstable_behavior()


# Запуск программы
if __name__ == "__main__":
    main()
