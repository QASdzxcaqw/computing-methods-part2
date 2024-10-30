import numpy as np
import matplotlib.pyplot as plt
import time

# Функция Розенброка
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

# Градиент функции Розенброка
def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    grad[1] = 200 * (x[1] - x[0] ** 2)
    return grad

# Метод градиентного спуска
def gradient_descent(starting_point, learning_rate, beta,  max_iter):
    x = starting_point
    history = []
    for _ in range(max_iter):
        history.append(x.copy())
        x -= learning_rate * rosenbrock_gradient(x)
    return x, history

# Метод тяжелого шарика
def heavy_ball_method(starting_point, learning_rate, beta, max_iter):
    x = starting_point
    v = np.zeros_like(x)
    history = []
    for _ in range(max_iter):
        history.append(x.copy())
        grad = rosenbrock_gradient(x)
        v = beta * v + learning_rate * grad
        x -= v
    return x, history

# Метод Нестерова
def nesterov_method(starting_point, learning_rate, beta, max_iter):
    x = starting_point
    v = np.zeros_like(x)
    history = []
    for _ in range(max_iter):
        history.append(x.copy())
        grad = rosenbrock_gradient(x - beta * v)
        v = beta * v + learning_rate * grad
        x -= v
    return x, history

# Метод Ньютона
def newton_method(starting_point, max_iter):
    x = starting_point
    history = []
    for _ in range(max_iter):
        history.append(x.copy())
        grad = rosenbrock_gradient(x)
        hessian = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                            [-400 * x[0], 200]])
        x -= np.linalg.solve(hessian, grad)
    return x, history

# Сравнение методов
def compare_methods():
    starting_point = np.array([-1.2, 1.0])
    max_iter = 1000
    learning_rate = 0.001
    beta = 0.9

    # Сравнение времени
    methods = {
        "Gradient Descent": gradient_descent,
        "Heavy Ball": heavy_ball_method,
        "Nesterov": nesterov_method,
        "Newton": newton_method
    }

    results = {}
    for name, method in methods.items():
        start_time = time.time()
        if name == "Newton":
            final_x, history = method(starting_point, max_iter)
        else:
            final_x, history = method(starting_point, learning_rate, beta, max_iter)
        elapsed_time = time.time() - start_time
        results[name] = {
            "final_x": final_x,
            "elapsed_time": elapsed_time,
            "function_calls": len(history),
            "final_value": rosenbrock(final_x)
        }

    # Печать результатов
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Final x: {result['final_x']}")
        print(f"  Time: {result['elapsed_time']:.4f}s")
        print(f"  Function calls: {result['function_calls']}")
        print(f"  Final value: {result['final_value']:.4f}\n")

# Визуализация
def visualize_methods():
    starting_point = np.array([-1.2, 1.0])
    max_iter = 1000
    learning_rate = 0.001
    beta = 0.9

    methods = {
        "Gradient Descent": gradient_descent,
        "Heavy Ball": heavy_ball_method,
        "Nesterov": nesterov_method,
        "Newton": newton_method
    }

    plt.figure(figsize=(12, 8))

    for name, method in methods.items():
        if name == "Newton":
            final_x, history = method(starting_point, max_iter)
        else:
            final_x, history = method(starting_point, learning_rate, beta, max_iter)

        history = np.array(history)
        plt.plot(history[:, 0], history[:, 1], label=name)

    plt.title('Trajectory of Optimization Methods')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_methods()
    visualize_methods()
