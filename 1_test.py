import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
from slae_analysis import analyze_slae

class TestSLAEConditioning(unittest.TestCase):
    def setUp(self):
        # Диапазон возмущений
        self.perturbation_range = np.logspace(-10, -2, 50)

    def plot_results(self, title, cond_number, errors):
        plt.plot(self.perturbation_range, errors, label=f"cond={cond_number:.2e}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Perturbation ε")
        plt.ylabel("Error |x - x̃|")
        plt.title(title)
        plt.legend()
        plt.grid(True)

    def test_hilbert_matrix(self):
        """Проверка и визуализация для матриц Гильберта."""
        plt.figure(figsize=(8, 6))
        for n in range(3, 6):
            A = hilbert(n)
            b = np.ones(n)
            cond_number, errors = analyze_slae(A, b, self.perturbation_range)
            self.plot_results(f"Hilbert matrix (n={n})", cond_number, errors)
            self.assertGreater(cond_number, 1e3, "Condition number should be high for Hilbert matrices.")
            self.assertGreater(np.max(errors), 1e-5, "Error should be noticeable for ill-conditioned matrices.")
        plt.show()

    def test_tridiagonal_matrix(self):
        """Проверка и визуализация для трехдиагональной матрицы."""
        plt.figure(figsize=(8, 6))
        n = 5
        A = np.diag([4] * n) + np.diag([-1] * (n - 1), k=1) + np.diag([-1] * (n - 1), k=-1)
        b = np.ones(n)
        cond_number, errors = analyze_slae(A, b, self.perturbation_range)
        self.plot_results("Tridiagonal matrix with diagonal dominance", cond_number, errors)
        self.assertLess(cond_number, 1e2, "Condition number should be small for well-conditioned matrices.")
        self.assertLess(np.max(errors), 1e-5, "Error should be small for well-conditioned matrices.")
        plt.show()

    def test_random_symmetric_matrix(self):
        """Проверка и визуализация для случайной симметричной матрицы."""
        plt.figure(figsize=(8, 6))
        np.random.seed(0)
        A = np.random.rand(5, 5)
        A = (A + A.T) / 2  # Симметричная матрица
        b = np.random.rand(5)
        cond_number, errors = analyze_slae(A, b, self.perturbation_range)
        self.plot_results("Random symmetric matrix", cond_number, errors)
        self.assertGreater(cond_number, 1e1, "Condition number should be moderately high for symmetric matrices.")
        self.assertLess(np.max(errors), 1e-3, "Error should not be very high for symmetric matrix.")
        plt.show()

    def test_identity_matrix(self):
        """Проверка и визуализация для единичной матрицы."""
        plt.figure(figsize=(8, 6))
        A = np.eye(3)
        b = np.array([1, 2, 3])
        cond_number, errors = analyze_slae(A, b, self.perturbation_range)
        self.plot_results("Identity matrix", cond_number, errors)
        self.assertAlmostEqual(cond_number, 1, delta=1e-10, msg="Condition number should be 1 for the identity matrix.")
        self.assertLess(np.max(errors), 1e-10, "Error should be close to zero for the identity matrix.")
        plt.show()

    def test_ill_conditioned_matrix(self):
        """Проверка и визуализация для плохо обусловленной матрицы."""
        plt.figure(figsize=(8, 6))
        A = np.array([[1, 0.99], [0.99, 0.98]])  # Плохо обусловленная матрица
        b = np.array([1, 1])
        cond_number, errors = analyze_slae(A, b, self.perturbation_range)
        self.plot_results("Highly ill-conditioned matrix", cond_number, errors)
        self.assertGreater(cond_number, 1e5, "Condition number should be high for ill-conditioned matrices.")
        self.assertGreater(np.max(errors), 1e-2, "Error should be significant for highly ill-conditioned matrix.")
        plt.show()

# Запуск тестов
if __name__ == "__main__":
    unittest.main()

