import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time


class MittagLefflerPlotter:
    """
        A class for drawing graphs of Mittag-Leffler related functions.
    """

    def __init__(self, m, omega=1.0, t_end=10.0, num_points=500, K=100) -> None:
        """
        Args:
            :param m: The 'm' parameter (float)
            :param omega: Angular frequency
            :param t_end: End time for plotting
            :param num_points: Number of points to plot
            :param K: Number of terms for series approximation
        """

        self.m = float(m)
        self.omega = omega
        self.t = np.linspace(0, t_end, num_points)
        self.K = K
        print(f"Plotter initialized with m={self.m}, ω={self.omega}, K={self.K}")

    def _approx_f1(self, t, alpha) -> np.ndarray:
        """
        Calculates f1(t) = E_{m*alpha, 1}(-(\omega*t)^{m*alpha})

        This is calculated as:
        f1(t) = [ Sum_{k=0 to K-1} (z^k) / Gamma(m*alpha*k + 1) ]
        where z = -(\omega*t)^{m*alpha}
        """
        alpha_ml = self.m * alpha
        t_safe = np.where(t == 0, 1e-100, t)
        z = -(self.omega * t_safe) ** alpha_ml

        # k=0 の項: z^0 / Gamma(1) = 1
        series_sum = np.ones_like(t, dtype=float)

        # k=1 から K-1 の項
        for k in range(1, self.K):
            numerator = (z ** k)
            gamma_arg = alpha_ml * k + 1
            denominator = gamma(gamma_arg)
            term = numerator / denominator
            series_sum += term

        # t=0 で 1 になる
        series_sum[t == 0] = 1.0

        return series_sum

    def _approx_f2(self, t, alpha):
        """
        Calculates f2(t) = t * E_{m*alpha, 2}(-(\omega*t)^{m*alpha})
        """
        alpha_ml = self.m * alpha
        # ★ 修正: t_safe を使用
        t_safe = np.where(t == 0, 1e-100, t)
        z = -(self.omega * t_safe) ** alpha_ml

        # k=0 の項: z^0 / Gamma(2) = 1
        series_sum = np.ones_like(t, dtype=float)

        # k=1 から K-1 の項
        for k in range(1, self.K):
            numerator = (z ** k)
            gamma_arg = alpha_ml * k + 2
            denominator = gamma(gamma_arg)
            term = numerator / denominator
            series_sum += term

        result = t * series_sum
        result[t == 0] = 0.0
        return result

    def _approx_f3(self, t, alpha):
        """
        Calculates f3(t) = t^2 * E_{m*alpha, 3}(-(\omega*t)^{m*alpha})
        """
        alpha_ml = self.m * alpha
        t_safe = np.where(t == 0, 1e-100, t)
        z = -(self.omega * t_safe) ** alpha_ml

        # k=0 の項: z^0 / Gamma(3) = 1/2
        series_sum = np.full_like(t, 0.5, dtype=float)

        for k in range(1, self.K):
            numerator = (z ** k)
            gamma_arg = alpha_ml * k + 3
            denominator = gamma(gamma_arg)
            term = numerator / denominator
            series_sum += term

        result = (t ** 2) * series_sum
        result[t == 0] = 0.0
        return result

    def calculate_f1(self, alpha):
        """
        Calculates f1 and prepares the label.
        """
        y = self._approx_f1(self.t, alpha)
        beta = self.m * alpha
        label = fr'$E_{{{beta:.2f}, 1}}(-z)$: $\alpha={alpha:.2f}$'
        return y, label

    def calculate_f2(self, alpha):
        """
        Calculates f2 and prepares the label.
        """
        y = self._approx_f2(self.t, alpha)
        beta = self.m * alpha
        label = fr'$tE_{{{beta:.2f}, 2}}(-z)$: $\alpha={alpha:.2f}$'
        return y, label

    def calculate_f3(self, alpha):
        """
        Calculates f3 and prepares the label.
        """
        y = self._approx_f3(self.t, alpha)
        beta = self.m * alpha
        label = fr'$t^2E_{{{beta:.2f}, 3}}(-z)$: $\alpha={alpha:.2f}$'
        return y, label

    def plot_function(self, function_name, alphas, filename):
        """
        Plots the specified function for a list of alpha values.
        """
        print(f"Plotting function: {function_name}")
        start_time = time.time()

        plt.figure(figsize=(10, 6))

        if function_name == 'f1':
            calculator = self.calculate_f1
            title = r'$f_1(t) = E_{3\alpha, 1}(-(\omega t)^{3\alpha})$ with $\omega=1$'
            y_limit = (-7.6, 7.6)

        elif function_name == 'f2':
            calculator = self.calculate_f2
            title = r'$f_2(t) = tE_{3\alpha, 2}(-(\omega t)^{3\alpha})$ with $\omega=1$'
            y_limit = (-7.6, 7.6)

        elif function_name == 'f3':
            calculator = self.calculate_f3
            title = r'$f_3(t) = t^2E_{3\alpha, 3}(-(\omega t)^{3\alpha})$ with $\omega=1$'
            y_limit = (-7.6, 7.6)

        else:
            raise ValueError(f"Unknown function_name: {function_name}")

        for alpha in alphas:
            y, label = calculator(alpha)
            plt.plot(self.t, y, label=label, linewidth=2)

        # graph decoration
        plt.title(title, fontsize=16)
        plt.xlabel(r'$t$', fontsize=14)
        plt.ylabel(r'$f(t)$', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(color='black', linewidth=0.5)
        plt.legend(fontsize=12, loc='best')
        plt.ylim(y_limit)
        plt.tight_layout()
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig(filename)
        plt.close()

        end_time = time.time()
        print(f"Graph saved to '{filename}'.")
        print(f"Plot generation took {end_time - start_time:.2f} seconds.")


# --- execution part ---
if __name__ == '__main__':
    plotter = MittagLefflerPlotter(m=3.0, t_end=20.0, omega=1.00, K=100)
    alphas_to_plot = [0.30, 0.64, 1.0]

    # 1. Plot f1 = E_{3*alpha, 1}
    print("\n--- Plot of Function f1 ---")
    plotter.plot_function(
        function_name='f1',
        alphas=alphas_to_plot,
        filename='MittagLeffler_f1_E3a_1.png'
    )

    # 2. Plot f2 = t * E_{3*alpha, 2}
    print("\n--- Plot of Function f2 ---")
    plotter.plot_function(
        function_name='f2',
        alphas=alphas_to_plot,
        filename='MittagLeffler_f2_tE3a_2.png'
    )

    # 3. Plot f3 = t^2 * E_{3*alpha, 3}
    print("\n--- Plot of Function f3 ---")
    plotter.plot_function(
        function_name='f3',
        alphas=alphas_to_plot,
        filename='MittagLeffler_f3_t2E3a_3.png'
    )