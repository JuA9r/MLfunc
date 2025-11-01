import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time


class MittagLefflerPlotter:
    def __init__(self, m, omega=1.0, t_end=10.0, num_points=500, K=100) -> None:
        self.m = float(m)
        self.omega = omega
        self.t = np.linspace(0, t_end, num_points)
        self.K = K
        print(f"Plotter initialized with m={self.m}, ω={self.omega}, K={self.K}")

    def _approx_series(self, t, alpha_ml, beta_offset_scalar) -> np.ndarray:
        t_safe = np.where(t == 0, 1e-100, t)
        z = -(self.omega * t_safe) ** alpha_ml

        # k=0 の項: z^0 / Gamma(beta_offset)
        k0_term = 1.0 / gamma(beta_offset_scalar)
        series_sum = np.full_like(t, k0_term, dtype=float)

        for k in range(1, self.K):
            numerator = (z ** k)
            gamma_arg = alpha_ml * k + beta_offset_scalar
            denominator = gamma(gamma_arg)
            term = numerator / denominator
            series_sum += term

        return series_sum

    def calculate_f1(self, alpha):
        """
        Calculates f1(t) = E_{m*a, 1}(-(\omega*t)^{m*a})
        """
        alpha_ml = self.m * alpha
        beta_offset = 1.0

        y = self._approx_series(self.t, alpha_ml, beta_offset)

        # t=0 -> 1 (E_{a,1}(0) = 1)
        y[self.t == 0] = 1.0

        label = fr'$E_{{{alpha_ml:.2f}, 1}}(-z)$: $\alpha={alpha:.2f}$'
        return y, label

    def calculate_f2(self, alpha):
        """
        Calculates f2(t) = t^a * E_{m*a, a+1}(-(\omega*t)^{m*a})
        """
        alpha_ml = self.m * alpha
        beta_offset = alpha + 1.0

        series_sum = self._approx_series(self.t, alpha_ml, beta_offset)

        y = (self.t ** alpha) * series_sum

        # t=0 -> 0
        y[self.t == 0] = 0.0

        label = fr'$t^{{\alpha}}E_{{{alpha_ml:.2f}, \alpha+1}}(-z)$: $\alpha={alpha:.2f}$'  # ★ 修正
        return y, label

    def calculate_f3(self, alpha):
        """
        Calculates f3(t) = t^{2a} * E_{m*a, 2a+1}(-(\omega*t)^{m*a})
        """
        alpha_ml = self.m * alpha
        beta_offset = (2.0 * alpha) + 1.0

        series_sum = self._approx_series(self.t, alpha_ml, beta_offset)

        y = (self.t ** (2.0 * alpha)) * series_sum

        # t=0 -> 0
        y[self.t == 0] = 0.0

        label = fr'$t^{{2\alpha}}E_{{{alpha_ml:.2f}, 2\alpha+1}}(-z)$: $\alpha={alpha:.2f}$'  # ★ 修正
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
            title = r'$f_2(t) = t^\alpha E_{3\alpha, \alpha+1}(-(\omega t)^{3\alpha})$ with $\omega=1$'  # ★ 修正
            y_limit = (-7.6, 7.6)

        elif function_name == 'f3':
            calculator = self.calculate_f3
            title = r'$f_3(t) = t^{2\alpha} E_{3\alpha, 2\alpha+1}(-(\omega t)^{3\alpha})$ with $\omega=1$'  # ★ 修正
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
    # m=3.0
    plotter = MittagLefflerPlotter(m=3.0, t_end=20.0, omega=1.00, K=100)

    alphas_to_plot = [0.30, 0.60, 0.70, 1.0]

    # 1. Plot f1 = E_{3a, 1}
    print("\n--- Plot of Function f1 ---")
    plotter.plot_function(
        function_name='f1',
        alphas=alphas_to_plot,
        filename='Sequential_f1_E3a_1_v2.png'
    )

    # 2. Plot f2 = t^a * E_{3a, a+1}
    print("\n--- Plot of Function f2 ---")
    plotter.plot_function(
        function_name='f2',
        alphas=alphas_to_plot,
        filename='Sequential_f2_tE3a_a+1_v2.png'
    )

    # 3. Plot f3 = t^{2a} * E_{3a, 2a+1}
    print("\n--- Plot of Function f3 ---")
    plotter.plot_function(
        function_name='f3',
        alphas=alphas_to_plot,
        filename='Sequential_f3_t2E3a_2a+1_v2.png'
    )

print("\nSequential plots (v2) generated successfully.")