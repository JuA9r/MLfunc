import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time


class MittagLefflerPlotter:
    """
    This class is plot Mittag Leffler function graph.
    """

    def __init__(self, m, omega=1.0, t_end=10.0, num_points=500, K=100) -> None:
        """
        Args:
            :param m: 'm' parameter
            :param omega: 'omega' parameter
            :param t_end: 't_end' parameter
            :param num_points: 'num_points' parameter
            :param K: 'K' parameter
        """
        self.m = float(m)
        self.omega = omega
        self.t = np.linspace(0, t_end, num_points)
        self.K = K
        print(f"Plotter initialized with m={self.m}, ω={self.omega}, K={self.K}")

    def _approx_series(self, t, alpha, beta_offset) -> np.ndarray:
        """
        Calculates the sum of a series.
        Sum_{k=0 to K-1} (z^k) / Gamma(m*alpha*k + beta_offset)
        ここで z = -(\omega*t)^{m*alpha}
        """
        alpha_ml = self.m * alpha
        t_safe = np.where(t == 0, 1e-100, t)
        z = -(self.omega * t_safe) ** alpha_ml

        # k=0: z^0 / Gamma(beta_offset)
        k0_term = 1.0 / gamma(beta_offset)
        series_sum = np.full_like(t, k0_term, dtype=float)

        # k=1 ~ K-1
        for k in range(1, self.K):
            numerator = (z ** k)
            gamma_arg = alpha_ml * k + beta_offset
            denominator = gamma(gamma_arg)
            term = numerator / denominator
            series_sum += term

        return series_sum

    def calculate_function(self, t_power, beta_offset, alpha):
        """
        Calculate functions and generate labels.
        f(t) = t^{t_power} * E_{m*alpha, beta_offset}(-(\omega*t)^{m*alpha})
        """
        # Calculate the sum of a series
        series_sum = self._approx_series(self.t, alpha, beta_offset)

        # Multiply by a power of t
        y = (self.t ** t_power) * series_sum

        # Processing when t=0
        if t_power == 0:
            y[self.t == 0] = 1.0  # E_{beta, 1}(0) = 1
        else:
            y[self.t == 0] = 0.0  # t^p * E_{...}(0) = 0 for p > 0

        # Generate label
        beta_val = self.m * alpha
        t_prefix = ""
        if t_power == 1:
            t_prefix = "t"
        elif t_power > 1:
            t_prefix = f"t^{{{t_power}}}"

        label = fr'${t_prefix}E_{{{beta_val:.2f}, {beta_offset}}}(-z)$: $\alpha={alpha:.2f}$'

        return y, label

    def plot_function(self, t_power, beta_offset, alphas, filename):
        """
        Plots a function with the specified parameters
        """
        print(f"Plotting function: t_power={t_power}, beta_offset={beta_offset}")
        start_time = time.time()

        plt.figure(figsize=(10, 6))

        # Generate title
        t_prefix = ""
        if t_power == 1:
            t_prefix = "t"
        elif t_power > 1:
            t_prefix = f"t^{{{t_power}}}"

        title = (
            fr'$f(t) = {t_prefix}E_{{{self.m}\alpha, '
            fr'{beta_offset}}}(-(\omega t)^{{{self.m}\alpha}})$ with $\omega={self.omega}$'
        )

        # y-axis range (fixed)
        y_limit = (-10.6, 10.6)

        for alpha in alphas:
            y, label = self.calculate_function(t_power, beta_offset, alpha)
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
    plotter = MittagLefflerPlotter(m=4.0, t_end=20.0, omega=1.00, K=100)
    alphas_to_plot = [0.30, 0.64, 1.0]

    # 1. Plot f1 (t^0, offset=1)
    print("\n--- Plot of Function f1 ---")
    plotter.plot_function(
        t_power=0,
        beta_offset=1,
        alphas=alphas_to_plot,
        filename='MittagLeffler_f1_E4a_1_v3.png'
    )

    # 2. Plot f2 (t^1, offset=2)
    print("\n--- Plot of Function f2 ---")
    plotter.plot_function(
        t_power=1,
        beta_offset=2,
        alphas=alphas_to_plot,
        filename='MittagLeffler_f2_tE4a_2_v3.png'
    )

    # 3. Plot f3 (t^2, offset=3)
    print("\n--- Plot of Function f3 ---")
    plotter.plot_function(
        t_power=2,
        beta_offset=3,
        alphas=alphas_to_plot,
        filename='MittagLeffler_f3_t2E4a_3_v3.png'
    )

    # 4. Plot f4 (t^3, offset=4)
    print("\n --- Plot of Function f4 ---")
    plotter.plot_function(
        t_power=3,
        beta_offset=4,
        alphas=alphas_to_plot,
        filename='MittagLeffler_f3_t3E4a_4_v3.png'
    )