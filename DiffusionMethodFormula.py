import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import time


class MittagLefflerPlotter:
    def __init__(self, a=-1.0, t_end=10.0, num_points=500, K=100) -> None:
        self.a = float(a)
        self.t = np.linspace(0, t_end, num_points)
        self.K = K
        print(f"Plotter initialized with a={self.a}, K={self.K}")

    def _approx_series(self, t, alpha, beta_offset) -> np.ndarray:
        t_safe = np.where(t == 0, 1e-100, t)

        z = self.a * (t_safe ** alpha)

        k0_term = 1.0 / gamma(beta_offset)
        series_sum = np.full_like(t, k0_term, dtype=float)

        for k in range(1, self.K):
            numerator = (z ** k)
            gamma_arg = alpha * k + beta_offset
            denominator = gamma(gamma_arg)

            if denominator == 0 or np.isinf(denominator):
                term = 0.0
            else:
                term = numerator / denominator

            series_sum += term

        return series_sum

    def calculate_function(self, t_power, beta_offset, alpha):
        series_sum = self._approx_series(self.t, alpha, beta_offset)

        y = (self.t ** t_power) * series_sum

        if t_power == 0:
            y[self.t == 0] = 1.0 / gamma(beta_offset)
        else:
            y[self.t == 0] = 0.0

        t_prefix = ""
        if t_power == 1:
            t_prefix = "t"
        elif t_power > 1:
            t_prefix = f"t^{{{t_power}}}"

        beta_from_t_power = t_power + 1

        label = fr'${t_prefix}E_{{{alpha:.2f}, {beta_offset:.2f}}}({self.a} t^{{{alpha:.2f}}})$'

        return y, label

    def plot_function(self, t_power, beta_offset, alphas, filename):
        print(f"Plotting function: t_power={t_power}, beta_offset={beta_offset}")
        start_time = time.time()

        plt.figure(figsize=(10, 6))

        t_prefix = ""
        if t_power == 1:
            t_prefix = "t"
        elif t_power > 1:
            t_prefix = f"t^{{{t_power}}}"

        title = (
            fr'$f(x) = {t_prefix}E_{{\alpha, {beta_offset}}}({{a}} t^{{\alpha}})\quad({{a=-1.0}})$'
        )

        y_limit = (-1.2, 1.2)

        for alpha in alphas:
            y, label = self.calculate_function(t_power, beta_offset, alpha)
            plt.plot(self.t, y, label=label, linewidth=2)

        # グラフ装飾
        plt.title(title, fontsize=16)
        plt.xlabel(r'$x$', fontsize=14)
        plt.ylabel(r'$f(x)$', fontsize=14)
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


if __name__ == '__main__':
    plotter = MittagLefflerPlotter(t_end=20.0, a=-1.00, K=100)
    alphas_to_plot = [1.30, 1.64, 1.96]

    print("\n--- Plot of Function f1 (E_{alpha, 1}) ---")
    plotter.plot_function(
        t_power=0,
        beta_offset=1,
        alphas=alphas_to_plot,
        filename='ML_DML.png'
    )