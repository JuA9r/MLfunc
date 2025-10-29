import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


class MittagLefflerPlotter:
    """
    A class for drawing graphs of two special functions, including the Mittag-Leffler function.

    Function 1: f1(t) = E_{self.m*alpha, 1}(-(\omega*t)^{self.m*alpha})
    Function 2: f2(t) = (\omega*t)^\alpha E_{self.m*alpha, 1+\alpha}(-(\omega*t)^{self.m*alpha})
    """

    def __init__(self, omega=1.0, t_end=10.0, num_points=500, K=70) -> None:
        """
        Args:
            omega (float): Frequency parameters (omega)
            t_end (float): The maximum value of time t.
            num_points (int): The number of sampling points at time t.
            K (int): The number of terms in a series expansion.
        """
        self.m = input("Enter a decimal number for m : ")
        self.omega = omega
        self.t = np.linspace(0, t_end, num_points)
        self.K = K

    def _approx_f1(self, t, alpha) -> np.ndarray[float]:
        alpha_ml = float(self.m) * alpha
        z = -(self.omega * t) ** alpha_ml
        result = np.zeros_like(t, dtype=float)

        # K=0 term: 1/Gamma(1) = 1
        result += 1.0

        for k in range(1, self.K):
            # Term: z^k / Gamma(k*alpha_ml + 1)
            term = (z ** k) / gamma(k * alpha_ml + 1)
            result += term

        return result

    def _approx_f2(self, t, alpha) -> np.ndarray[float]:
        # sum_{k=0}^{\infty} [ (-1)^k * (omega*t)^{\alpha*(self.m*k+1)} / Gamma(self.m*alpha*k + 1+alpha) ]

        result = np.zeros_like(t, dtype=float)

        for k in range(self.K):
            # Numerator: (-1)^k * (omega*t)^{\alpha*(self.m*k+1)}
            exponent = alpha * (float(self.m) * k + 1)
            numerator = ((-1.0) ** k) * np.power(self.omega * t, exponent)

            # Denominator: Gamma(self.m*alpha*k + 1+alpha)
            gamma_arg = float(self.m) * alpha * k + 1 + alpha
            denominator = gamma(gamma_arg)

            term = numerator / denominator
            result += term

        return result

    def calculate_f1(self, alpha) -> tuple[np.ndarray[float], str]:
        if alpha == 0.5:
            # alpha=0.5 -> e^{-(\omega t)}
            y = np.exp(-(self.omega * self.t))
            label = fr'$E_{{{self.m}\alpha, 1}}(-z)$: $\alpha=0.5$ ($e^{{-\omega t}}$)'

        elif alpha == 1.0:
            # alpha=1.0 -> cos(\omega t)
            y = np.cos(self.omega * self.t)
            label = fr'$E_{{{self.m}\alpha, 1}}(-z)$: $\alpha=1.0$ ($\cos(\omega t)$)'

        else:
            y = self._approx_f1(self.t, alpha)
            label = fr'$E_{{{self.m}\alpha, 1}}(-z)$: $\alpha={alpha:.2f}$'

        return y, label

    def calculate_f2(self, alpha) -> tuple[np.ndarray[float], str]:
        if alpha == 1.0:
            # alpha=1.0 -> sin(\omega t)
            y = np.sin(self.omega * self.t)
            label = fr'$(\omega t)^\alpha E_{{{self.m}\alpha, 1+\alpha}}(-z)$: $\alpha=1.0$ ($\sin(\omega t)$)'

        else:
            y = self._approx_f2(self.t, alpha)
            label = fr'$(\omega t)^\alpha E_{{{self.m}\alpha, 1+\alpha}}(-z)$: $\alpha={alpha:.2f}$'

        return y, label

    def plot_function(self, function_name, alphas, filename='plot.png') -> None:
        """
        Draws a graph using the specified function and alpha value.

        Args:
            function_name (str): 'f1' or 'f2'
            alphas (list): A list of alpha values to draw
            filename (str): File name to save
        """
        plt.figure(figsize=(10, 6))
        if function_name == 'f1':
            calculator = self.calculate_f1
            title = fr'$f_1(t) = E_{{{self.m}\alpha, 1}}(-(\omega t)^{{{self.m}\alpha}})$ with $\omega=1$'

        elif function_name == 'f2':
            calculator = self.calculate_f2
            title = fr'$f_2(t) = (\omega t)^\alpha E_{{{self.m}\alpha, 1+\alpha}}(-(\omega t)^{{{self.m}\alpha}})$' \
                    ' with $\omega=1$'
        else:
            raise ValueError

        for alpha in alphas:
            y, label = calculator(alpha)
            plt.plot(self.t, y, label=label)

        # graph decoration
        plt.title(title, fontsize=16)
        plt.xlabel(r'$t$', fontsize=14)
        plt.ylabel(r'$f(t)$', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(color='black', linewidth=0.5)
        plt.legend(fontsize=12, loc='lower right')
        plt.ylim(-1.1, 1.1)
        plt.savefig(filename)
        plt.close()
        print(f"Graph saved to '{filename}'.")


# --- execution part ---
if __name__ == '__main__':
    plotter = MittagLefflerPlotter(t_end=50.0)
    alphas_to_plot = [0.30, 0.64, 1.0]

    # 1. Draw the graph of the first function (E_{2*alpha, 1})
    print("--- Plot of Function 1 ---")
    plotter.plot_function(
        function_name='f1',
        alphas=alphas_to_plot,
        filename=f'MittagLeffler_F1_m={plotter.m}.png'
    )

    # 2. Plot the graph of the second function ((\omega t)^\alpha E_{2*alpha, 1+alpha})
    print("\n--- Plot of Function 2 ---")
    plotter.plot_function(
        function_name='f2',
        alphas=alphas_to_plot,
        filename=f'MittagLeffler_F2_m={plotter.m}.png'
    )