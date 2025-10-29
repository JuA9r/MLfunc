import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


class MittagLefflerPlotter:
    def __init__(self, omega=1.0, t_end=10.0, num_points=500, K=70) -> None: ...