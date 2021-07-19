import numpy as np


class LowPassFilter:
    def __init__(self, dimension: int, h: float, w_cutoff: float):
        if w_cutoff == np.inf:
            self.a = 1.
        else:
            self.a = h * w_cutoff / (1 + h * w_cutoff)

        self.n = dimension
        self.x = None

    def update(self, u: np.array):
        assert u.size == self.n

        if self.x is None:
            self.x = u
        else:
            self.x = (1 - self.a) * self.x + self.a * u

    def reset_state(self):
        self.x = None

    def has_valid_state(self):
        return self.x is not None

    def get_current_state(self):
        assert self.x is not None
        return self.x.copy()
