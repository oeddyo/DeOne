import numpy as np
from numpy.typing import NDArray

from src.core import Function


class Square(Function):
    def forward(self, x: NDArray) -> NDArray:
        return x**2

    def backward(self, d_out: NDArray) -> NDArray:
        x: NDArray = self.input.data
        g: NDArray = 2 * d_out * x
        return g


class Exp(Function):
    def forward(self, x: NDArray) -> NDArray:
        y: NDArray = np.exp(x)
        return y

    def backward(self, d_out: NDArray) -> NDArray:
        x: NDArray = self.input.data
        r: NDArray = d_out * np.exp(x)
        return r
