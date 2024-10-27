from typing import override

import numpy as np
from numpy.typing import NDArray

from src.core import Function


class Square(Function):
    @override
    def forward(self, x: NDArray) -> NDArray:  # type: ignore
        return x**2

    def backward(self, d_out: NDArray) -> NDArray:  # type: ignore
        x: NDArray = self.inputs[0].data
        g: NDArray = 2 * d_out * x
        return g


class Exp(Function):
    def forward(self, x: NDArray) -> NDArray:  # type: ignore
        y: NDArray = np.exp(x)
        return y

    def backward(self, d_out: NDArray) -> NDArray:  # type: ignore
        x: NDArray = self.inputs[0].data
        r: NDArray = d_out * np.exp(x)
        return r
