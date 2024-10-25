from __future__ import annotations  # Add this to allow circular type ref

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override


class Variable:
    def __init__(self, data: NDArray) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, creator: Function) -> None:
        pass


class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        r = self.forward(input.data)
        output = Variable(r)

        self.output = output
        output.set_creator(self)

        return output

    def forward(self, x: NDArray) -> NDArray[np.float64]:
        raise NotImplementedError()

    def backward(self, d_out: NDArray) -> NDArray[np.float64]:
        raise NotImplementedError()


class Square(Function):
    @override
    def forward(self, x: NDArray) -> NDArray:
        return x**2

    @override
    def backward(self, d_out: NDArray) -> NDArray:
        x: NDArray[np.float64] = self.input.data
        return d_out * x * 2


class Exp(Function):
    @override
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.exp(x)

    @override
    def backward(self, d_out: NDArray[np.float64]) -> NDArray[np.float64]:
        x: NDArray[np.float64] = self.input.data
        # only here =>
        y: NDArray[np.float64] = d_out * np.exp(x)
        return y


t = np.array([1, 23])
v = Variable(t)
s = Square()
