import numpy as np
from numpy.typing import NDArray
from typing_extensions import override


class Variable:
    def __init__(self, data: NDArray) -> None:
        self.data = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        r = self.forward(input.data)
        return Variable(r)

    def forward(self, x: NDArray) -> NDArray:
        raise NotImplementedError()


class Square(Function):
    @override
    def forward(self, x: NDArray) -> NDArray:
        return x**2


t = np.array([1, 23])
v = Variable(t)
s = Square()
