from __future__ import annotations  # Add this to allow circular type ref

import numpy as np
from numpy.typing import NDArray

from src.util import as_array


class Variable:
    def __init__(self, data: NDArray) -> None:
        # variable only allows None or NDArray as data
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "Type {} is not supported. Variable supports only np.NDArray"
            )

        self.data: NDArray = data
        self.grad: NDArray | None = None
        self.creator: Function | None = None

    def set_creator(self, creator: Function) -> None:
        self.creator = creator

    def backward(self) -> None:
        """Do back propagation from this variable"""

        if self.creator is None:
            raise TypeError("bad bad")

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: list[Function] = [self.creator]

        while funcs:
            creator = funcs.pop()
            x, y = creator.input, creator.output

            if y.grad is None:
                raise ValueError("grad cannot be None during backprop")

            prev_g = creator.backward(y.grad)
            x.grad = prev_g

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        r = self.forward(input.data)

        # force output as array because numpy on 0 dimension will output scalar
        r = as_array(r)
        output = Variable(r)
        self.output = output
        output.set_creator(self)

        return output

    def forward(self, x: NDArray) -> NDArray:
        raise NotImplementedError()

    def backward(self, d_out: NDArray) -> NDArray:
        raise NotImplementedError()
