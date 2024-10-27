from __future__ import annotations  # Add this to allow circular type ref

from typing import override

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
            gys: list[NDArray] = [y.grad for y in creator.outputs]  # type: ignore
            prev_gs = creator.backward(*gys)

            if not isinstance(prev_gs, tuple):
                prev_gs = (prev_gs,)

            for x, gx in zip(creator.inputs, prev_gs, strict=True):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs: Variable) -> Variable | tuple[Variable, ...]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = []
        for y in ys:
            # force output as array because numpy on 0 dimension will output scalar
            output = Variable(as_array(y))
            output.set_creator(self)
            outputs.append(output)

        self.inputs = inputs
        self.outputs = outputs

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def forward(self, *x: NDArray) -> tuple[NDArray, ...]:
        raise NotImplementedError()

    def backward(self, *d_out: NDArray) -> tuple[NDArray, ...]:
        raise NotImplementedError()


# Operations are simple and clean
class Add(Function):
    @override
    def forward(self, a: NDArray, b: NDArray) -> NDArray:  # type: ignore
        r: NDArray = a + b
        return r

    def backward(self, grad: NDArray) -> tuple[NDArray, NDArray]:  # type: ignore
        return grad, grad


"""
f = Add()
v1 = Variable(np.array([12]))
v2 = Variable(np.array([1]))
r = f(v1, v2)

r.backward()

print(v1.grad)
"""
