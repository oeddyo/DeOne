import numpy as np
import pytest
from numpy.ma.testutils import assert_almost_equal, assert_equal

from src.core import Exp, Square, Variable


# Former VariableTest class tests
def test_create() -> None:
    with pytest.raises(TypeError):
        # specifically testing scalar mis-use
        Variable(1)  # type: ignore


def test_init() -> None:
    v = Variable(np.array([1, 2, 3]))
    assert_equal(np.array([1, 2, 3]), v.data)
    assert v.grad is None
    assert v.creator is None


# Former TestCore class tests
def test_output_ndarray() -> None:
    x = Variable(np.array(1.0))
    f = Square()
    r = f(x)
    assert isinstance(r.data, np.ndarray)


def test_square() -> None:
    v = Variable(np.array([1, 2, 3]))
    s = Square()
    np.testing.assert_array_equal([1, 4, 9], s(v).data)

    # set input correctly
    np.testing.assert_equal(v.data, s.input.data)
    np.testing.assert_equal([1, 4, 9], s.output.data)


def test_chained_backprop() -> None:
    s1 = Square()
    exp = Exp()
    s2 = Square()

    v = Variable(np.array([0.5]))
    res: Variable = s2(exp(s1(v)))

    res.backward()
    assert_almost_equal(v.grad, np.array([3.29744254]))


def test_gradient_check() -> None:
    f1 = Square()
    f2 = Square()
    f3 = Exp()
    f4 = Square()

    def complex_f(x):  # type: ignore
        return f4(f3(f2(f1(x))))

    def numeric_diff(f, x: Variable):  # type: ignore
        h = 1e-6
        r1 = f(Variable(x.data + h))
        r2 = f(Variable(x.data - h))

        print(r1.data, r2.data)
        g = (r1.data - r2.data) / (2 * h)
        return g

    x = Variable(np.array([0.12, 0.1, 1.02]))
    y = complex_f(x)
    y.backward()

    assert_almost_equal(x.grad, numeric_diff(complex_f, x), 5)
