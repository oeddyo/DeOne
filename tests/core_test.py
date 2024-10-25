import unittest

import numpy as np
from numpy.ma.testutils import assert_almost_equal

from src.core import Variable, Square, Exp


class TestCore(unittest.TestCase):
    def test_square(self) -> None:
        v = Variable(np.array([1, 2, 3]))
        s = Square()
        np.testing.assert_array_equal([1, 4, 9], s(v).data)

        # set input correctly
        np.testing.assert_equal(v.data, s.input.data)

        np.testing.assert_equal([1, 4, 9], s.output.data)

    def test_backward(self) -> None:
        v = Variable(np.array([3]))
        s = Square()

        s(v)
        print(s.backward(np.array([1])))

    def test_chained_backprop(self) -> None:
        s1 = Square()
        exp = Exp()
        s2 = Square()

        v = Variable(np.array([0.5]))
        res: Variable = s2(exp(s1(v)))

        res.backward()

        assert_almost_equal(v.grad, np.array([3.29744254]))

    def test_gradient_check(self) -> None:
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
