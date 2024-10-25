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
