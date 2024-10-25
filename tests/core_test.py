import unittest

import numpy as np

from src.core import Variable, Square


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
