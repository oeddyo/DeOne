from typing import cast

import numpy as np
import numpy.typing as npt


def as_array(x: int | float | npt.NDArray) -> npt.NDArray:
    if np.isscalar(x):
        return np.array(x)
    return cast(npt.NDArray, x)  # tell mypy this is definitely an NDArray
