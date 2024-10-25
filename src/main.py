import numpy as np
from numpy.typing import NDArray


# Basic types
def process_string(text: str) -> str:
    return text.upper()


# Lists
def process_numbers(numbers: list[int]) -> list[float]:
    return [float(n) * 1.5 for n in numbers]


# Optional (when None is allowed)
def maybe_process(text: str | None = None) -> str:
    if text is None:
        return "default"
    return text.upper()


# NumPy arrays
def process_array(data: NDArray[np.float64]) -> NDArray[np.float64]:
    return data * 2.0
