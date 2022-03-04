from functools import reduce
from operator import matmul
from typing import Callable

import numpy as np


def create_char_pol(*roots: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create characteristic polynomial from roots dynamically

    Return function that look like x -> (x - root[0]) @ (x - root[1]) @ ...
    (Subtraction is not element-wise, it's like in true math!)
    """
    dim = len(roots)

    char_pol = lambda x: (
        reduce(matmul, [x - np.identity(dim) * root for root in roots])
    )

    return char_pol


def get_control(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    C = np.column_stack([
        b,
        A @ b,
        np.linalg.matrix_power(A, 2) @ b,
        np.linalg.matrix_power(A, 3) @ b
    ])

    theta = (
        -np.array([[0, 0, 0, 1]])
        @ np.linalg.inv(C)
        @ create_char_pol(-1, -2, -7.5, -4)(A)
    )

    return theta

