from typing import Optional

import numpy as np


class PendODESystem:
    def __init__(
            self,
            pend_mass=0.127,
            cart_mass=1.206,
            moment_of_inertia=0.001,
            length=0.178,
            K_f=1.726,
            K_s=4.487,
            B_c=5.4,
            B_p=0.002,
    ) -> None:

        A_0 = np.array([
            [pend_mass + cart_mass, -pend_mass * length],
            [-pend_mass * length, moment_of_inertia + pend_mass * length * length]
        ])

        A_1 = np.array([
            [B_c, 0],
            [0, B_p]
        ])

        A_2 = np.array([
            [0, 0],
            [0, -pend_mass * 9.8 * length]
        ])

        first = -np.linalg.inv(A_0) @ A_2
        second = -np.linalg.inv(A_0) @ A_1
        third = np.linalg.inv(A_0) @ np.array([[1], [0]])

        self.A_ = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [first[0][0], first[0][1], second[0][0], second[0][1]],
            [first[1][0], first[1][1], second[1][0], second[1][1]]
        ])

        self.b_ = np.array([
            [0],
            [0],
            third[0],
            third[1]
        ])

    @property
    def A(self):
        return self.A_

    @property
    def b(self):
        return self.b_


def linear_system(
        t: np.ndarray,
        x: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        theta: np.ndarray,
        x_0: Optional[np.ndarray] = None
) -> np.ndarray:
    if x_0 is None:
        x_0 = x

    return A @ x + b @ theta @ x_0
