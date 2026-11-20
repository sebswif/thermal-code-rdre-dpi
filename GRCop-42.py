"""Run the GRCop-42 chamber thermal checks."""

import numpy as np

from functions import run_chamber_checks


def k_grcop(T):  # GRCop-42 notes: 350 W/mK max, 320 W/mK at 1073 K (decreasing)
    return np.interp(T, [293.0, 1073.0], [350.0, 320.0])


GRCOP = dict(name='GRCop-42', k=k_grcop, rho=8800, cp=385, T_sus=800.0, T_max=900.0,
                       note="SSoB limits; data to 1073 K, yield 25 ksi printed")

if __name__ == "__main__":
    run_chamber_checks(GRCOP)
