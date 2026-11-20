"""Run thermal checks for CuCrZr."""

import math
from functools import partial

import numpy as np

from functions import run_injector_report, run_chamber_checks, run_sections


CUCRZR = dict(name='CuCrZr', k=lambda T: 320.0 + 0 * np.asarray(T), rho=8900, cp=385,
                       T_sus=743.0, T_max=773.0, note="Softens 470-500 C")


def run_heat_soak():
    depth_mm = math.sqrt(9.3e-5 * 7) * 1e3
    print(f"Copper reference: diffusion depth in 7 s = {depth_mm:.0f} mm")


if __name__ == "__main__":
    run_chamber_checks(CUCRZR)
    run_sections((
        ("CuCrZr: INJECTOR FACE", partial(run_injector_report, CUCRZR)),
        ("COPPER REFERENCE HEAT SOAK", run_heat_soak),
    ))
