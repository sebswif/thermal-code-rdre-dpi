"""Run thermal checks for 316L stainless steel."""

import math
from functools import partial

import numpy as np

from functions import run_injector_report, run_chamber_checks, run_sections


STAINLESS = dict(name='316L', k=lambda T: np.interp(T, [293, 773, 1073], [14.0, 19.5, 23.0]),
                       rho=8000, cp=500, T_sus=1000.0, T_max=1150.0,
                       note="Assumed limits; strength roughly halves by ~1000 K")



def run_heat_soak():
    alpha = 16 / (STAINLESS["rho"] * STAINLESS["cp"])
    depth_mm = math.sqrt(alpha * 7) * 1e3
    print(f"316L heat soak: diffusion depth in 7 s = {depth_mm:.1f} mm")


if __name__ == "__main__":
    run_chamber_checks(STAINLESS)
    run_sections((
        ("316L stainless steel: INJECTOR FACE", partial(run_injector_report, STAINLESS)),
        ("316L HEAT SOAK", run_heat_soak),
    ))
