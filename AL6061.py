"""Run the A6061-RAM2 outer-body thermal checks."""

import numpy as np

from functions import Q_AVG_BASIS, Q_PEAK_BASIS, run_chamber_checks, run_sections, size_channels


AL6061 = dict(name='A6061-RAM2', k=lambda T: 119.0 + 0 * np.asarray(T), rho=2730, cp=900,
                       T_sus=473.0, T_max=573.0,
                       note="Rapid yield loss 473-573 K (Elementum/Akhani data)")


def run_al6061_thickness_sweep():
    print("\nA6061-RAM2: wall thickness needed to stay <= 473 K at 16.7 MW/m^2 (outer body):")
    for tw in (1.0, 0.7, 0.5, 0.3):
        r = size_channels("Outer body", "outer", 85e-3, 104, 2.55, AL6061, Q_AVG_BASIS, Q_PEAK_BASIS,
                          t_wall=tw * 1e-3, quiet=True)
        print(f"  {tw:.1f} mm wall -> hot wall {r['T_wg']:.0f} K, coolant-side {r['T_wc']:.0f} K (boils at {r['T_sat']:.0f} K)")


if __name__ == "__main__":
    run_chamber_checks(AL6061)
    run_sections((
        ("A6061-RAM2: HOT-WALL THICKNESS SWEEP", run_al6061_thickness_sweep),
    ))
