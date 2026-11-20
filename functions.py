"""Shared thermal models and reporting for the chamber and injector."""

import math
from functools import partial
import numpy as np
from CoolProp.CoolProp import PropsSI

PSI = 6894.757

# =============================================================================
# INPUTS
# =============================================================================
# --- operating point (SSoB rows 10-19) ---
PC_PSIA, OF = 271.0, 3.0
M_OX, M_F = 0.6555, 0.2185            # kg/s
T_FIRE = 7.0                          # s total firing (row 17)

# --- chamber gas temperature from RocketCEA (fallback = SSoB row 194) ---
try:
    from rocketcea.cea_obj import CEA_Obj
    T_GAS = CEA_Obj(oxName="LOX", fuelName="CH4").get_Tcomb(Pc=PC_PSIA, MR=OF) / 1.8
    T_GAS_SRC = "RocketCEA"
except Exception:
    T_GAS, T_GAS_SRC = 3319.0, "SSoB fallback (RocketCEA not installed)"

# --- bodies and cooling (SSoB rows 28-37, 195-210) ---
L_CH = 51e-3                          # m, injector face to throat
BODIES = {   # name: (side, hot-wall diameter, channels, water kg/s)
    "Outer body": ("outer", 85e-3, 104, 2.55),
    "Centerbody": ("inner", 75e-3, 96, 2.17),
}
W_CH, H_CH = 1.0e-3, 1.3e-3           # channel width x depth
T_WALL = 1.0e-3                       # hot-wall thickness (assumed, row 205)
T_BACK = 1.0e-3                       # wall behind the channels [verify]
T_W_IN = 293.15                       # K water inlet (row 206)
P_OUT = 400 * PSI                     # channel outlet pressure (row 198)
X_UNCOOLED = 3.0e-3                   # first 3 mm of each body has no channel

# --- heat flux cases (MW/m^2) ---
Q_AVG_BASIS, Q_PEAK_BASIS = 9.8e6, 16.7e6     # SSoB rows 185-186
FLUX_CASES = {
    "Basis peak (SSoB)":                    16.7e6,
    "Basis, unscaled UAH throat":           20.5e6,
    "Closest published: UAH 76 mm near inj": 23.6e6,   # GCH4/GOX, ~same diameter
    "UAH 25 mm near injector":              25.6e6,
    "President's case":                     31.0e6,
    "Upper literature bound":               40.0e6,
}

# --- injector, 36-triplet design (injector study, 9 Nov) ---
INJ = dict(N_EL=36, D_OX=0.75e-3, D_F=0.42e-3, L_OX=5.0e-3, D_FEED=1.6e-3,
           H_IMP=1.5e-3, THETA=math.radians(40))   # 40 deg kept; 50 deg fan rejected
INJ["A_OFF"] = INJ["H_IMP"] * math.tan(INJ["THETA"])   # methane exit offset ~1.26 mm
INJ_FLUX = [10e6, 16.7e6, 23.6e6, 31e6]
T_LOX, T_CH4 = 90.0, 112.0
P_OX_G, P_F_G = 457 * PSI, 511 * PSI


# =============================================================================
# PART 1: CHANNEL SIZING
# =============================================================================
def water_props(T, P):
    g = lambda k: PropsSI(k, "T", T, "P", P, "Water")
    return g("D"), g("C"), g("V"), g("L")


def size_channels(name, side, D_hot, N, m_dot, M, q_avg, q_peak, t_wall=T_WALL,
                  quiet=False):
    A_hot = math.pi * D_hot * L_CH
    Q = q_avg * A_hot
    cp0 = PropsSI("C", "T", T_W_IN, "P", P_OUT, "Water")
    dT = Q / (m_dot * cp0)
    T_out = T_W_IN + dT
    rho, cp, mu, kf = water_props(0.5 * (T_W_IN + T_out), P_OUT)
    T_sat = PropsSI("T", "P", P_OUT, "Q", 0, "Water")
    sign = 1 if side == "outer" else -1
    pitch = math.pi * (D_hot + sign * 2 * (t_wall + H_CH / 2)) / N
    rib = pitch - W_CH
    A_ch = W_CH * H_CH
    Dh = 4 * A_ch / (2 * (W_CH + H_CH))
    v = m_dot / N / (rho * A_ch)
    Re = rho * v * Dh / mu
    h_c = 0.023 * Re**0.8 * (cp * mu / kf)**0.4 * kf / Dh
    T_wg = 600.0
    for _ in range(30):                               # iterate k(T)
        k = float(M["k"](0.5 * (T_wg + T_out)))
        mf = math.sqrt(2 * h_c / (k * rib))
        eta = math.tanh(mf * H_CH) / (mf * H_CH)
        q_cool = q_peak * (math.pi * D_hot / N) / (W_CH + 2 * H_CH * eta)
        T_wc = T_out + q_cool / h_c
        T_new = T_wc + q_peak * t_wall / k
        if abs(T_new - T_wg) < 0.05:
            break
        T_wg = T_new
    f = (0.790 * math.log(Re) - 1.64) ** -2
    dP = f * (L_CH / Dh) * rho * v**2 / 2
    res = dict(Q=Q, dT=dT, T_out=T_out, v=v, Re=Re, h_c=h_c, eta=eta, k=k,
               T_wc=T_wc, T_wg=T_wg, T_sat=T_sat, dP=dP, pitch=pitch, rib=rib,
               h_g=q_peak / (T_GAS - T_wg))
    if not quiet:
        ok = lambda T, lim: "PASS" if T <= lim else "FAIL"
        print(f"\n=== {name}, {M['name']} ===")
        print(f"Heat load / water rise   : {Q/1e3:.1f} kW / {dT:.1f} K  (q avg {q_avg/1e6:.1f}, peak {q_peak/1e6:.1f} MW/m^2)")
        print(f"Channels                 : {N} x {W_CH*1e3:.1f} x {H_CH*1e3:.1f} mm, land {rib*1e3:.2f} mm")
        print(f"Velocity / Re / h_c      : {v:.1f} m/s / {Re:,.0f} / {h_c:,.0f} W/m^2K (fin eff {eta:.2f})")
        print(f"Wall k at temperature    : {k:.0f} W/mK")
        print(f"Coolant-side wall        : {T_wc:.0f} K (boils at {T_sat:.0f} K, margin {T_sat-T_wc:.0f} K)")
        print(f"Hot-gas-side wall        : {T_wg:.0f} K -> {M['T_sus']:.0f} K {ok(T_wg, M['T_sus'])}, "
              f"{M['T_max']:.0f} K {ok(T_wg, M['T_max'])}")
        print(f"Channel dP               : {dP/1e5:.2f} bar | implied gas-side h {res['h_g']:,.0f} W/m^2K")
    return res


# =============================================================================
# PART 2: FIRST 3 mm + GASKET SEAT (2D wall slice, x = from face, y = into wall)
# =============================================================================
def wall_strip(body, M, q, x_start=X_UNCOOLED, t_end=None, contact_h=0.0,
               T_plate=250.0, dx=0.1e-3, X_len=15e-3):
    """Steady (t_end=None) or transient temperature field of the wall near the face.
    Channels smeared over the pitch: land conduction + water sink in the channel layer.
    End face x=0 adiabatic unless contact_h > 0 (contact to injector plate at T_plate)."""
    side, D, N, m = BODIES[body]
    ch = size_channels(body, side, D, N, m, M, Q_AVG_BASIS, Q_PEAK_BASIS, quiet=True)
    Y = T_WALL + H_CH + T_BACK
    x = np.arange(dx / 2, X_len, dx); y = np.arange(dx / 2, Y, dx)
    X, Yg = np.meshgrid(x, y, indexing="ij")
    in_ch = (X >= x_start) & (Yg > T_WALL) & (Yg < T_WALL + H_CH)
    frac = np.where(in_ch, ch["rib"] / ch["pitch"], 1.0)           # solid fraction
    G_w = np.where(in_ch, ch["h_c"] * (W_CH + 2 * H_CH) / (ch["pitch"] * H_CH) * dx * dx, 0.0)
    T_w = ch["T_out"]                                                # conservative: outlet water
    Qin = np.zeros_like(X); Qin[:, 0] = q * dx                       # per unit width
    Gc = np.zeros_like(X); Gc[0, :] = contact_h * dx                 # end-face contact

    def cond(T):
        k = M["k"](T) * frac
        kx = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :] + 1e-30)
        ky = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:] + 1e-30)
        return kx, ky                                                 # W/mK (dx/dx cancels)

    def flow(T):
        kx, ky = cond(T)
        F = Qin - G_w * (T - T_w) - Gc * (T - T_plate)
        fx = kx * np.diff(T, axis=0); F[:-1] += fx; F[1:] -= fx
        fy = ky * np.diff(T, axis=1); F[:, :-1] += fy; F[:, 1:] -= fy
        return F

    if t_end is None:                                                 # steady: sparse solve, k(T) iterate
        import scipy.sparse as sp, scipy.sparse.linalg as spla
        T = np.full_like(X, 500.0); n = X.size; idx = np.arange(n).reshape(X.shape)
        for _ in range(8):
            kx, ky = cond(T)
            r, c, v = [], [], []
            for K, a, b in ((kx, idx[:-1], idx[1:]), (ky, idx[:, :-1], idx[:, 1:])):
                a, b, K = a.ravel(), b.ravel(), K.ravel()
                r += [a, b, a, b]; c += [b, a, a, b]; v += [-K, -K, K, K]
            A = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n, n))
            A = A + sp.diags((G_w + Gc).ravel())
            rhs = (Qin + G_w * T_w + Gc * T_plate).ravel()
            T = spla.spsolve(A.tocsc(), rhs).reshape(X.shape)
    else:                                                             # transient from water temp
        T = np.full_like(X, T_W_IN)
        Cap = M["rho"] * M["cp"] * dx * dx * frac
        dt = 0.2 * dx * dx * M["rho"] * M["cp"] / float(M["k"](293))
        t = 0.0
        while t < t_end:
            T = T + dt * flow(T) / Cap; t += dt
    tip = T[X < x_start].max() if x_start > 0 else T[:, 0].max()
    first_ch = T[(X >= x_start) & (X < x_start + 0.5e-3) & (Yg >= T_WALL - dx) & (Yg <= T_WALL + dx)].max()
    gasket = T[0, Yg[0] > T_WALL + H_CH].mean()                       # end face, behind channel layer
    return dict(tip=tip, first_ch=first_ch, gasket=gasket, T_sat=ch["T_sat"], T=T, X=X, Y=Yg)


# =============================================================================
# PART 3: INJECTOR FACE
# =============================================================================
def h_tube(fluid, T, P, m_dot, n, d):
    rho = PropsSI("D", "T", T, "P", P, fluid); mu = PropsSI("V", "T", T, "P", P, fluid)
    k = PropsSI("L", "T", T, "P", P, fluid); cp = PropsSI("C", "T", T, "P", P, fluid)
    v = m_dot / (rho * n * math.pi / 4 * d**2)
    Re = rho * v * d / mu
    return 0.023 * Re**0.8 * (cp * mu / k)**0.4 * k / d, v


def injector_face(M, q, dr=0.25e-3):
    """Steady 2D axisymmetric plate model (reaches steady in ~3 s for copper; for
    316L the 7 s transient is also reported). Holes smeared into ring sinks."""
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    N = INJ["N_EL"]
    h_ox, v_ox = h_tube("Oxygen", T_LOX, P_OX_G, M_OX, N, INJ["D_OX"])
    h_fd, _ = h_tube("Oxygen", T_LOX, P_OX_G, M_OX, N, INJ["D_FEED"])
    h_f, v_f = h_tube("Methane", T_CH4, P_F_G, M_F, 2 * N, INJ["D_F"])
    r = np.arange(28e-3 + dr / 2, 56e-3, dr); z = np.arange(dr / 2, 18e-3, dr)
    R, Z = np.meshgrid(r, z, indexing="ij"); NR, NZ = R.shape
    box = lambda r1, r2, z1, z2: (R > r1) & (R < r2) & (Z > z1) & (Z < z2)
    fluid = box(38.6e-3, 41.4e-3, 10e-3, 18e-3) | box(30.5e-3, 34.5e-3, 7.6e-3, 18e-3) | box(45.5e-3, 49.5e-3, 7.6e-3, 18e-3)
    solid = ~fluid
    Gox = np.zeros_like(R); Gf = np.zeros_like(R)

    def path(G, r0, z0, r1, z1, gp, steps=3000):
        L = math.hypot(r1 - r0, z1 - z0)
        for s in (np.arange(steps) + 0.5) / steps:
            i = min(int((r0 + (r1 - r0) * s - 28e-3) / dr), NR - 1); j = min(int((z0 + (z1 - z0) * s) / dr), NZ - 1)
            G[i, j] += gp * L / steps
    path(Gox, 40e-3, 0, 40e-3, INJ["L_OX"], h_ox * math.pi * INJ["D_OX"] * N)
    path(Gox, 40e-3, INJ["L_OX"], 40e-3, 10e-3, h_fd * math.pi * INJ["D_FEED"] * N)
    a = INJ["A_OFF"]; dz = 7.6e-3; drf = dz * math.tan(INJ["THETA"])
    path(Gf, 40e-3 - a, 0, 40e-3 - a - drf, dz, h_f * math.pi * INJ["D_F"] * N)
    path(Gf, 40e-3 + a, 0, 40e-3 + a + drf, dz, h_f * math.pi * INJ["D_F"] * N)
    for g, b in ((Gox, box(38.6e-3, 41.4e-3, 10e-3, 18e-3)),
                 (Gf, box(30.5e-3, 34.5e-3, 7.6e-3, 18e-3) | box(45.5e-3, 49.5e-3, 7.6e-3, 18e-3))):
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            face = 2 * math.pi * R * (dr if ax == 1 else dr)
            g += np.where(solid & np.roll(b, sh, ax), 5000.0 * face, 0)
    Gox[~solid] = 0; Gf[~solid] = 0
    hot = (R[:, 0] > 37.5e-3) & (R[:, 0] < 42.5e-3)
    Qin = np.zeros_like(R); Qin[:, 0] = q * 2 * math.pi * R[:, 0] * dr * hot
    idx = -np.ones(R.shape, int); idx[solid] = np.arange(solid.sum())
    T = np.full_like(R, 300.0)
    for _ in range(6):
        k = np.where(solid, M["k"](T), 0)
        Kr = 2 * math.pi * (R[:-1] + dr / 2) * k[:-1] * k[1:] * 2 / (k[:-1] + k[1:] + 1e-30)
        Kz = 2 * math.pi * R[:, :-1] * k[:, :-1] * k[:, 1:] * 2 / (k[:, :-1] + k[:, 1:] + 1e-30)
        rr, cc, vv = [], [], []
        for K, A_, B_ in ((Kr, idx[:-1], idx[1:]), (Kz, idx[:, :-1], idx[:, 1:])):
            mk = (K > 0) & (A_ >= 0) & (B_ >= 0)
            i, j, kk = A_[mk], B_[mk], K[mk]
            rr += [i, j, i, j]; cc += [j, i, i, j]; vv += [-kk, -kk, kk, kk]
        A = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(solid.sum(),) * 2)
        A = A + sp.diags((Gox + Gf)[solid])
        b = (Qin + Gox * T_LOX + Gf * T_CH4)[solid]
        T = np.full_like(R, np.nan); T[solid] = spla.spsolve(A.tocsc(), b)
    Tface = np.nanmax(T)
    # discrete-hole penalty: half the element pitch spread through ~the LOX-hole depth
    s_half = math.pi * 2 * 40e-3 / N / 2
    pen = q * s_half**2 / (2 * float(M["k"](Tface)) * INJ["L_OX"])
    return dict(T_face=Tface, T_spot=Tface + pen, v_ox=v_ox, v_f=v_f,
                hole_ox=np.nanmax(np.where(Gox > 0, T, np.nan)))


# =============================================================================
# PART 4: MATERIAL TRADE
# =============================================================================
def run_material_comparison(materials):
    print(f"{'material':<12}{'k W/mK':>8}{'cooled wall @16.7':>19}{'limit':>8}{'wall verdict':>14}{'3 mm tip @23.6':>16}")
    for M in materials:
        rw = size_channels("Outer body", "outer", 85e-3, 104, 2.55, M, Q_AVG_BASIS, Q_PEAK_BASIS, quiet=True)
        tip = wall_strip("Outer body", M, 23.6e6)["tip"]
        v = "OK" if rw["T_wg"] <= M["T_sus"] else ("MARGINAL" if rw["T_wg"] <= M["T_max"] else "FAIL")
        print(f"{M['name']:<12}{float(M['k'](rw['T_wg'])):>8.0f}{rw['T_wg']:>17.0f} K{M['T_sus']:>7.0f}K{v:>14}{tip:>14.0f} K")


def run_injector_report(M):
    print(f"36 triplets: LOX {INJ['D_OX']*1e3:.2f} mm, CH4 {INJ['D_F']*1e3:.2f} mm, jets meet "
          f"{INJ['H_IMP']*1e3:.1f} mm up, CH4 exit offset {INJ['A_OFF']*1e3:.2f} mm")
    print(f"{'material':<10}{'q':>6}{'face (smeared)':>16}{'between holes':>15}{'limit':>8}{'verdict':>9}")
    mat = M["name"]
    for q in INJ_FLUX:
        r = injector_face(M, q)
        v = "OK" if r["T_spot"] <= M["T_sus"] else ("MARGINAL" if r["T_spot"] <= M["T_max"] else "FAIL")
        print(f"{mat:<10}{q/1e6:>6.1f}{r['T_face']:>14.0f} K{r['T_spot']:>13.0f} K{M['T_sus']:>7.0f}K{v:>9}")


def run_sections(sections):
    for title, report in sections:
        print("\n" + "#" * 72 + f"\n# {title}\n" + "#" * 72)
        report()


def run_channel_report(material):
    print(f"Chamber gas temperature: {T_GAS:.0f} K ({T_GAS_SRC})")
    tot = 0
    for name, (side, D, N, m) in BODIES.items():
        tot += size_channels(name, side, D, N, m, material, Q_AVG_BASIS, Q_PEAK_BASIS)["Q"]
    print(f"\nTotal chamber heat load (51 mm section): {tot/1e3:.0f} kW")


def run_near_face_report(material):
    print(f"{'Flux case':<38}{'q':>6}{'body':>12}{'3 mm tip':>10}{'gasket seat':>13}"
          f"{'1st ch wall':>12}{'boils?':>8}")
    for case, q in FLUX_CASES.items():
        for body in BODIES:
            r = wall_strip(body, material, q)
            boil = "yes" if r["first_ch"] > r["T_sat"] else "no"
            print(f"{case:<38}{q/1e6:>6.1f}{body:>12}{r['tip']-273.15:>8.0f} C{r['gasket']-273.15:>11.0f} C"
                  f"{r['first_ch']:>10.0f} K{boil:>8}")
    print("\nCheck: transient 7 s vs steady (outer body, 31 MW/m^2)")
    s = wall_strip("Outer body", material, 31e6)
    for te in (0.5, 1.0, 3.0, 7.0):
        r = wall_strip("Outer body", material, 31e6, t_end=te, dx=0.15e-3)
        print(f"  t = {te:>3.1f} s: tip {r['tip']-273.15:.0f} C, gasket {r['gasket']-273.15:.0f} C"
              f"   (steady: {s['tip']-273.15:.0f} / {s['gasket']-273.15:.0f} C)")
    print("\nWith contact to the cold injector plate (h = 5,000 W/m^2K to 250 K), 31 MW/m^2:")
    for body in BODIES:
        r = wall_strip(body, material, 31e6, contact_h=5000)
        print(f"  {body:<11}: tip {r['tip']-273.15:.0f} C, gasket {r['gasket']-273.15:.0f} C")
    print(f"\nChannel start distance sweep (outer body, {material['name']}, "
          f"tip limits {material['T_sus']:.0f} K / {material['T_max']:.0f} K):")
    print(f"{'q':>6}" + "".join(f"{xs:>9.1f}mm" for xs in (0.5, 1.0, 1.5, 2.0, 3.0)))
    for q in (16.7e6, 23.6e6, 31e6, 40e6):
        row = [wall_strip("Outer body", material, q, x_start=xs * 1e-3)["tip"] for xs in (0.5, 1.0, 1.5, 2.0, 3.0)]
        print(f"{q/1e6:>6.1f}" + "".join(f"{T:>9.0f} K" for T in row))


def run_chamber_checks(material):
    run_sections((
        ("PART 1: CHANNEL SIZING", partial(run_channel_report, material)),
        ("PART 2: FIRST 3 mm + GASKET SEAT", partial(run_near_face_report, material)),
        ("PART 3: OUTER BODY MATERIAL CHECK", partial(run_material_comparison, (material,))),
    ))
