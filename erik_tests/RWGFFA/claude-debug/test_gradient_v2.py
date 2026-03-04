"""
Test gradient consistency with current normalization choices.
"""

import sys
sys.path.insert(0, "/Users/ell579/Documents/Physics/lattice/gpt/erik_tests/RWGFFA")
import gpt as g
import numpy as np

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

grid = g.grid([4, 4, 4, 4], g.double)
U = g.qcd.gauge.unit(grid)
M = 1.0
eps = 0.01
nd = grid.nd
V = grid.gsites

D_components = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)


def fourier_kinetic_energy(pos_mom):
    """H_p = (1/V) Σ_k tr(P†(k) D(k) P(k)) - matches fourier_hmc_test.py"""
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            contribution = g.eval(D_components[(mu, nu)] * tr_PdagP)
            H_p += g.sum(contribution)

    return H_p.real / V  # User's normalization


def compute_velocity_current(pos_mom):
    """v = (2/V) IFFT[D · FFT[P]] - current implementation"""
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    fft_inv = g.adj(fft)
    velocity = [g.eval((2.0 / V) * (fft_inv * v_k[mu])) for mu in range(nd)]
    return velocity


def compute_velocity_v2(pos_mom):
    """v = 2 IFFT[D · FFT[P]] - without extra 1/V"""
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    fft_inv = g.adj(fft)
    velocity = [g.eval(2.0 * (fft_inv * v_k[mu])) for mu in range(nd)]
    return velocity


def compute_velocity_v3(pos_mom):
    """v = (2/V^2) IFFT[D · FFT[P]] - with extra 1/V^2"""
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    fft_inv = g.adj(fft)
    velocity = [g.eval((2.0 / (V * V)) * (fft_inv * v_k[mu])) for mu in range(nd)]
    return velocity


# Generate momenta using same procedure as fourier_hmc_test.py
rng = g.random("test-gradient-v2")
eta_x = g.group.cartesian(U)
rng.normal_element(eta_x)

fft = g.fft()
eta_k = [g.eval(fft * eta_x[mu]) for mu in range(nd)]

P_k = [g.lattice(grid, eta_k[0].otype) for mu in range(nd)]
for mu in range(nd):
    P_k[mu][:] = 0
    for nu in range(nd):
        P_k[mu] += g.eval(sqrtDinv[(mu, nu)] * eta_k[nu])

fft_inv = g.adj(fft)
# With user's /V factor in IFFT
pos_mom = [g.eval(fft_inv * P_k[mu] / V) for mu in range(nd)]

g.message(f"Volume V = {V}")
g.message(f"Momentum norms:")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(pos_mom[mu]) * pos_mom[mu])).real
    g.message(f"  ||P[{mu}]||² = {norm:.6f}")

# Numerical gradient test
eps_num = 1e-6
delta_P = g.group.cartesian(U)
rng.normal_element(delta_P)

# Normalize
delta_norm = sum(g.sum(g.trace(g.adj(delta_P[mu]) * delta_P[mu])).real for mu in range(nd))
for mu in range(nd):
    delta_P[mu] /= np.sqrt(delta_norm)

H0 = fourier_kinetic_energy(pos_mom)
P_plus = [g.eval(pos_mom[mu] + eps_num * delta_P[mu]) for mu in range(nd)]
P_minus = [g.eval(pos_mom[mu] - eps_num * delta_P[mu]) for mu in range(nd)]
H_plus = fourier_kinetic_energy(P_plus)
H_minus = fourier_kinetic_energy(P_minus)
dH_numeric = (H_plus - H_minus) / (2 * eps_num)

g.message(f"\nH(P) = {H0:.6f}")
g.message(f"Numeric dH/dP · δP = {dH_numeric:.6f}")

# Test different velocity formulas
v_current = compute_velocity_current(pos_mom)
v_v2 = compute_velocity_v2(pos_mom)
v_v3 = compute_velocity_v3(pos_mom)

dH_current = sum(g.sum(g.trace(g.adj(v_current[mu]) * delta_P[mu])).real for mu in range(nd))
dH_v2 = sum(g.sum(g.trace(g.adj(v_v2[mu]) * delta_P[mu])).real for mu in range(nd))
dH_v3 = sum(g.sum(g.trace(g.adj(v_v3[mu]) * delta_P[mu])).real for mu in range(nd))

g.message(f"\nAnalytic v·δP (current 2/V): {dH_current:.6f}")
g.message(f"Analytic v·δP (v2, just 2): {dH_v2:.6f}")
g.message(f"Analytic v·δP (v3, 2/V²): {dH_v3:.6f}")

g.message(f"\nRatios (should be 1.0 for correct formula):")
g.message(f"  current (2/V):  {dH_numeric / dH_current:.6f}")
g.message(f"  v2 (2):         {dH_numeric / dH_v2:.6f}")
g.message(f"  v3 (2/V²):      {dH_numeric / dH_v3:.6f}")
