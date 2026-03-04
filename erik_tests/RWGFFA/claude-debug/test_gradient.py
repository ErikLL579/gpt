"""
Test that velocity = D·P is the correct gradient of H_p = P† D P
"""

import gpt as g
import numpy as np

from fourier_hmc import compute_D_components

grid = g.grid([4, 4, 4, 4], g.double)
U = g.qcd.gauge.unit(grid)
M = 1.0
eps = 0.5

D_components = compute_D_components(grid, M, eps)


def fourier_kinetic_energy(pos_mom):
    """H_p = P†(k) D(k) P(k)"""
    nd = 4
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            contribution = g.eval(D_components[(mu, nu)] * tr_PdagP)
            H_p += g.sum(contribution)

    return H_p.real


def compute_velocity(pos_mom):
    """v = (2/V) D · P"""
    nd = 4
    V = grid.gsites
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


# Generate random momenta
rng = g.random("test")
pos_mom = g.group.cartesian(U)
rng.normal_element(pos_mom)

# Compute velocity analytically
v_analytic = compute_velocity(pos_mom)

# Compute gradient numerically using directional derivative
# dH/dP · δP = lim_{ε→0} [H(P + ε δP) - H(P)] / ε
eps_num = 1e-6

# Random direction
delta_P = g.group.cartesian(U)
rng.normal_element(delta_P)

# Scale delta_P to have unit norm
delta_norm = sum(g.sum(g.trace(g.adj(delta_P[mu]) * delta_P[mu])).real for mu in range(4))
for mu in range(4):
    delta_P[mu] /= np.sqrt(delta_norm)

# Numerical directional derivative
H0 = fourier_kinetic_energy(pos_mom)

P_plus = [g.eval(pos_mom[mu] + eps_num * delta_P[mu]) for mu in range(4)]
H_plus = fourier_kinetic_energy(P_plus)

P_minus = [g.eval(pos_mom[mu] - eps_num * delta_P[mu]) for mu in range(4)]
H_minus = fourier_kinetic_energy(P_minus)

dH_numeric = (H_plus - H_minus) / (2 * eps_num)

# Analytic directional derivative: v · δP
dH_analytic = sum(g.sum(g.trace(g.adj(v_analytic[mu]) * delta_P[mu])).real for mu in range(4))

# Also try: 2 * Re(v · δP) in case there's a factor of 2
dH_analytic_2x = 2 * dH_analytic

V = grid.gsites  # Volume
g.message(f"Volume V = {V}")
g.message(f"H(P) = {H0:.6f}")
g.message(f"Numeric dH/dP · δP = {dH_numeric:.6f}")
g.message(f"Analytic v · δP = {dH_analytic:.6f}")
g.message(f"Analytic / V = {dH_analytic / V:.6f}")
g.message(f"Analytic / (V/2) = {dH_analytic / (V/2):.6f}")
g.message(f"2 * Analytic / V = {2 * dH_analytic / V:.6f}")
g.message(f"Ratio (numeric/analytic) = {dH_numeric / dH_analytic:.6f}")
