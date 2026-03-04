"""
Test gradient with corrected normalizations.
H_p = V * Σ_k tr(P†(k) D(k) P(k))  (compensate for GPT's 1/V in FFT)
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
    """H_p = V * Σ_k tr(P†(k) D(k) P(k))"""
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]
    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            contribution = g.eval(D_components[(mu, nu)] * tr_PdagP)
            H_p += g.sum(contribution)
    return H_p.real * V


def compute_velocity(pos_mom, coeff):
    """v = coeff * IFFT[D · FFT[P]]"""
    fft = g.fft()
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    return [g.eval(coeff * (fft_inv * v_k[mu])) for mu in range(nd)]


# Generate momenta (no /V in IFFT)
rng = g.random("test-gradient-v3")
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
pos_mom = [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]

g.message(f"Volume V = {V}")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(pos_mom[mu]) * pos_mom[mu])).real
    g.message(f"||P[{mu}]||² = {norm:.4f}")

# Numerical gradient
eps_num = 1e-6
delta_P = g.group.cartesian(U)
rng.normal_element(delta_P)
delta_norm = sum(g.sum(g.trace(g.adj(delta_P[mu]) * delta_P[mu])).real for mu in range(nd))
for mu in range(nd):
    delta_P[mu] /= np.sqrt(delta_norm)

H0 = fourier_kinetic_energy(pos_mom)
P_plus = [g.eval(pos_mom[mu] + eps_num * delta_P[mu]) for mu in range(nd)]
P_minus = [g.eval(pos_mom[mu] - eps_num * delta_P[mu]) for mu in range(nd)]
dH_numeric = (fourier_kinetic_energy(P_plus) - fourier_kinetic_energy(P_minus)) / (2 * eps_num)

g.message(f"\nH(P) = {H0:.2f}")
g.message(f"Numeric dH/dP · δP = {dH_numeric:.6f}")

# Test velocity formulas
for coeff_name, coeff in [("2", 2.0), ("2V", 2*V), ("2/V", 2/V), ("2/V²", 2/(V*V))]:
    v = compute_velocity(pos_mom, coeff)
    dH_analytic = sum(g.sum(g.trace(g.adj(v[mu]) * delta_P[mu])).real for mu in range(nd))
    ratio = dH_numeric / dH_analytic if abs(dH_analytic) > 1e-20 else float('inf')
    g.message(f"  coeff={coeff_name:6s}: v·δP = {dH_analytic:12.6f}, ratio = {ratio:.6f}")
