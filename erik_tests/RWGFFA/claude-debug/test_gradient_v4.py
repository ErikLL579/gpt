"""Test gradient with 1/2 factor."""
import gpt as g
import numpy as np
from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

grid = g.grid([4, 4, 4, 4], g.double)
V = grid.gsites
nd = grid.nd
M, eps = 1.0, 0.5

D_components = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)

U = g.qcd.gauge.unit(grid)
rng = g.random("grad-test")
P = g.group.cartesian(U)
rng.normal_element(P)

# Apply D^{-1/2}
fft = g.fft()
P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
tmp_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
for mu in range(nd):
    tmp_k[mu][:] = 0
    for nu in range(nd):
        tmp_k[mu] += g.eval(sqrtDinv[(mu, nu)] * P_k[nu])
fft_inv = g.adj(fft)
for mu in range(nd):
    P[mu] @= g.eval(fft_inv * tmp_k[mu])


def kinetic_energy(P):
    """H_p = (1/2) Σ_k tr(P†(k) D(k) P(k))"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))
    return 0.5 * H_p.real


def velocity(P, coeff):
    """v = coeff * IFFT[D·FFT[P]]"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    return [g.eval(coeff * (fft_inv * v_k[mu])) for mu in range(nd)]


# Numerical gradient
eps_num = 1e-5
delta_P = g.group.cartesian(U)
rng.normal_element(delta_P)
delta_norm = sum(g.sum(g.trace(g.adj(delta_P[mu]) * delta_P[mu])).real for mu in range(nd))
for mu in range(nd):
    delta_P[mu] /= np.sqrt(delta_norm)

P_plus = [g.eval(P[mu] + eps_num * delta_P[mu]) for mu in range(nd)]
P_minus = [g.eval(P[mu] - eps_num * delta_P[mu]) for mu in range(nd)]
dH_numeric = (kinetic_energy(P_plus) - kinetic_energy(P_minus)) / (2 * eps_num)

g.message(f"H_p = {kinetic_energy(P):.4f}")
g.message(f"Numeric dH/dP·δP = {dH_numeric:.6f}")

# Test different velocity coefficients
for name, coeff in [("1/V", 1/V), ("2/V", 2/V), ("1", 1.0), ("2", 2.0)]:
    v = velocity(P, coeff)
    dH_analytic = sum(g.sum(g.trace(g.adj(v[mu]) * delta_P[mu])).real for mu in range(nd))
    ratio = dH_numeric / dH_analytic if abs(dH_analytic) > 1e-15 else float('inf')
    g.message(f"coeff={name:4s}: v·δP = {dH_analytic:.6f}, ratio = {ratio:.4f}")
