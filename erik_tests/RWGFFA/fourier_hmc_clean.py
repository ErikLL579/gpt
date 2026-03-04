"""
Clean Fourier-Accelerated HMC Test
Using correct in-place field update pattern.
"""
import gpt as g
import numpy as np
from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

grid = g.grid([4, 4, 4, 4], g.double)
V = grid.gsites
nd = grid.nd

M = 1.0
eps = 0.5
beta = 6.0
tau = 0.5

g.message(f"Grid: 4^4, V={V}, M={M}, eps={eps}, beta={beta}, tau={tau}")

D_components = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)


def fill_fourier_momenta(rng, P):
    """Fill P in-place with Fourier-accelerated momenta."""
    # Generate Gaussian noise in position space
    rng.normal_element(P)

    # FFT to k-space
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    # Apply D^{-1/2}
    tmp_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        tmp_k[mu][:] = 0
        for nu in range(nd):
            tmp_k[mu] += g.eval(sqrtDinv[(mu, nu)] * P_k[nu])

    # IFFT back to position space, store in P
    fft_inv = g.adj(fft)
    for mu in range(nd):
        P[mu] @= g.eval(fft_inv * tmp_k[mu])


def kinetic_energy(P):
    """H_p = V * Σ_k tr(P†(k) D(k) P(k)) - V compensates for GPT's 1/V in FFT"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))

    return V * H_p.real


def velocity(P):
    """v = IFFT[D·FFT[P]] - matches GPT's std_action.gradient convention"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    fft_inv = g.adj(fft)
    return [g.eval(fft_inv * v_k[mu]) for mu in range(nd)]


# Create fields ONCE
U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)

wilson = g.qcd.gauge.action.wilson(beta)
std_action = g.qcd.scalar.action.mass_term()
sympl = g.algorithms.integrator.symplectic
rng = g.random("hmc-test")

# Standard HMC
g.message("\n=== Standard HMC ===")

for n_steps in [10, 20, 40]:
    # Reset U in-place
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]
    # Fill P with Gaussian
    rng.normal_element(P)

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: std_action.gradient(P, P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = std_action(P) + wilson(U)
    mdint(tau)
    H1 = std_action(P) + wilson(U)

    dt = tau / n_steps
    g.message(f"n_steps={n_steps:3d}, dt={dt:.5f}: dH = {H1-H0:.6e}")

# Fourier HMC
g.message("\n=== Fourier HMC ===")

for n_steps in [10, 20, 40, 80]:
    # Reset U in-place
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]
    # Fill P with Fourier momenta
    fill_fourier_momenta(rng, P)

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: velocity(P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = kinetic_energy(P) + wilson(U)
    mdint(tau)
    H1 = kinetic_energy(P) + wilson(U)

    dt = tau / n_steps
    g.message(f"n_steps={n_steps:3d}, dt={dt:.5f}: dH = {H1-H0:.6e}")
