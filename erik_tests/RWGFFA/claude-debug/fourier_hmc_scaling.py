"""
Test dH scaling with n_steps for Fourier HMC.
"""
import gpt as g
import numpy as np
from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

grid = g.grid([4, 4, 4, 4], g.double)
V = grid.gsites
nd = grid.nd

M, eps, beta, tau = 1.0, 0.5, 6.0, 0.5
D_components = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)


def fill_fourier_momenta(rng, P):
    rng.normal_element(P)
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
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))
    return 0.5 * H_p.real


def velocity(P):
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    return [g.eval((1.0 / V) * (fft_inv * v_k[mu])) for mu in range(nd)]


U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)
wilson = g.qcd.gauge.action.wilson(beta)
sympl = g.algorithms.integrator.symplectic

# Use SAME initial conditions for all tests
rng_save = g.random("fixed-seed")
U_save = g.qcd.gauge.unit(grid)
P_save = g.group.cartesian(U_save)
fill_fourier_momenta(rng_save, P_save)

g.message("=== Using SAME initial P for all tests ===")

for n_steps in [4, 8, 16, 32, 64]:
    # Reset to saved state
    for mu in range(nd):
        U[mu] @= U_save[mu]
        P[mu] @= P_save[mu]

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: velocity(P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H_p0 = kinetic_energy(P)
    S_g0 = wilson(U)
    H0 = H_p0 + S_g0

    mdint(tau)

    H_p1 = kinetic_energy(P)
    S_g1 = wilson(U)
    H1 = H_p1 + S_g1

    dt = tau / n_steps
    dH_p = H_p1 - H_p0
    dS_g = S_g1 - S_g0
    dH = H1 - H0

    g.message(f"n={n_steps:3d}, dt={dt:.5f}: dH={dH:.6e}, dH_p={dH_p:.6e}, dS_g={dS_g:.6e}")
