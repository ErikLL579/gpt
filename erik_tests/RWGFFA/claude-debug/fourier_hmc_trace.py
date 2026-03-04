"""
Trace velocity calls during Fourier HMC.
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

velocity_call_count = [0]


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
    velocity_call_count[0] += 1
    P_norm = sum(g.sum(g.trace(g.adj(P[mu]) * P[mu])).real for mu in range(nd))

    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    v = [g.eval((1.0 / V) * (fft_inv * v_k[mu])) for mu in range(nd)]

    v_norm = sum(g.sum(g.trace(g.adj(v[mu]) * v[mu])).real for mu in range(nd))
    g.message(f"  velocity call {velocity_call_count[0]}: ||P||²={P_norm:.2f}, ||v||²={v_norm:.4f}")
    return v


U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)
wilson = g.qcd.gauge.action.wilson(beta)
sympl = g.algorithms.integrator.symplectic
rng = g.random("trace")

# Reset
for mu in range(nd):
    U[mu] @= g.qcd.gauge.unit(grid)[mu]
fill_fourier_momenta(rng, P)

n_steps = 4
velocity_call_count[0] = 0

ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
iq = sympl.update_q(U, lambda: velocity(P))
mdint = sympl.leap_frog(n_steps, ip, iq)

g.message(f"\n=== Starting integration with {n_steps} steps ===")
H0 = kinetic_energy(P) + wilson(U)
g.message(f"H0 = {H0:.6f}")

mdint(tau)

H1 = kinetic_energy(P) + wilson(U)
g.message(f"H1 = {H1:.6f}")
g.message(f"dH = {H1-H0:.6f}")
g.message(f"Total velocity calls: {velocity_call_count[0]}")
