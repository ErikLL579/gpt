"""
Fourier HMC with detailed diagnostics.
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
    return H_p.real


def velocity(P):
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    return [g.eval((2.0 / V) * (fft_inv * v_k[mu])) for mu in range(nd)]


U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)
wilson = g.qcd.gauge.action.wilson(beta)
sympl = g.algorithms.integrator.symplectic
rng = g.random("debug")

# Single trajectory with detailed output
for mu in range(nd):
    U[mu] @= g.qcd.gauge.unit(grid)[mu]
fill_fourier_momenta(rng, P)

n_steps = 20
ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
iq = sympl.update_q(U, lambda: velocity(P))
mdint = sympl.leap_frog(n_steps, ip, iq)

H_p0 = kinetic_energy(P)
S_g0 = wilson(U)
H0 = H_p0 + S_g0

g.message(f"Before: H_p={H_p0:.6f}, S_g={S_g0:.6f}, H={H0:.6f}")

# Check P and v magnitudes
P_norm = sum(g.sum(g.trace(g.adj(P[mu]) * P[mu])).real for mu in range(nd))
v = velocity(P)
v_norm = sum(g.sum(g.trace(g.adj(v[mu]) * v[mu])).real for mu in range(nd))
g.message(f"Before: ||P||²={P_norm:.2f}, ||v||²={v_norm:.2f}")

mdint(tau)

H_p1 = kinetic_energy(P)
S_g1 = wilson(U)
H1 = H_p1 + S_g1

g.message(f"After:  H_p={H_p1:.6f}, S_g={S_g1:.6f}, H={H1:.6f}")
g.message(f"Changes: dH_p={H_p1-H_p0:.6f}, dS_g={S_g1-S_g0:.6f}, dH={H1-H0:.6f}")

P_norm1 = sum(g.sum(g.trace(g.adj(P[mu]) * P[mu])).real for mu in range(nd))
v1 = velocity(P)
v_norm1 = sum(g.sum(g.trace(g.adj(v1[mu]) * v1[mu])).real for mu in range(nd))
g.message(f"After:  ||P||²={P_norm1:.2f}, ||v||²={v_norm1:.2f}")

# Now compare: what if we use standard kinetic energy with same P?
std_action = g.qcd.scalar.action.mass_term()
g.message(f"\nFor comparison, standard H_p = {std_action(P):.6f}")
