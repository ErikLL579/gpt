"""
Check gauge field evolution in Fourier HMC.
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
rng = g.random("debug2")

fill_fourier_momenta(rng, P)

# Single leapfrog step manually to see what happens
dt = 0.1
n_half_steps = 5

g.message(f"=== Manual leapfrog, dt={dt} ===")

# Initial state
plaq0 = g.qcd.gauge.plaquette(U)
H_p0 = kinetic_energy(P)
S_g0 = wilson(U)
g.message(f"Initial: plaq={plaq0:.8f}, H_p={H_p0:.4f}, S_g={S_g0:.4f}")

# Half step for P
force = wilson.gradient(U, U)
for mu in range(nd):
    P[mu] -= (dt/2) * force[mu]
g.message(f"After P half-step: H_p={kinetic_energy(P):.4f}")

# Full steps
for step in range(n_half_steps):
    # Update U
    v = velocity(P)
    for mu in range(nd):
        U[mu] @= g.group.compose(dt * v[mu], U[mu])

    plaq = g.qcd.gauge.plaquette(U)
    S_g = wilson(U)
    g.message(f"After U step {step+1}: plaq={plaq:.8f}, S_g={S_g:.4f}")

    # Update P (except last step)
    if step < n_half_steps - 1:
        force = wilson.gradient(U, U)
        for mu in range(nd):
            P[mu] -= dt * force[mu]
        g.message(f"After P step {step+1}: H_p={kinetic_energy(P):.4f}")

# Final half step for P
force = wilson.gradient(U, U)
for mu in range(nd):
    P[mu] -= (dt/2) * force[mu]

H_p1 = kinetic_energy(P)
S_g1 = wilson(U)
plaq1 = g.qcd.gauge.plaquette(U)
g.message(f"Final: plaq={plaq1:.8f}, H_p={H_p1:.4f}, S_g={S_g1:.4f}")
g.message(f"dH = {(H_p1+S_g1)-(H_p0+S_g0):.6f}")
