"""
Simple Fourier-Accelerated HMC Test
"""
import gpt as g
import numpy as np
from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

grid = g.grid([4, 4, 4, 4], g.double)
V = grid.gsites
nd = grid.nd

# Parameters - use larger eps for stability
M = 1.0
eps = 0.5  # Larger eps = less aggressive acceleration
beta = 6.0
tau = 1.0

g.message(f"Grid: 4^4, V={V}, M={M}, eps={eps}, beta={beta}")

# Precompute kernels
D_components = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)

# Check D magnitude
D_sum = g.sum(g.adj(D_components[(0,0)]) * D_components[(0,0)]).real
g.message(f"||D_00||² = {D_sum:.2f}, avg per site = {D_sum/V:.4f}")


def generate_momenta(rng, U):
    """Generate Fourier-accelerated momenta P = IFFT[D^{-1/2} FFT[η]]"""
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
    return [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]


def kinetic_energy(P):
    """H_p = V * Σ_k tr(P†(k) D(k) P(k))"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))

    return H_p.real * V


def velocity(P):
    """v = 2 * IFFT[D · FFT[P]]"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    fft_inv = g.adj(fft)
    return [g.eval(2.0 * (fft_inv * v_k[mu])) for mu in range(nd)]


# Actions
wilson = g.qcd.gauge.action.wilson(beta)
U = g.qcd.gauge.unit(grid)
rng = g.random("test")
P = generate_momenta(rng, U)

# Check magnitudes
P_norm = sum(g.sum(g.trace(g.adj(P[mu]) * P[mu])).real for mu in range(nd))
v = velocity(P)
v_norm = sum(g.sum(g.trace(g.adj(v[mu]) * v[mu])).real for mu in range(nd))
g.message(f"||P||² = {P_norm:.2f}, ||v||² = {v_norm:.2f}, ratio = {v_norm/P_norm:.2f}")

# Standard HMC comparison
g.message("\n=== Standard HMC ===")
std_action = g.qcd.scalar.action.mass_term()
sympl = g.algorithms.integrator.symplectic

for n_steps in [10, 20, 40]:
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]
    rng.normal_element(P)

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: std_action.gradient(P, P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = std_action(P) + wilson(U)
    mdint(tau)
    H1 = std_action(P) + wilson(U)

    dt = tau / n_steps
    g.message(f"n_steps={n_steps:3d}, dt={dt:.4f}: dH = {H1-H0:.6e}")

g.message("Standard HMC completed!")

# Fourier HMC - try with smaller tau first
g.message("\n=== Fourier HMC ===")
tau_fourier = 0.1  # Smaller trajectory

for n_steps in [10, 20, 40, 80]:
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]
    P = generate_momenta(rng, U)

    # Check velocity before integration
    v = velocity(P)
    v_max = max(g.sum(g.trace(g.adj(v[mu]) * v[mu])).real for mu in range(nd))
    g.message(f"Before n_steps={n_steps}: max ||v_mu||² = {v_max:.2f}")

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: velocity(P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = kinetic_energy(P) + wilson(U)

    try:
        mdint(tau_fourier)
        H1 = kinetic_energy(P) + wilson(U)
        dt = tau_fourier / n_steps
        g.message(f"n_steps={n_steps:3d}, dt={dt:.4f}: dH = {H1-H0:.6e}")
    except Exception as e:
        g.message(f"n_steps={n_steps:3d}: FAILED - {type(e).__name__}")
        break
