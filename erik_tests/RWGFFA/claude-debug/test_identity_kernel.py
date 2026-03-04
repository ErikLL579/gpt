"""
Test with trivial kernel D_{μν}(k) = δ_{μν}.
Should reproduce standard HMC results.
"""
import gpt as g
import numpy as np

grid = g.grid([4, 4, 4, 4], g.double)
V = grid.gsites
nd = grid.nd

beta = 6.0
tau = 0.5

# Build trivial D = identity
D_components = {}
for mu in range(nd):
    for nu in range(nd):
        D_mu_nu = g.complex(grid)
        if mu == nu:
            D_mu_nu[:] = 1.0
        else:
            D_mu_nu[:] = 0.0
        D_components[(mu, nu)] = D_mu_nu


def kinetic_energy_fourier(P):
    """H_p = V * Σ_k tr(P†(k) D(k) P(k)) - scaled to match standard HMC"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))
    return V * H_p.real  # V factor to compensate for GPT's 1/V in FFT


def velocity_fourier(P):
    """v = IFFT[D·FFT[P]] - to match std_action.gradient which returns P"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    return [g.eval(fft_inv * v_k[mu]) for mu in range(nd)]  # no extra factors


U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)
wilson = g.qcd.gauge.action.wilson(beta)
std_action = g.qcd.scalar.action.mass_term()
sympl = g.algorithms.integrator.symplectic
rng = g.random("test-identity")

# Compare kinetic energies
rng.normal_element(P)
H_std = std_action(P)
H_fourier = kinetic_energy_fourier(P)

g.message(f"Standard H_p = {H_std:.6f}")
g.message(f"Fourier H_p (D=I) = {H_fourier:.6f}")
g.message(f"Ratio = {H_fourier / H_std:.6f}")
g.message(f"V = {V}, Ratio * V = {H_fourier / H_std * V:.6f}")

# Compare velocities
v_std = std_action.gradient(P, P)
v_fourier = velocity_fourier(P)

v_std_norm = sum(g.sum(g.trace(g.adj(v_std[mu]) * v_std[mu])).real for mu in range(nd))
v_fourier_norm = sum(g.sum(g.trace(g.adj(v_fourier[mu]) * v_fourier[mu])).real for mu in range(nd))

g.message(f"\nStandard ||v||² = {v_std_norm:.6f}")
g.message(f"Fourier ||v||² (D=I) = {v_fourier_norm:.6f}")
g.message(f"Ratio = {v_fourier_norm / v_std_norm:.6f}")

# Check if v_fourier = v_std / V
diff_norm = sum(g.sum(g.trace(g.adj(g.eval(v_fourier[mu] - v_std[mu]/V)) * g.eval(v_fourier[mu] - v_std[mu]/V))).real for mu in range(nd))
g.message(f"||v_fourier - v_std/V||² = {diff_norm:.2e}")

# Now run HMC with both
g.message("\n=== Standard HMC ===")
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

    g.message(f"n_steps={n_steps:3d}: dH = {H1-H0:.6e}")

g.message("\n=== Fourier HMC with D=I ===")
for n_steps in [10, 20, 40]:
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]
    rng.normal_element(P)

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: velocity_fourier(P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = kinetic_energy_fourier(P) + wilson(U)
    mdint(tau)
    H1 = kinetic_energy_fourier(P) + wilson(U)

    g.message(f"n_steps={n_steps:3d}: dH = {H1-H0:.6e}")
