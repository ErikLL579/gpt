"""
Fourier-Accelerated HMC Test (No Gauge Fixing)
Tests the corrected Fourier kernel implementation.
"""
import gpt as g
import numpy as np

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

# Grid setup - 4D
size = 4
grid = g.grid([size, size, size, size], g.double)
V = grid.gsites
nd = grid.nd

rng = g.random("fourier-hmc-test-nogf")

# Fourier kernel parameters
M = 1.0
eps_fourier = 0.5  # IR regulator for Fourier kernel

# Wilson action parameter
beta = 10.

# Trajectory parameters
tau = 0.8
n_md_steps = 12

g.message(f"Grid: {size}^4, V={V}, nd={nd}, M={M}, eps_fourier={eps_fourier}, beta={beta}")
g.message(f"tau={tau}, n_md_steps={n_md_steps}")

# Precompute Fourier kernels (using corrected functions)
g.message("Precomputing Fourier kernels (corrected projector normalization)...")
D_components = compute_D_components(grid, M, eps_fourier)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps_fourier)


# === Fourier HMC Functions ===

def fill_fourier_momenta(rng, P):
    """Fill P in-place with Fourier-accelerated momenta."""
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
    """H_p = V * Σ_k tr(P†(k) D(k) P(k))"""
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
    """v = IFFT[D·FFT[P]]"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])
    fft_inv = g.adj(fft)
    return [g.eval(fft_inv * v_k[mu]) for mu in range(nd)]


def trace_U(U):
    """Average link trace."""
    total = sum(g.sum(g.trace(U[mu])) for mu in range(nd))
    return total.real / (V * nd * 3.0)


# === Setup ===

# Initialize gauge field (start from unit, thermalize with HMC)
U = g.qcd.gauge.unit(grid)

# Conjugate momenta
P = g.group.cartesian(U)

# Actions
wilson_action = g.qcd.gauge.action.wilson(beta)

# Symplectic integrator components
sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

g.message("Starting Fourier HMC test (no gauge fixing, corrected kernel)...")

# === Main Loop ===
accept, total = 0, 0
plaq_measurements = []
trace_measurements = []
wf_measurements = []
n_trajectories = 20000

for it in range(n_trajectories):
    # Thermalization: no accept/reject for first 10 trajectories
    no_accept_reject = it < 10

    # Reset U to start of trajectory for metropolis
    accrej = metro(U)

    # Draw new momenta and compute initial H
    fill_fourier_momenta(rng, P)
    h0 = kinetic_energy(P) + wilson_action(U)

    # Create integrator fresh each trajectory
    ip = sympl.update_p(P, lambda: wilson_action.gradient(U, U))
    iq = sympl.update_q(U, lambda: velocity(P))
    mdint = sympl.leap_frog(n_md_steps, ip, iq)

    # Run MD trajectory
    mdint(tau)

    # Compute final H
    h1 = kinetic_energy(P) + wilson_action(U)
    dH = h1 - h0

    # Accept/reject
    if no_accept_reject:
        accepted = True
    else:
        accepted = accrej(h1, h0)

    accept += accepted
    total += 1

    plaq = g.qcd.gauge.plaquette(U)
    plaq_measurements.append(plaq)
    trace_measurements.append(trace_U(U))

    # Wilson flow measurement every 15 trajectories
    if (it % 15 == 0):
        U_wf = g.copy(U)
        for l in range(80):  # t = 4
            U_wf = g.qcd.gauge.smear.wilson_flow(U_wf, epsilon=0.05)
        wff = g.qcd.gauge.energy_density(U_wf)
        wf_measurements.append(wff)

    g.message(f"[{it}/{n_trajectories}] plaq = {plaq:.6f}, link_trace = {trace_U(U):.6f}, dH = {dH:.6e}, acc = {accept/total:.3f}")

# Save measurements
np.save("plaq_measurements_fourier_test_nogf.npy", np.asarray(plaq_measurements))
np.save("trace_measurements_fourier_test_nogf.npy", np.asarray(trace_measurements))
np.save("wf_measurements_fourier_test_nogf.npy", np.asarray(wf_measurements))

g.message(f"Final acceptance rate: {accept/total:.3f}")
g.message("Measurements saved to plaq_measurements_fourier_test_nogf.npy, etc.")
