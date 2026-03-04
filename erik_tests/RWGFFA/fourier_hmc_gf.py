"""
Fourier-Accelerated HMC with 50/50 Gauge Fixing (4D)
Combines the working Fourier HMC with gauge fixing steps.
"""
import gpt as g
import numpy as np
import random

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

# Grid setup - 4D
size = 4
grid = g.grid([size, size, size, size], g.double)
V = grid.gsites
nd = grid.nd

rng = g.random("fourier-hmc-gf")

# Fourier kernel parameters
M = 1.0
eps_fourier = 0.5  # IR regulator for Fourier kernel

# Wilson action parameter
beta = 6.0

# Trajectory parameters
tau = 1.0
n_md_steps = 20

g.message(f"Grid: {size}^4, V={V}, nd={nd}, M={M}, eps_fourier={eps_fourier}, beta={beta}")
g.message(f"tau={tau}, n_md_steps={n_md_steps}")

# Precompute Fourier kernels
D_components = compute_D_components(grid, M, eps_fourier)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps_fourier)


# === Gauge Fixing Functions (adapted from gfft-no-jac.py for 4D) ===

def ftg(U, eps):
    """Gauge fixing transformation."""
    # dmuAmu = sum_mu (U_mu(x) - U_mu(x-mu))
    B = U[0] - g.cshift(U[0], 0, -1)
    for mu in range(1, nd):
        B += U[mu] - g.cshift(U[mu], mu, -1)

    # Apply gauge transformation to all directions
    U_prime = []
    for mu in range(nd):
        U_mu_prime = g(
            g.matrix.exp(-eps * g.qcd.gauge.project.traceless_anti_hermitian(B))
            * U[mu]
            * g.matrix.exp(+eps * g.qcd.gauge.project.traceless_anti_hermitian(g.cshift(B, mu, +1)))
        )
        U_prime.append(U_mu_prime)

    return U_prime


def ft0(U):
    """Forward gauge fixing step."""
    return ftg(U, eps=-5e-2)


# Setup differentiable field transformation for inverse
fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic


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

g.message("Starting Fourier HMC with 50/50 gauge fixing...")

# === Main Loop ===
accept, total = 0, 0
plaq_measurements = []
trace_measurements = []
n_gf_steps = 4

for it in range(100):
    # Thermalization: no accept/reject for first 10 trajectories
    no_accept_reject = it < 10

    # Reset U to start of trajectory for metropolis
    accrej = metro(U)

    # Draw new momenta and compute initial H
    fill_fourier_momenta(rng, P)
    h0 = kinetic_energy(P) + wilson_action(U)

    # Create integrator fresh each trajectory (like in fourier_hmc_clean.py)
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

    g.message(f"////////////// PRE GAUGE FIXING //////////////////////")
    g.message(f"link trace = {trace_U(U):.6f}")

    # 50/50 gauge fixing after thermalization
    if it > 50:
        # Setup dft for inverse (needs current U)
        dft = g.qcd.gauge.smear.differentiable_field_transformation(
            U,
            ft0,
            g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
            g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
            g.algorithms.optimize.non_linear_cg(
                maxiter=100, eps=1e-15, step=1e-1, line_search=ls2, beta=fr
            ),
        )

        if random.randint(0, 1) == 0:
            g.message("////////////// FORWARD GAUGE FIXING //////////////////////")
            for i in range(n_gf_steps):
                Z = ft0(U)
                for mu in range(nd):
                    U[mu] @= Z[mu]
                g.message(f"link trace = {trace_U(U):.6f}")

        else:
            g.message("////////////// REVERSE GAUGE FIXING //////////////////////")
            for i in range(n_gf_steps):
                Z = dft.inverse(U)
                for mu in range(nd):
                    U[mu] @= Z[mu]
                g.message(f"link trace = {trace_U(U):.6f}")

    g.message(f"Fourier HMC plaq = {plaq:.6f}, dH = {dH:.6e}, acceptance = {accept/total:.3f}")
    g.message(f"TRAJECTORY NUM {it}")

# Save measurements
np.save("plaq_measurements_fourier_gf.npy", np.asarray(plaq_measurements))
np.save("trace_measurements_fourier_gf.npy", np.asarray(trace_measurements))
