"""
Fourier-Accelerated HMC Test
"""

import gpt as g
import numpy as np

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components


def generate_fourier_momenta(rng, grid, M, eps, U):
    """
    Generate Fourier-accelerated momenta.

    1. Generate Gaussian noise η(x) in position space
    2. FFT to momentum space: η(x) → η(k)
    3. Apply P(k) = D^{-1/2}(k) · η(k)
    4. IFFT to position space: P(k) → P(x)

    Returns pos_mom: position space momenta
    """
    nd = grid.nd

    # 1. Generate Gaussian noise in position space
    eta_x = g.group.cartesian(U)
    rng.normal_element(eta_x)

    # 2. FFT to momentum space
    fft = g.fft()
    eta_k = [g.eval(fft * eta_x[mu]) for mu in range(nd)]

    # 3. Apply D^{-1/2}
    sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)
    P_k = [g.lattice(grid, eta_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        P_k[mu][:] = 0
        for nu in range(nd):
            P_k[mu] += g.eval(sqrtDinv[(mu, nu)] * eta_k[nu])

    # 4. IFFT to position space
    fft_inv = g.adj(fft)

    # GPT's IFFT has no extra factor (FFT has 1/V, IFFT has none)
    pos_mom = [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]

    return pos_mom


def compute_velocity(pos_mom, D_components, grid):
    """
    Compute velocity v = ∂H_p/∂P = (2/V) D · P.

    The factor of 2/V comes from:
    - 2 from the Hermitian structure: ∂(P†DP)/∂P = 2DP
    - 1/V from FFT normalization

    1. FFT P(x) → P(k)
    2. Apply v(k) = D(k) · P(k)
    3. IFFT v(k) → v(x)
    4. Scale by 2/V

    Returns velocity in position space.
    """
    nd = grid.nd
    V = grid.gsites
    fft = g.fft()

    # FFT to momentum space
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    # Apply D
    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    # IFFT to position space and scale by 2
    # With H_p = V * Σ_k D tr(P†P), gradient is v = 2 * IFFT[D·FFT[P]]
    fft_inv = g.adj(fft)
    velocity = [g.eval(2.0 * (fft_inv * v_k[mu])) for mu in range(nd)]

    return velocity


def fourier_kinetic_energy(pos_mom, D_components, grid):
    """
    Compute kinetic energy H_p = P†(k) D(k) P(k) in momentum space.
    """
    nd = grid.nd
    fft = g.fft()

    # FFT to momentum space
    P_k = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]

    # Compute H_p = Σ_{k,μ,ν} tr(P_μ(k)† D_{μν}(k) P_ν(k))
    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            contribution = g.eval(D_components[(mu, nu)] * tr_PdagP)
            H_p += g.sum(contribution)

    # GPT's FFT includes 1/V, so P(k) is smaller by 1/V
    # To get proper extensive H_p ~ O(V), multiply by V
    return H_p.real * grid.gsites


# === Main ===

# Parameters
grid_size = 4
beta = 6.0
M = 1.0
epss = 0.01  # Larger eps for testing without gauge fixing
tau = 0.5
n_steps = 20

g.message("=" * 50)
g.message("Fourier HMC Test")
g.message(f"Grid: {grid_size}^4, beta={beta}, M={M}, eps={epss}")
g.message(f"tau={tau}, n_steps={n_steps}, dt={tau/n_steps}")
g.message("=" * 50)

# Setup
grid = g.grid([grid_size] * 4, g.double)
rng = g.random("fourier-hmc-test")

U = g.qcd.gauge.unit(grid)

# Precompute D
D_components = compute_D_components(grid, M, epss)

# Generate position space momenta
g.message("\nGenerating momenta...")
pos_mom = generate_fourier_momenta(rng, grid, M, epss, U)

# Check momenta
for mu in range(4):
    norm = g.sum(g.trace(g.adj(pos_mom[mu]) * pos_mom[mu])).real
    g.message(f"  ||pos_mom[{mu}]||² = {norm:.4f}")

# Actions
wilson_action = g.qcd.gauge.action.wilson(beta)

def hamiltonian():
    H_p = fourier_kinetic_energy(pos_mom, D_components, grid)
    S_gauge = wilson_action(U)
    return H_p + S_gauge

# Symplectic integrator
sympl = g.algorithms.integrator.symplectic

# Momentum update: P -= dt * dS_gauge/dU
ip = sympl.update_p(pos_mom, lambda: wilson_action.gradient(U, U))

# Coordinate update: U = exp(dt * v) * U, where v = D·P
iq = sympl.update_q(U, lambda: compute_velocity(pos_mom, D_components, grid))

# Leapfrog integrator
mdint = sympl.leap_frog(n_steps, ip, iq)

# First test standard HMC for comparison
g.message("\n=== Standard HMC (for comparison) ===")
standard_action = g.qcd.scalar.action.mass_term()

def standard_hamiltonian():
    return standard_action(pos_mom) + wilson_action(U)

for n_steps in [10, 20, 40]:
    # Reset
    for mu in range(4):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]

    # Standard momenta
    rng.normal_element(pos_mom)

    #g.message(f"testing momenta value: steps = {n_steps} has  {pos_mom[0][0,0,0,0]} ")

    ip_std = sympl.update_p(pos_mom, lambda: wilson_action.gradient(U, U))
    iq_std = sympl.update_q(U, lambda: standard_action.gradient(pos_mom, pos_mom))
    mdint_std = sympl.leap_frog(n_steps, ip_std, iq_std)

    H0 = standard_hamiltonian()
    mdint_std(tau)
    H1 = standard_hamiltonian()
    dH = H1 - H0
    dt = tau / n_steps

    g.message(f"  n_steps={n_steps:3d}, dt={dt:.5f}: dH = {dH:.6e}")

# Now test Fourier HMC with SAME initial conditions
g.message("\n=== Fourier HMC ===")

# Generate momenta once and save
rng_fixed = g.random("fourier-fixed-seed")

rng_1 = g.random("1")
rng_2 = g.random("2")
rng_3 = g.random("3")
rng_4 = g.random("4")  

rng = [rng_1, rng_2, rng_3, rng_4]

#for mu in range(4):
#    U[mu] @= g.qcd.gauge.unit(grid)[mu]



U1 = g.qcd.gauge.unit(grid)
#rng_1.normal_element(U1)
"""
mom1 = g.group.cartesian(U1)
rng_1.normal_element(mom1)

a0 = g.qcd.scalar.action.mass_term()
a1 = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic
    
ip1 = sympl.update_p(mom1, lambda: a1.gradient(U1, U1))
iq1 = sympl.update_q(U1, lambda: a0.gradient(mom1, mom1))

g.message(f"mom initial = {mom1[0][0,0,0,0]}")
g.message(f"test gauge force: { wilson_action.gradient(U1, U1)[0][0,0,0,0]}")

g.message(f"initial gauge link U1[0][0,0,0,0] = { U1[0][0,0,0,0]}")

mdint1 = sympl.leap_frog(20, ip1, iq1)

mdint1(1.)

g.message(f"mom after update = {mom1[0][0,0,0,0]}")
g.message(f"test link after update U1[0][0,0,0,0] = { U1[0][0,0,0,0]}")
"""



#pos_mom_initial = generate_fourier_momenta(rng_fixed, grid, M, epss, U)

pos_mom = generate_fourier_momenta(rng[0], grid, M, epss, U1)
g.message(f"pos_mom initial = {pos_mom[0][0,0,0,0]}")
g.message(f"test gauge force: { wilson_action.gradient(U1, U1)[0][0,0,0,0]}")

ip2 = sympl.update_p(pos_mom, lambda: wilson_action.gradient(U1, U1))
iq2 = sympl.update_q(U1, lambda: compute_velocity(pos_mom, D_components, grid))

g.message(f"initial gauge link U1[0][0,0,0,0] = { U1[0][0,0,0,0]}")
g.message(f"test velocity: {compute_velocity(pos_mom, D_components, grid)[0][0,0,0,0]}")

mdint2 = sympl.leap_frog(2, ip2, iq2)
mdint2(0.1)

g.message(f"=========================================================")
g.message(f"=========================================================")
g.message(f"=========================================================")

g.message(f"pos_mom after update = {pos_mom[0][0,0,0,0]}")
g.message(f"test link after update U1[0][0,0,0,0] = { U1[0][0,0,0,0]}")
g.message(f"test gauge force: { wilson_action.gradient(U1, U1)[0][0,0,0,0]}")
g.message(f"test velocity: {compute_velocity(pos_mom, D_components, grid)[0][0,0,0,0]}")


g.message(f"=========================================================")
g.message(f"=========================================================")
g.message(f"=========================================================")

mdint2 = sympl.leap_frog(2, ip2, iq2)
mdint2(0.1)

g.message(f"pos_mom after update = {pos_mom[0][0,0,0,0]}")
g.message(f"test link after update U1[0][0,0,0,0] = { U1[0][0,0,0,0]}")
g.message(f"test gauge force: { wilson_action.gradient(U1, U1)[0][0,0,0,0]}")
g.message(f"test velocity: {compute_velocity(pos_mom, D_components, grid)[0][0,0,0,0]}")


tau = 4.

for i, n_steps in enumerate([10, 20, 40, 80]):
    # Reset to same initial conditions
    for mu in range(4):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]
        #pos_mom[mu] @= pos_mom_initial[mu]

    pos_mom = generate_fourier_momenta(rng[i], grid, M, epss, U)

    #g.message(f"testing velocity values: steps = {n_steps} yeilds v[0][0,0,0,0] {compute_velocity(pos_mom, D_components, grid)[0][0,0,0,0]}")

    ip = sympl.update_p(pos_mom, lambda: wilson_action.gradient(U, U))
    iq = sympl.update_q(U, lambda: compute_velocity(pos_mom, D_components, grid))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = hamiltonian()
    H_p0 = fourier_kinetic_energy(pos_mom, D_components, grid)
    S_g0 = wilson_action(U)

    mdint(tau)

    H1 = hamiltonian()
    H_p1 = fourier_kinetic_energy(pos_mom, D_components, grid)
    S_g1 = wilson_action(U)
    dH = H1 - H0
    dt = tau / n_steps

    g.message(f"  n_steps={n_steps:3d}: dH={dH:.6e}, dH_p={H_p1-H_p0:.6e}, dS_g={S_g1-S_g0:.6e}")
    #g.message(f" initial Fourier KE = {H_p0} , final FKE = {H_p1} "  )
    #g.message(f" initial wilson action = {S_g0}, final wilson action = {S_g1}")



#g.message(f"testing velocity values: v[0][0,0,0,0] {compute_velocity(pos_mom, D_components, grid)[0][0,0,0,0]}")

