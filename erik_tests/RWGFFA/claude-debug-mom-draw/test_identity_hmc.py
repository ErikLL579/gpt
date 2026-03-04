"""
Test HMC with identity kernel D_{μν}(k) = δ_{μν}.

This should reproduce standard HMC results exactly, verifying that the
Fourier machinery (FFT, momentum generation, velocity computation) is correct.

If this works but the full Fourier kernel doesn't, the issue is with D(k) itself.
"""
import gpt as g
import numpy as np

# Grid setup
size = 4
grid = g.grid([size, size, size, size], g.double)
V = grid.gsites
nd = grid.nd

g.message(f"Grid: {size}^4, V={V}, nd={nd}")

rng = g.random("identity-hmc-test")

# Wilson action parameter
beta = 6.0
tau = 1.0

# Build IDENTITY kernel: D_{μν}(k) = δ_{μν}
# This means D^{-1/2} = I as well
g.message("Building identity kernel D_{μν} = δ_{μν}...")

D_components = {}
sqrtDinv_components = {}
for mu in range(nd):
    for nu in range(nd):
        D_mu_nu = g.complex(grid)
        if mu == nu:
            D_mu_nu[:] = 1.0
        else:
            D_mu_nu[:] = 0.0
        D_components[(mu, nu)] = D_mu_nu
        sqrtDinv_components[(mu, nu)] = g.copy(D_mu_nu)  # sqrt(I) = I, I^{-1/2} = I


def fill_momenta_with_identity_kernel(rng, P):
    """
    Generate momenta using identity kernel: P = IFFT[D^{-1/2} · FFT[η]] = IFFT[FFT[η]] = η

    With D^{-1/2} = I, this should just return Gaussian noise.
    But we go through the full machinery to test it.
    """
    # Generate Gaussian noise
    rng.normal_element(P)

    # FFT to k-space
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    # Apply D^{-1/2} = I (this should be a no-op)
    tmp_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        tmp_k[mu][:] = 0
        for nu in range(nd):
            tmp_k[mu] += g.eval(sqrtDinv_components[(mu, nu)] * P_k[nu])

    # IFFT back to position space
    fft_inv = g.adj(fft)
    for mu in range(nd):
        P[mu] @= g.eval(fft_inv * tmp_k[mu])


def kinetic_energy_identity(P):
    """
    H_p = V * Σ_k tr(P†(k) D(k) P(k)) with D = I

    This should equal V * Σ_k tr(P†(k) P(k)) = V * (1/V)² * V * Σ_x tr(P†(x) P(x))
                     = (1/V) * Σ_x tr(P†P)

    But GPT's mass_term action computes (1/2) Σ_x tr(P†P), so we need to be careful.
    """
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))

    return V * H_p.real


def velocity_identity(P):
    """
    v = IFFT[D · FFT[P]] with D = I

    This should equal IFFT[FFT[P]] = P (since GPT's FFT is unitary: IFFT·FFT = I)
    """
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    v_k = [g.lattice(grid, P_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        v_k[mu][:] = 0
        for nu in range(nd):
            v_k[mu] += g.eval(D_components[(mu, nu)] * P_k[nu])

    fft_inv = g.adj(fft)

    # Result needs to be cast back to algebra type for symplectic integrator
    result = [g.eval(fft_inv * v_k[mu]) for mu in range(nd)]

    # Create properly-typed output in the same algebra as P
    v_out = g.group.cartesian(P)
    for mu in range(nd):
        v_out[mu] @= result[mu]

    return v_out


# Create fields
U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)

wilson = g.qcd.gauge.action.wilson(beta)
std_action = g.qcd.scalar.action.mass_term()
sympl = g.algorithms.integrator.symplectic

# === First, verify that our identity kernel machinery matches standard HMC ===
g.message("\n" + "="*60)
g.message("TEST 1: Verify identity kernel matches standard HMC")
g.message("="*60)

# Generate momenta and check that identity kernel gives same result as direct
rng.normal_element(P)
P_copy = g.copy(P)

# Standard kinetic energy
H_std = std_action(P)

# Identity kernel kinetic energy
H_identity = kinetic_energy_identity(P)

# Check velocity
v_std = std_action.gradient(P, P)
v_identity = velocity_identity(P)

g.message(f"\nKinetic energy comparison:")
g.message(f"  Standard:        H_p = {H_std:.6f}")
g.message(f"  Identity kernel: H_p = {H_identity:.6f}")
g.message(f"  Ratio:           {H_identity / H_std:.6f}")

v_std_norm = sum(g.sum(g.trace(g.adj(v_std[mu]) * v_std[mu])).real for mu in range(nd))
v_identity_norm = sum(g.sum(g.trace(g.adj(v_identity[mu]) * v_identity[mu])).real for mu in range(nd))

g.message(f"\nVelocity comparison:")
g.message(f"  Standard:        ||v||² = {v_std_norm:.6f}")
g.message(f"  Identity kernel: ||v||² = {v_identity_norm:.6f}")
g.message(f"  Ratio:           {v_identity_norm / v_std_norm:.6f}")

# Check if velocities are proportional
diff_norm = sum(g.sum(g.trace(g.adj(g.eval(v_identity[mu] - v_std[mu])) * g.eval(v_identity[mu] - v_std[mu]))).real for mu in range(nd))
g.message(f"  ||v_identity - v_std||² = {diff_norm:.2e}")

# === TEST 2: Run HMC trajectories and compare dH scaling ===
g.message("\n" + "="*60)
g.message("TEST 2: Standard HMC - dH scaling")
g.message("="*60)

for n_steps in [10, 20, 40]:
    # Reset U
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]

    # Standard Gaussian momenta
    rng.normal_element(P)

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: std_action.gradient(P, P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = std_action(P) + wilson(U)
    mdint(tau)
    H1 = std_action(P) + wilson(U)

    dt = tau / n_steps
    g.message(f"n_steps={n_steps:3d}, dt={dt:.5f}: dH = {H1-H0:.6e}")


g.message("\n" + "="*60)
g.message("TEST 3: Identity kernel HMC - dH scaling")
g.message("="*60)

for n_steps in [10, 20, 40]:
    # Reset U
    for mu in range(nd):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]

    # Generate momenta through identity kernel machinery
    fill_momenta_with_identity_kernel(rng, P)

    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: velocity_identity(P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = kinetic_energy_identity(P) + wilson(U)
    mdint(tau)
    H1 = kinetic_energy_identity(P) + wilson(U)

    dt = tau / n_steps
    g.message(f"n_steps={n_steps:3d}, dt={dt:.5f}: dH = {H1-H0:.6e}")


# === TEST 4: Full HMC run and check plaquette ===
g.message("\n" + "="*60)
g.message("TEST 4: Full HMC thermalization - compare plaquette")
g.message("="*60)

n_therm = 20
n_traj = 50
n_md_steps = 20

# Standard HMC
g.message("\nStandard HMC thermalization...")
U_std = g.qcd.gauge.unit(grid)
P_std = g.group.cartesian(U_std)
metro_std = g.algorithms.markov.metropolis(rng)

plaq_std = []
accept_std = 0

for it in range(n_therm + n_traj):
    accrej = metro_std(U_std)
    rng.normal_element(P_std)

    ip = sympl.update_p(P_std, lambda: wilson.gradient(U_std, U_std))
    iq = sympl.update_q(U_std, lambda: std_action.gradient(P_std, P_std))
    mdint = sympl.leap_frog(n_md_steps, ip, iq)

    H0 = std_action(P_std) + wilson(U_std)
    mdint(tau)
    H1 = std_action(P_std) + wilson(U_std)

    if it < n_therm:
        accepted = True  # Always accept during thermalization
        for mu in range(nd):
            U_std[mu] @= U_std[mu]  # Keep the update
    else:
        accepted = accrej(H1, H0)
        accept_std += accepted

    plaq = g.qcd.gauge.plaquette(U_std)
    if it >= n_therm:
        plaq_std.append(plaq)

    if it % 10 == 0:
        g.message(f"  Traj {it}: plaq={plaq:.6f}, dH={H1-H0:.2e}")

g.message(f"\nStandard HMC results:")
g.message(f"  Acceptance: {accept_std}/{n_traj} = {accept_std/n_traj:.2f}")
g.message(f"  <plaq> = {np.mean(plaq_std):.6f} +/- {np.std(plaq_std)/np.sqrt(len(plaq_std)):.6f}")


# Identity kernel HMC
g.message("\nIdentity kernel HMC thermalization...")
U_id = g.qcd.gauge.unit(grid)
P_id = g.group.cartesian(U_id)
metro_id = g.algorithms.markov.metropolis(rng)

plaq_id = []
accept_id = 0

for it in range(n_therm + n_traj):
    accrej = metro_id(U_id)
    fill_momenta_with_identity_kernel(rng, P_id)

    ip = sympl.update_p(P_id, lambda: wilson.gradient(U_id, U_id))
    iq = sympl.update_q(U_id, lambda: velocity_identity(P_id))
    mdint = sympl.leap_frog(n_md_steps, ip, iq)

    H0 = kinetic_energy_identity(P_id) + wilson(U_id)
    mdint(tau)
    H1 = kinetic_energy_identity(P_id) + wilson(U_id)

    if it < n_therm:
        accepted = True
        for mu in range(nd):
            U_id[mu] @= U_id[mu]
    else:
        accepted = accrej(H1, H0)
        accept_id += accepted

    plaq = g.qcd.gauge.plaquette(U_id)
    if it >= n_therm:
        plaq_id.append(plaq)

    if it % 10 == 0:
        g.message(f"  Traj {it}: plaq={plaq:.6f}, dH={H1-H0:.2e}")

g.message(f"\nIdentity kernel HMC results:")
g.message(f"  Acceptance: {accept_id}/{n_traj} = {accept_id/n_traj:.2f}")
g.message(f"  <plaq> = {np.mean(plaq_id):.6f} +/- {np.std(plaq_id)/np.sqrt(len(plaq_id)):.6f}")


g.message("\n" + "="*60)
g.message("SUMMARY")
g.message("="*60)
g.message(f"Standard HMC:        <plaq> = {np.mean(plaq_std):.6f}")
g.message(f"Identity kernel HMC: <plaq> = {np.mean(plaq_id):.6f}")
g.message(f"Difference:          {abs(np.mean(plaq_std) - np.mean(plaq_id)):.6f}")
