"""
Compare kinetic energies from different momentum generation schemes.

1. Standard HMC: P ~ N(0,1), H_p = (1/2) tr(P†P)
2. Identity kernel: P = IFFT[I · FFT[η]], H_p = V * Σ_k tr(P†(k) I P(k))
3. Fourier kernel: P = IFFT[D^{-1/2} · FFT[η]], H_p = V * Σ_k tr(P†(k) D(k) P(k))

Check magnitudes and whether H_p has imaginary parts.
"""
import gpt as g
import numpy as np

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

# Grid setup
size = 4
grid = g.grid([size, size, size, size], g.double)
V = grid.gsites
nd = grid.nd

g.message(f"Grid: {size}^4, V={V}, nd={nd}")

rng = g.random("ke-test")

# Fourier kernel parameters
M = 1.0
eps_fourier = 0.5

g.message(f"Fourier kernel: M={M}, eps={eps_fourier}")

# Build identity kernel
g.message("\nBuilding identity kernel...")
D_identity = {}
sqrtDinv_identity = {}
for mu in range(nd):
    for nu in range(nd):
        D_mu_nu = g.complex(grid)
        if mu == nu:
            D_mu_nu[:] = 1.0
        else:
            D_mu_nu[:] = 0.0
        D_identity[(mu, nu)] = D_mu_nu
        sqrtDinv_identity[(mu, nu)] = g.copy(D_mu_nu)

# Build Fourier kernel
g.message("Building Fourier kernel...")
D_fourier = compute_D_components(grid, M, eps_fourier)
sqrtDinv_fourier = compute_sqrt_Dinv_components(grid, M, eps_fourier)


def generate_standard_momenta(rng, U):
    """Standard Gaussian momenta."""
    P = g.group.cartesian(U)
    rng.normal_element(P)
    return P


def generate_momenta_with_kernel(rng, U, sqrtDinv):
    """Generate momenta: P = IFFT[D^{-1/2} · FFT[η]]"""
    eta = g.group.cartesian(U)
    rng.normal_element(eta)

    fft = g.fft()
    eta_k = [g.eval(fft * eta[mu]) for mu in range(nd)]

    P_k = [g.lattice(grid, eta_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        P_k[mu][:] = 0
        for nu in range(nd):
            P_k[mu] += g.eval(sqrtDinv[(mu, nu)] * eta_k[nu])

    fft_inv = g.adj(fft)
    return [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]


def standard_kinetic_energy(P):
    """H_p = (1/2) Σ_x tr(P†P) - this is what mass_term computes."""
    H_p = 0.0
    for mu in range(nd):
        H_p += g.sum(g.trace(g.adj(P[mu]) * P[mu]))
    return H_p / 2.0  # mass_term has 1/2 factor


def fourier_kinetic_energy(P, D_components):
    """H_p = V * Σ_k tr(P†(k) D(k) P(k))"""
    fft = g.fft()
    P_k = [g.eval(fft * P[mu]) for mu in range(nd)]

    H_p = 0.0
    for mu in range(nd):
        for nu in range(nd):
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)
            H_p += g.sum(g.eval(D_components[(mu, nu)] * tr_PdagP))

    return V * H_p  # Return complex value, don't take real yet


# Reference gauge field
U = g.qcd.gauge.unit(grid)

# Also check with mass_term action
std_action = g.qcd.scalar.action.mass_term()

g.message("\n" + "="*70)
g.message("KINETIC ENERGY COMPARISON")
g.message("="*70)

n_samples = 5

g.message(f"\n--- Standard HMC momenta ({n_samples} samples) ---")
for i in range(n_samples):
    P = generate_standard_momenta(rng, U)

    H_std = standard_kinetic_energy(P)
    H_action = std_action(P)

    g.message(f"Sample {i+1}:")
    g.message(f"  H_p (manual)     = {H_std}")
    g.message(f"  H_p (mass_term)  = {H_action}")
    g.message(f"  Type: {type(H_std)}, real={H_std.real:.4f}, imag={H_std.imag:.2e}")


g.message(f"\n--- Identity kernel momenta ({n_samples} samples) ---")
for i in range(n_samples):
    P = generate_momenta_with_kernel(rng, U, sqrtDinv_identity)

    H_fourier = fourier_kinetic_energy(P, D_identity)
    H_std = standard_kinetic_energy(P)

    g.message(f"Sample {i+1}:")
    g.message(f"  H_p (Fourier formula) = {H_fourier}")
    g.message(f"  H_p (standard formula) = {H_std}")
    g.message(f"  Fourier: real={H_fourier.real:.4f}, imag={H_fourier.imag:.2e}")
    g.message(f"  Ratio (Fourier.real / std.real) = {H_fourier.real / H_std.real:.6f}")


g.message(f"\n--- Fourier kernel momenta ({n_samples} samples) ---")
for i in range(n_samples):
    P = generate_momenta_with_kernel(rng, U, sqrtDinv_fourier)

    H_fourier = fourier_kinetic_energy(P, D_fourier)
    H_std = standard_kinetic_energy(P)

    g.message(f"Sample {i+1}:")
    g.message(f"  H_p (Fourier formula with D) = {H_fourier}")
    g.message(f"  H_p (standard formula)       = {H_std}")
    g.message(f"  Fourier: real={H_fourier.real:.4f}, imag={H_fourier.imag:.2e}")
    g.message(f"  Ratio (Fourier.real / std.real) = {H_fourier.real / H_std.real:.6f}")


# Check the D kernel values using numpy slicing to avoid the scalar bug
g.message("\n" + "="*70)
g.message("D KERNEL DIAGNOSTICS")
g.message("="*70)

coord_origin = np.array([[0,0,0,0]], dtype=np.int32)
coord_k1 = np.array([[1,0,0,0]], dtype=np.int32)
coord_k2 = np.array([[2,0,0,0]], dtype=np.int32)
coord_negk1 = np.array([[size-1,0,0,0]], dtype=np.int32)

g.message("\nIdentity kernel D[(0,0)] at a few sites:")
g.message(f"  [0,0,0,0]: {D_identity[(0,0)][coord_origin][0]}")
g.message(f"  [1,0,0,0]: {D_identity[(0,0)][coord_k1][0]}")

g.message("\nFourier kernel D[(0,0)] at a few sites:")
g.message(f"  [0,0,0,0]: {D_fourier[(0,0)][coord_origin][0]}")
g.message(f"  [1,0,0,0]: {D_fourier[(0,0)][coord_k1][0]}")
g.message(f"  [2,0,0,0]: {D_fourier[(0,0)][coord_k2][0]}")

g.message("\nFourier kernel D[(0,1)] (off-diagonal) at a few sites:")
g.message(f"  [0,0,0,0]: {D_fourier[(0,1)][coord_origin][0]}")
g.message(f"  [1,0,0,0]: {D_fourier[(0,1)][coord_k1][0]}")

# Check if D is Hermitian: D(-k)* = D(k)
g.message("\nHermiticity check D(-k)* = D(k):")
D00_k1 = D_fourier[(0,0)][coord_k1][0]
D00_negk1 = D_fourier[(0,0)][coord_negk1][0]
g.message(f"  D[(0,0)] at k=(1,0,0,0): {D00_k1}")
g.message(f"  D[(0,0)] at k=(-1,0,0,0) = k=(3,0,0,0): {D00_negk1}")
g.message(f"  D(-k)*: {np.conj(D00_negk1)}")
g.message(f"  |D(-k)* - D(k)|: {abs(np.conj(D00_negk1) - D00_k1):.2e}")


# Check sqrtDinv values
g.message("\n" + "="*70)
g.message("D^{-1/2} KERNEL DIAGNOSTICS")
g.message("="*70)

g.message("\nIdentity sqrtDinv[(0,0)] at k=0:")
g.message(f"  [0,0,0,0]: {sqrtDinv_identity[(0,0)][coord_origin][0]}")

g.message("\nFourier sqrtDinv[(0,0)] at a few sites:")
g.message(f"  [0,0,0,0]: {sqrtDinv_fourier[(0,0)][coord_origin][0]}")
g.message(f"  [1,0,0,0]: {sqrtDinv_fourier[(0,0)][coord_k1][0]}")
g.message(f"  [2,0,0,0]: {sqrtDinv_fourier[(0,0)][coord_k2][0]}")

# Verify D^{-1/2} @ D^{-1/2} @ D = I at a point
g.message("\nVerify D^{-1/2} @ D^{-1/2} @ D = I at k=(1,0,0,0):")
# Build the 4x4 matrices at this k-point
D_mat = np.zeros((nd, nd), dtype=complex)
sqrtDinv_mat = np.zeros((nd, nd), dtype=complex)
for mu in range(nd):
    for nu in range(nd):
        D_mat[mu, nu] = D_fourier[(mu, nu)][coord_k1][0]
        sqrtDinv_mat[mu, nu] = sqrtDinv_fourier[(mu, nu)][coord_k1][0]

should_be_I = sqrtDinv_mat @ sqrtDinv_mat @ D_mat
g.message(f"  D^{{-1/2}} @ D^{{-1/2}} @ D =\n{should_be_I}")
g.message(f"  Max deviation from I: {np.max(np.abs(should_be_I - np.eye(nd))):.2e}")

# Summary comparison
g.message("\n" + "="*70)
g.message("SUMMARY: Factor of 2 analysis")
g.message("="*70)
g.message("Standard formula: H_p = (1/2) Σ_x tr(P†P)")
g.message("mass_term action: appears to compute Σ_x tr(P†P) without the 1/2")
g.message("Fourier formula:  H_p = V * Σ_k tr(P†(k) D(k) P(k))")
g.message("")
g.message("With D=I and Parseval: V * Σ_k tr(P†P) = Σ_x tr(P†P)")
g.message("So Fourier formula (D=I) should match mass_term (both without 1/2)")
g.message("")
g.message("For proper HMC, we need H_p = (1/2) P†DP, so divide Fourier formula by 2")
