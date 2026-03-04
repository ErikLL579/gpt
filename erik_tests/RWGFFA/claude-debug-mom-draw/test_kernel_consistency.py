"""
Test that D * sqrt(D^{-1}) * sqrt(D^{-1}) = I

This verifies the kernel construction is consistent.
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

# Fourier kernel parameters
M = 1.0
eps = 0.5

g.message(f"Fourier kernel parameters: M={M}, eps={eps}")

# Build kernels
g.message("\nBuilding D and D^{-1/2} kernels...")
D = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)

# Get all coordinates
coords = g.coordinates(D[(0,0)])
n_sites = coords.shape[0]

g.message(f"Total sites: {n_sites}")

# For each site, build the 4x4 matrices and check D * sqrtDinv * sqrtDinv = I
g.message("\nChecking D * D^{-1/2} * D^{-1/2} = I at each k-point...")

max_deviation = 0.0
worst_site = None

# Check a subset of sites for detailed output
check_sites = [
    np.array([[0, 0, 0, 0]], dtype=np.int32),  # k = 0
    np.array([[1, 0, 0, 0]], dtype=np.int32),  # k = (2π/L, 0, 0, 0)
    np.array([[2, 0, 0, 0]], dtype=np.int32),  # k = (4π/L, 0, 0, 0)
    np.array([[1, 1, 0, 0]], dtype=np.int32),  # k = (2π/L, 2π/L, 0, 0)
    np.array([[1, 1, 1, 1]], dtype=np.int32),  # k = (2π/L, 2π/L, 2π/L, 2π/L)
    np.array([[size-1, 0, 0, 0]], dtype=np.int32),  # k = -2π/L (negative k)
]

for site_coord in check_sites:
    # Build 4x4 matrices at this k-point
    D_mat = np.zeros((nd, nd), dtype=np.complex128)
    sqrtDinv_mat = np.zeros((nd, nd), dtype=np.complex128)

    for mu in range(nd):
        for nu in range(nd):
            D_mat[mu, nu] = D[(mu, nu)][site_coord][0]
            sqrtDinv_mat[mu, nu] = sqrtDinv[(mu, nu)][site_coord][0]

    # Compute D * sqrtDinv * sqrtDinv
    product = D_mat @ sqrtDinv_mat @ sqrtDinv_mat

    # Should be identity
    deviation = np.max(np.abs(product - np.eye(nd)))

    site_tuple = tuple(site_coord[0])
    g.message(f"\nk-point {site_tuple}:")
    g.message(f"  D diagonal: [{D_mat[0,0]:.6f}, {D_mat[1,1]:.6f}, {D_mat[2,2]:.6f}, {D_mat[3,3]:.6f}]")
    g.message(f"  D^{{-1/2}} diagonal: [{sqrtDinv_mat[0,0]:.6f}, {sqrtDinv_mat[1,1]:.6f}, {sqrtDinv_mat[2,2]:.6f}, {sqrtDinv_mat[3,3]:.6f}]")
    g.message(f"  D * D^{{-1/2}} * D^{{-1/2}} diagonal: [{product[0,0]:.6f}, {product[1,1]:.6f}, {product[2,2]:.6f}, {product[3,3]:.6f}]")
    g.message(f"  Max |D * D^{{-1/2}} * D^{{-1/2}} - I|: {deviation:.2e}")

    if deviation > max_deviation:
        max_deviation = deviation
        worst_site = site_tuple


# Now check ALL sites
g.message("\n" + "="*60)
g.message("Checking ALL k-points...")
g.message("="*60)

all_deviations = []

for i in range(n_sites):
    site_coord = coords[i:i+1]  # Keep as 2D array

    D_mat = np.zeros((nd, nd), dtype=np.complex128)
    sqrtDinv_mat = np.zeros((nd, nd), dtype=np.complex128)

    for mu in range(nd):
        for nu in range(nd):
            D_mat[mu, nu] = D[(mu, nu)][site_coord][0]
            sqrtDinv_mat[mu, nu] = sqrtDinv[(mu, nu)][site_coord][0]

    product = D_mat @ sqrtDinv_mat @ sqrtDinv_mat
    deviation = np.max(np.abs(product - np.eye(nd)))
    all_deviations.append(deviation)

    if deviation > max_deviation:
        max_deviation = deviation
        worst_site = tuple(coords[i])

all_deviations = np.array(all_deviations)

g.message(f"\nStatistics over all {n_sites} k-points:")
g.message(f"  Max deviation:  {np.max(all_deviations):.2e}")
g.message(f"  Mean deviation: {np.mean(all_deviations):.2e}")
g.message(f"  Min deviation:  {np.min(all_deviations):.2e}")
g.message(f"  Worst site:     {worst_site}")

if np.max(all_deviations) < 1e-10:
    g.message("\n✓ PASSED: D * D^{-1/2} * D^{-1/2} = I to machine precision")
else:
    g.message(f"\n✗ FAILED: Max deviation {np.max(all_deviations):.2e} exceeds tolerance")


# Also verify that D and sqrtDinv are Hermitian in the sense D(-k)* = D(k)
g.message("\n" + "="*60)
g.message("Checking Hermiticity: D(-k)* = D(k)")
g.message("="*60)

max_herm_deviation = 0.0

for i in range(n_sites):
    k = coords[i]
    neg_k = (-k) % size  # Wrap to [0, L)

    # Find index of -k
    neg_k_idx = None
    for j in range(n_sites):
        if np.all(coords[j] == neg_k):
            neg_k_idx = j
            break

    if neg_k_idx is None:
        continue

    site_k = coords[i:i+1]
    site_negk = coords[neg_k_idx:neg_k_idx+1]

    for mu in range(nd):
        for nu in range(nd):
            D_k = D[(mu, nu)][site_k][0]
            D_negk = D[(mu, nu)][site_negk][0]

            herm_dev = np.abs(np.conj(D_negk) - D_k)
            if herm_dev > max_herm_deviation:
                max_herm_deviation = herm_dev

g.message(f"Max |D(-k)* - D(k)|: {max_herm_deviation:.2e}")

if max_herm_deviation < 1e-10:
    g.message("✓ PASSED: D satisfies Hermiticity condition")
else:
    g.message(f"✗ FAILED: Hermiticity violated by {max_herm_deviation:.2e}")


# Check that D is symmetric: D_{μν}(k) = D_{νμ}(k)
g.message("\n" + "="*60)
g.message("Checking symmetry: D_{μν}(k) = D_{νμ}(k)")
g.message("="*60)

max_sym_deviation = 0.0

for i in range(n_sites):
    site_coord = coords[i:i+1]

    for mu in range(nd):
        for nu in range(mu+1, nd):
            D_mn = D[(mu, nu)][site_coord][0]
            D_nm = D[(nu, mu)][site_coord][0]

            sym_dev = np.abs(D_mn - D_nm)
            if sym_dev > max_sym_deviation:
                max_sym_deviation = sym_dev

g.message(f"Max |D_{{μν}} - D_{{νμ}}|: {max_sym_deviation:.2e}")

if max_sym_deviation < 1e-10:
    g.message("✓ PASSED: D is symmetric in Lorentz indices")
else:
    g.message(f"✗ FAILED: Symmetry violated by {max_sym_deviation:.2e}")
