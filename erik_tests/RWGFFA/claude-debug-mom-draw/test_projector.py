"""
Check if P^L is actually a projector: P^L * P^L = P^L
The issue might be the normalization when eps != 0.
"""
import gpt as g
import numpy as np

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

size = 4
grid = g.grid([size, size, size, size], g.double)
nd = grid.nd
V = grid.gsites

M = 1.0
eps = 0.5

g.message(f"Testing with M={M}, eps={eps}")

# Get coordinates
tmp = g.complex(grid)
coord_array = g.coordinates(tmp)
n_sites = coord_array.shape[0]
L = grid.fdimensions

# Recompute the projector P^L directly
k_half = np.zeros((n_sites, nd), dtype=np.float64)
for mu in range(nd):
    n_mu = coord_array[:, mu].astype(np.float64)
    n_shifted = np.where(n_mu > L[mu] // 2, n_mu - L[mu], n_mu)
    k_half[:, mu] = n_shifted * (np.pi / L[mu])

sin_k_half = np.sin(k_half)
cos_k_half = np.cos(k_half)
sin_k_half_sq = np.sum(sin_k_half**2, axis=1)  # |k_hat|^2

k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

s2 = sin_k_half_sq + eps**2
k_hat_sq = sin_k_half_sq  # |k_hat|^2 = sum of sin^2(k/2)

g.message(f"\nAt k=0: s² = {s2[0]}, |k̂|² = {k_hat_sq[0]}")
g.message(f"At k=(π/2,0,0,0): s² = {s2[1]}, |k̂|² = {k_hat_sq[1]}")

# Build P^L with the code's normalization (dividing by s²)
PL_code = {}
for mu in range(nd):
    for nu in range(nd):
        PL_val = k_hat[:, mu] * k_hat_conj[:, nu] / s2
        PL_field = g.complex(grid)
        PL_field[coord_array] = PL_val.astype(np.complex128)
        PL_code[(mu, nu)] = PL_field

# Build P^L with correct projector normalization (dividing by |k_hat|²)
# Need to handle k=0 where |k_hat|² = 0
k_hat_sq_reg = np.where(k_hat_sq > 1e-10, k_hat_sq, 1.0)  # Avoid div by zero
PL_correct = {}
for mu in range(nd):
    for nu in range(nd):
        PL_val = np.where(k_hat_sq > 1e-10,
                         k_hat[:, mu] * k_hat_conj[:, nu] / k_hat_sq_reg,
                         0.0)
        PL_field = g.complex(grid)
        PL_field[coord_array] = PL_val.astype(np.complex128)
        PL_correct[(mu, nu)] = PL_field

# Check if P^L_code is a projector: P^L * P^L = P^L
g.message("\n=== Check if P^L (code version, /s²) is a projector ===")
PL2_code = {}
for mu in range(nd):
    for nu in range(nd):
        tmp = g.lattice(PL_code[(0,0)])
        tmp[:] = 0
        for rho in range(nd):
            tmp += g.eval(PL_code[(mu, rho)] * PL_code[(rho, nu)])
        PL2_code[(mu, nu)] = tmp

# Compare P^L² to P^L
for mu in range(nd):
    diff = g.eval(PL2_code[(mu, mu)] - PL_code[(mu, mu)])
    rms = np.sqrt(g.sum(g.adj(diff) * diff).real / V)
    avg_PL = g.sum(PL_code[(mu, mu)]).real / V
    avg_PL2 = g.sum(PL2_code[(mu, mu)]).real / V
    g.message(f"  [{mu},{mu}]: avg P^L = {avg_PL:.6f}, avg (P^L)² = {avg_PL2:.6f}, ratio = {avg_PL2/avg_PL:.6f}")

# Check trace of P^L (should be 1 for a rank-1 projector)
tr_PL_code = g.lattice(PL_code[(0,0)])
tr_PL_code[:] = 0
for mu in range(nd):
    tr_PL_code += PL_code[(mu, mu)]
avg_tr = g.sum(tr_PL_code).real / V
g.message(f"\n  tr(P^L) average = {avg_tr:.6f} (should be 1 for projector)")

# Check with correct normalization
g.message("\n=== Check if P^L (correct version, /|k̂|²) is a projector ===")
PL2_correct = {}
for mu in range(nd):
    for nu in range(nd):
        tmp = g.lattice(PL_correct[(0,0)])
        tmp[:] = 0
        for rho in range(nd):
            tmp += g.eval(PL_correct[(mu, rho)] * PL_correct[(rho, nu)])
        PL2_correct[(mu, nu)] = tmp

for mu in range(nd):
    diff = g.eval(PL2_correct[(mu, mu)] - PL_correct[(mu, mu)])
    rms = np.sqrt(g.sum(g.adj(diff) * diff).real / V)
    avg_PL = g.sum(PL_correct[(mu, mu)]).real / V
    avg_PL2 = g.sum(PL2_correct[(mu, mu)]).real / V
    if abs(avg_PL) > 1e-10:
        g.message(f"  [{mu},{mu}]: avg P^L = {avg_PL:.6f}, avg (P^L)² = {avg_PL2:.6f}, ratio = {avg_PL2/avg_PL:.6f}")
    else:
        g.message(f"  [{mu},{mu}]: avg P^L = {avg_PL:.6f}, avg (P^L)² = {avg_PL2:.6f}")

tr_PL_correct = g.lattice(PL_correct[(0,0)])
tr_PL_correct[:] = 0
for mu in range(nd):
    tr_PL_correct += PL_correct[(mu, mu)]
avg_tr_correct = g.sum(tr_PL_correct).real / V
g.message(f"\n  tr(P^L) average = {avg_tr_correct:.6f} (should be 1 for projector)")

# The ratio |k̂|²/s² tells us the projector "leakage"
ratio = k_hat_sq / s2
g.message(f"\n=== Ratio |k̂|²/s² (projector scaling factor) ===")
g.message(f"  At k=0: {ratio[0]:.6f}")
g.message(f"  At k=(π/2,0,0,0): {ratio[1]:.6f}")
g.message(f"  Average: {np.mean(ratio):.6f}")
g.message(f"  Min: {np.min(ratio):.6f}, Max: {np.max(ratio):.6f}")
