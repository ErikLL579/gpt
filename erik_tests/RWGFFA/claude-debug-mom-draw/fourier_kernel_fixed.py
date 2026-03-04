"""
Corrected Fourier kernel with proper projector normalization.

The key fix: P^L must be normalized by |k̂|², not s².
The ε regulator only appears in the eigenvalues, not the projector.
"""
import gpt as g
import numpy as np


def compute_D_components_fixed(grid, M, eps):
    """
    Compute D_μν(k) with correct projector normalization.

    D has eigenvalues:
      - 1/s² for transverse modes (3 eigenvalues)
      - 1/M² for longitudinal mode (1 eigenvalue)

    where s² = |k̂|² + ε² and |k̂|² = Σ_μ sin²(k_μ/2).

    D = (1/s²) P^T + (1/M²) P^L
      = (1/s²) I + (1/M² - 1/s²) P^L

    where P^L = k̂ k̂† / |k̂|² is the TRUE projector (normalized by |k̂|², not s²).
    """
    nd = grid.nd
    L = grid.fdimensions

    tmp = g.complex(grid)
    coord_array = g.coordinates(tmp)
    n_sites = coord_array.shape[0]

    # Compute k/2
    k_half = np.zeros((n_sites, nd), dtype=np.float64)
    for mu in range(nd):
        n_mu = coord_array[:, mu].astype(np.float64)
        n_shifted = np.where(n_mu > L[mu] // 2, n_mu - L[mu], n_mu)
        k_half[:, mu] = n_shifted * (np.pi / L[mu])

    sin_k_half = np.sin(k_half)
    cos_k_half = np.cos(k_half)

    # |k̂|² = Σ sin²(k/2)
    k_hat_sq = np.sum(sin_k_half**2, axis=1)

    # s² = |k̂|² + ε²
    s2 = k_hat_sq + eps**2

    # k̂_μ = e^{-ik_μ/2} sin(k_μ/2)
    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

    # Eigenvalues
    inv_s2 = 1.0 / s2
    inv_M2 = 1.0 / (M**2)
    coeff = inv_M2 - inv_s2  # coefficient for P^L

    # Handle k=0 where |k̂|² = 0: P^L = 0 there
    # Use regularized denominator for P^L
    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    D_components = {}
    for mu in range(nd):
        for nu in range(nd):
            # P^L_{μν} = k̂_μ k̂*_ν / |k̂|²  (TRUE projector)
            PL_mu_nu = np.where(k_hat_sq > 1e-30,
                               k_hat[:, mu] * k_hat_conj[:, nu] / k_hat_sq_safe,
                               0.0)

            if mu == nu:
                D_val = inv_s2 + coeff * PL_mu_nu
            else:
                D_val = coeff * PL_mu_nu

            D_mu_nu = g.complex(grid)
            D_mu_nu[coord_array] = D_val.astype(np.complex128)
            D_components[(mu, nu)] = D_mu_nu

    return D_components


def compute_sqrt_Dinv_components_fixed(grid, M, eps):
    """
    Compute D^{-1/2}_μν(k) with correct projector normalization.

    D^{-1/2} has eigenvalues:
      - s for transverse modes
      - M for longitudinal mode

    D^{-1/2} = s P^T + M P^L = s I + (M - s) P^L

    where P^L = k̂ k̂† / |k̂|² is the TRUE projector.
    """
    nd = grid.nd
    L = grid.fdimensions

    tmp = g.complex(grid)
    coord_array = g.coordinates(tmp)
    n_sites = coord_array.shape[0]

    k_half = np.zeros((n_sites, nd), dtype=np.float64)
    for mu in range(nd):
        n_mu = coord_array[:, mu].astype(np.float64)
        n_shifted = np.where(n_mu > L[mu] // 2, n_mu - L[mu], n_mu)
        k_half[:, mu] = n_shifted * (np.pi / L[mu])

    sin_k_half = np.sin(k_half)
    cos_k_half = np.cos(k_half)

    k_hat_sq = np.sum(sin_k_half**2, axis=1)
    s2 = k_hat_sq + eps**2
    s = np.sqrt(s2)

    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

    coeff = M - s  # coefficient for P^L

    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    sqrtDinv_components = {}
    for mu in range(nd):
        for nu in range(nd):
            PL_mu_nu = np.where(k_hat_sq > 1e-30,
                               k_hat[:, mu] * k_hat_conj[:, nu] / k_hat_sq_safe,
                               0.0)

            if mu == nu:
                sqrtDinv_val = s + coeff * PL_mu_nu
            else:
                sqrtDinv_val = coeff * PL_mu_nu

            sqrtDinv_mu_nu = g.complex(grid)
            sqrtDinv_mu_nu[coord_array] = sqrtDinv_val.astype(np.complex128)
            sqrtDinv_components[(mu, nu)] = sqrtDinv_mu_nu

    return sqrtDinv_components


# Quick test
if __name__ == "__main__":
    size = 4
    grid = g.grid([size, size, size, size], g.double)
    nd = grid.nd
    V = grid.gsites

    M = 1.0
    eps = 0.5

    g.message(f"Testing FIXED kernels with M={M}, eps={eps}")

    D = compute_D_components_fixed(grid, M, eps)
    sqrtDinv = compute_sqrt_Dinv_components_fixed(grid, M, eps)

    # Check D * sqrtDinv * sqrtDinv = I
    g.message("\nComputing D * D^{-1/2} * D^{-1/2}...")

    result = {}
    for mu in range(nd):
        for nu in range(nd):
            tmp = g.lattice(D[(0, 0)])
            tmp[:] = 0
            for rho in range(nd):
                for sigma in range(nd):
                    tmp += g.eval(D[(mu, rho)] * sqrtDinv[(rho, sigma)] * sqrtDinv[(sigma, nu)])
            result[(mu, nu)] = tmp

    g.message("\nDiagonal elements (should be 1):")
    for mu in range(nd):
        val = g.sum(result[(mu, mu)]) / V
        g.message(f"  [{mu},{mu}]: avg = {val}")

    g.message("\nOff-diagonal elements (should be 0):")
    for mu in range(nd):
        for nu in range(nd):
            if mu != nu:
                val = g.sum(result[(mu, nu)]) / V
                g.message(f"  [{mu},{nu}]: avg = {val}")
