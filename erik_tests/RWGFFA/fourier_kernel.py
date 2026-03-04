import gpt as g
import numpy as np

def build_fourier_kernel(grid, M, eps=1e-8):
    """
    Build the Fourier acceleration kernel D_{mu,nu}(k) from Zhao (2018) Eqs. 3.2-3.4.

    D has eigenvalues:
      - 1/s² for transverse modes (3 eigenvalues)
      - 1/M² for longitudinal mode (1 eigenvalue)

    where s² = |k̂|² + ε² and |k̂|² = Σ_μ sin²(k_μ/2).

    D = (1/s²) P^T + (1/M²) P^L = (1/s²) I + (1/M² - 1/s²) P^L

    IMPORTANT: P^L must be normalized by |k̂|², not s², to be a true projector.
    P^L_{μν} = k̂_μ k̂*_ν / |k̂|²

    Parameters:
        grid: gpt grid object
        M: mass parameter for gauge fixing strength
        eps: IR regulator to avoid division by zero at k=0

    Returns:
        D: lattice of nd x nd complex matrices, one at each momentum site
    """

    # Get lattice dimensions
    L = grid.fdimensions
    nd = grid.nd  # number of dimensions (should be 4)

    # Create the 4x4 complex matrix lattice
    D = g.mcomplex(grid, nd)

    # Get all coordinates as numpy array: shape (n_sites, nd)
    coord_array = g.coordinates(D)
    n_sites = coord_array.shape[0]

    # Compute momenta: k_mu = 2*pi*n_mu / L_mu
    # Use symmetric Brillouin zone: k ∈ [-π, π)
    # Shift n > L/2 to n - L
    k_half = np.zeros((n_sites, nd), dtype=np.float64)
    for mu in range(nd):
        n_mu = coord_array[:, mu].astype(np.float64)
        n_shifted = np.where(n_mu > L[mu] // 2, n_mu - L[mu], n_mu)
        k_half[:, mu] = n_shifted * (np.pi / L[mu])

    # Compute sin(k_mu/2) and cos(k_mu/2)
    sin_k_half = np.sin(k_half)  # shape (n_sites, nd)
    cos_k_half = np.cos(k_half)  # shape (n_sites, nd)

    # |k̂|² = Σ sin²(k/2)
    k_hat_sq = np.sum(sin_k_half**2, axis=1)  # shape (n_sites,)

    # k_hat_mu = e^{-i k_mu/2} * sin(k_mu/2)
    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2  # shape (n_sites, nd)

    # k_hat_conj_nu = e^{+i k_nu/2} * sin(k_nu/2)
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2  # shape (n_sites, nd)

    # s² = |k̂|² + ε²
    s2 = k_hat_sq + eps**2  # shape (n_sites,)
    inv_s2 = 1.0 / s2  # shape (n_sites,)
    inv_M2 = 1.0 / (M**2)
    coeff = inv_M2 - inv_s2  # shape (n_sites,)

    # Handle k=0 where |k̂|² = 0: P^L = 0 there
    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    # Build the nd x nd matrix at each site
    D_array = np.zeros((n_sites, nd, nd), dtype=np.complex128)

    for mu in range(nd):
        for nu in range(nd):
            # P^L_{mu,nu} = k_hat[mu] * k_hat_conj[nu] / |k̂|²  (TRUE projector)
            PL_mu_nu = np.where(k_hat_sq > 1e-30,
                               k_hat[:, mu] * k_hat_conj[:, nu] / k_hat_sq_safe,
                               0.0)

            # D_{mu,nu} = delta_{mu,nu} * inv_s2 + PL_mu_nu * coeff
            if mu == nu:
                D_array[:, mu, nu] = inv_s2 + PL_mu_nu * coeff
            else:
                D_array[:, mu, nu] = PL_mu_nu * coeff

    # Now we need to set the D lattice values
    # D is a mcomplex(grid, nd) which is an nd x nd complex matrix at each site
    # We set it using coordinate indexing
    D[coord_array] = D_array.reshape(n_sites, nd * nd)

    return D


def build_sqrt_fourier_kernel(grid, M, eps=1e-8):
    """
    Build sqrt(D)_{mu,nu}(k) - the square root of the Fourier kernel.

    sqrt(D) has eigenvalues:
      - 1/s for transverse modes
      - 1/M for longitudinal mode

    sqrt(D) = (1/s) P^T + (1/M) P^L = (1/s) I + (1/M - 1/s) P^L

    IMPORTANT: P^L must be normalized by |k̂|², not s², to be a true projector.

    NOTE: For momentum generation from P ~ exp(-½ P† D P), we need D^{-1/2}, not sqrt(D).
    Use build_sqrt_Dinv_kernel() for momentum generation.
    """

    L = grid.fdimensions
    nd = grid.nd

    sqrtD = g.mcomplex(grid, nd)

    coord_array = g.coordinates(sqrtD)
    n_sites = coord_array.shape[0]

    # Use symmetric Brillouin zone: k ∈ [-π, π)
    k_half = np.zeros((n_sites, nd), dtype=np.float64)
    for mu in range(nd):
        n_mu = coord_array[:, mu].astype(np.float64)
        n_shifted = np.where(n_mu > L[mu] // 2, n_mu - L[mu], n_mu)
        k_half[:, mu] = n_shifted * (np.pi / L[mu])

    sin_k_half = np.sin(k_half)
    cos_k_half = np.cos(k_half)

    # |k̂|² = Σ sin²(k/2)
    k_hat_sq = np.sum(sin_k_half**2, axis=1)

    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

    s2 = k_hat_sq + eps**2
    inv_sqrt_s2 = 1.0 / np.sqrt(s2)
    inv_M = 1.0 / M
    coeff = inv_M - inv_sqrt_s2

    # Handle k=0 where |k̂|² = 0
    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    sqrtD_array = np.zeros((n_sites, nd, nd), dtype=np.complex128)

    for mu in range(nd):
        for nu in range(nd):
            # P^L_{μν} = k̂_μ k̂*_ν / |k̂|²  (TRUE projector)
            PL_mu_nu = np.where(k_hat_sq > 1e-30,
                               k_hat[:, mu] * k_hat_conj[:, nu] / k_hat_sq_safe,
                               0.0)
            if mu == nu:
                sqrtD_array[:, mu, nu] = inv_sqrt_s2 + PL_mu_nu * coeff
            else:
                sqrtD_array[:, mu, nu] = PL_mu_nu * coeff

    sqrtD[coord_array] = sqrtD_array.reshape(n_sites, nd * nd)

    return sqrtD


def build_sqrt_Dinv_kernel(grid, M, eps=1e-8):
    """
    Build D^{-1/2}_{mu,nu}(k) - the inverse square root of the Fourier kernel.

    Since D = (1/s²) P^T + (1/M²) P^L, and P^T, P^L are orthogonal projectors:
    D^{-1/2} = s P^T + M P^L = s δ_{mu,nu} + (M - s) P^L_{mu,nu}

    This is used for momentum generation: P = D^{-1/2} · η where η ~ N(0,1)
    gives P distributed as exp(-½ P† D P).

    Important symmetry: D^{-1/2}(-k)* = D^{-1/2}(k)
    This ensures P(x) is Hermitian when η(x) is Hermitian (GPT's Lie algebra basis).

    IMPORTANT: P^L must be normalized by |k̂|², not s², to be a true projector.
    P^L_{μν} = k̂_μ k̂*_ν / |k̂|²

    Parameters:
        grid: gpt grid object
        M: mass parameter for gauge fixing strength
        eps: IR regulator

    Returns:
        sqrtDinv: lattice of nd x nd complex matrices
    """
    L = grid.fdimensions
    nd = grid.nd

    sqrtDinv = g.mcomplex(grid, nd)

    coord_array = g.coordinates(sqrtDinv)
    n_sites = coord_array.shape[0]

    # Symmetric Brillouin zone: k ∈ [-π, π)
    k_half = np.zeros((n_sites, nd), dtype=np.float64)
    for mu in range(nd):
        n_mu = coord_array[:, mu].astype(np.float64)
        n_shifted = np.where(n_mu > L[mu] // 2, n_mu - L[mu], n_mu)
        k_half[:, mu] = n_shifted * (np.pi / L[mu])

    sin_k_half = np.sin(k_half)
    cos_k_half = np.cos(k_half)

    # |k̂|² = Σ sin²(k/2)
    k_hat_sq = np.sum(sin_k_half**2, axis=1)

    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

    # s² = |k̂|² + ε²
    s2 = k_hat_sq + eps**2
    s = np.sqrt(s2)
    coeff = M - s  # coefficient for P^L term

    # Handle k=0 where |k̂|² = 0: P^L = 0 there
    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    sqrtDinv_array = np.zeros((n_sites, nd, nd), dtype=np.complex128)

    for mu in range(nd):
        for nu in range(nd):
            # P^L_{μν} = k̂_μ k̂*_ν / |k̂|²  (TRUE projector)
            PL_mu_nu = np.where(k_hat_sq > 1e-30,
                               k_hat[:, mu] * k_hat_conj[:, nu] / k_hat_sq_safe,
                               0.0)
            if mu == nu:
                # D^{-1/2}_{mu,mu} = s + (M - s) * P^L_{mu,mu}
                sqrtDinv_array[:, mu, nu] = s + coeff * PL_mu_nu
            else:
                # D^{-1/2}_{mu,nu} = (M - s) * P^L_{mu,nu}
                sqrtDinv_array[:, mu, nu] = coeff * PL_mu_nu

    sqrtDinv[coord_array] = sqrtDinv_array.reshape(n_sites, nd * nd)

    return sqrtDinv


def test_fourier_kernel():
    """Test the Fourier kernel construction."""

    size = 4
    grid = g.grid([size, size, size, size], g.double)

    M = 1.0  # gauge fixing mass parameter
    eps = 1e-6  # IR regulator

    g.message(f"Building Fourier kernel on {size}^4 lattice with M={M}, eps={eps}")

    D = build_fourier_kernel(grid, M, eps)

    g.message(f"D lattice type: {D.otype}")

    # Check values at specific sites
    # At k=0 (site [0,0,0,0]): sin(k/2)^2 = 0, so s2 = eps^2
    # P^L = 0 (since k_hat = 0), so D = (1/eps^2) * I
    D_at_origin = D[0, 0, 0, 0]
    g.message(f"D at k=0:")
    g.message(f"{D_at_origin}")
    expected_diag = 1.0 / eps**2
    g.message(f"Expected diagonal: {expected_diag}")

    # At k=(2π/L, 0, 0, 0): k_half = (π/L, 0, 0, 0)
    D_at_k1 = D[1, 0, 0, 0]
    g.message(f"D at k=(2π/{size}, 0, 0, 0):")
    g.message(f"{D_at_k1}")

    # Also test sqrt(D)
    g.message("Building sqrt(D)...")
    sqrtD = build_sqrt_fourier_kernel(grid, M, eps)
    sqrtD_at_origin = sqrtD[0, 0, 0, 0]
    g.message(f"sqrt(D) at k=0:")
    g.message(f"{sqrtD_at_origin}")
    expected_sqrt_diag = 1.0 / eps
    g.message(f"Expected diagonal: {expected_sqrt_diag}")

    # Verify that sqrtD @ sqrtD ≈ D at a few points
    g.message("Verifying sqrtD @ sqrtD ≈ D...")
    nd = grid.nd
    sqrtD_k1 = sqrtD[1, 0, 0, 0].array  # use .array to get numpy array
    D_k1_check = sqrtD_k1 @ sqrtD_k1
    D_k1_actual = D_at_k1.array
    g.message(f"sqrtD @ sqrtD at k=(2π/{size}, 0, 0, 0):")
    g.message(f"{D_k1_check}")
    g.message(f"D at same point:")
    g.message(f"{D_k1_actual}")
    g.message(f"Difference (should be ~0):")
    g.message(f"{np.max(np.abs(D_k1_check - D_k1_actual))}")

    # Test D^{-1/2}
    g.message("\nBuilding D^{-1/2}...")
    sqrtDinv = build_sqrt_Dinv_kernel(grid, M, eps)
    sqrtDinv_at_origin = sqrtDinv[0, 0, 0, 0]
    g.message(f"D^{{-1/2}} at k=0 (diagonal should be eps={eps}):")
    g.message(f"{sqrtDinv_at_origin}")

    # Verify that sqrtDinv @ sqrtDinv ≈ D^{-1}
    g.message("Verifying D^{-1/2} @ D^{-1/2} @ D ≈ I...")
    sqrtDinv_k1 = sqrtDinv[1, 0, 0, 0].array
    Dinv_k1 = sqrtDinv_k1 @ sqrtDinv_k1
    should_be_identity = Dinv_k1 @ D_k1_actual
    g.message(f"D^{{-1/2}} @ D^{{-1/2}} @ D at k=(2π/{size}, 0, 0, 0):")
    g.message(f"{should_be_identity}")
    g.message(f"Difference from identity (should be ~0):")
    g.message(f"{np.max(np.abs(should_be_identity - np.eye(nd)))}")

    # Verify symmetry D(-k)* = D(k)
    g.message("\nVerifying D(-k)* = D(k) symmetry...")
    D_k1 = D[1, 0, 0, 0].array
    D_neg_k1 = D[size-1, 0, 0, 0].array  # -k corresponds to L-1 for k=1
    g.message(f"Max |D(-k)* - D(k)|: {np.max(np.abs(np.conj(D_neg_k1) - D_k1))}")

    g.message("\nTest completed.")

    return D, sqrtD, sqrtDinv


if __name__ == "__main__":
    test_fourier_kernel()
