import gpt as g
import numpy as np

# Import the Fourier kernel functions
from fourier_kernel import build_fourier_kernel, build_sqrt_Dinv_kernel


def compute_sqrt_Dinv_components(grid, M, eps):
    """
    Compute D^{-1/2}_μν(k) components directly as scalar fields.

    For sampling P from exp(-P† D P), we need P = D^{-1/2} · η where η ~ N(0,1).

    D^{-1/2} has eigenvalues:
      - s for transverse modes (3 eigenvalues)
      - M for longitudinal mode (1 eigenvalue)

    where s² = |k̂|² + ε² and |k̂|² = Σ_μ sin²(k_μ/2).

    D^{-1/2} = s P^T + M P^L = s I + (M - s) P^L

    IMPORTANT: P^L must be normalized by |k̂|², not s², to be a true projector.
    P^L_{μν} = k̂_μ k̂*_ν / |k̂|²

    Returns a dict: sqrtDinv_components[(mu, nu)] = complex lattice
    """
    nd = grid.nd
    L = grid.fdimensions

    # Get coordinates
    tmp = g.complex(grid)
    coord_array = g.coordinates(tmp)
    n_sites = coord_array.shape[0]

    # Compute momenta (symmetric Brillouin zone)
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
    s = np.sqrt(s2)

    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

    coeff = M - s  # coefficient for P^L term

    # Handle k=0 where |k̂|² = 0: P^L = 0 there
    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    # Build D^{-1/2} components as separate complex fields
    # D^{-1/2}_{μν} = s · δ_{μν} + (M - s) · P^L_{μν}
    sqrtDinv_components = {}
    for mu in range(nd):
        for nu in range(nd):
            # P^L_{μν} = k̂_μ k̂*_ν / |k̂|²  (TRUE projector)
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


def generate_fourier_momenta(rng, grid, M, eps, U):
    """
    Generate Fourier-accelerated conjugate momenta.

    For sampling P from exp(-P† D P), we need P = D^{-1/2} · η where η ~ N(0,1).

    GPT uses a Hermitian basis for Lie algebra elements (P† = P, traceless).
    For Hermitian fields, FFT satisfies: P̃(-k) = P̃(k)†
    Combined with D(-k)* = D(k), the result is guaranteed to be Hermitian.

    Steps:
    1. Generate Gaussian noise η_ν(x) in position space (Hermitian, traceless)
    2. FFT to momentum space: η_ν(k) [satisfies η(-k) = η(k)†]
    3. Apply D^{-1/2}: P_μ(k) = Σ_ν D^{-1/2}_μν(k) · η_ν(k)
    4. FFT^{-1} to get P_μ(x) in position space (Hermitian by construction)

    Parameters:
        rng: GPT random number generator
        grid: lattice grid
        M: gauge fixing mass parameter
        eps: IR regulator
        U: gauge field (used to get the algebra structure)

    Returns:
        mom: list of nd momentum lattices (Hermitian, traceless Lie algebra elements)
    """
    nd = grid.nd

    # Get D^{-1/2} components as scalar fields
    sqrtDinv_components = compute_sqrt_Dinv_components(grid, M, eps)

    # Generate Gaussian noise in POSITION space (Hermitian, traceless)
    eta_x = g.group.cartesian(U)
    rng.normal_element(eta_x)

    # Debug: verify eta_x is Hermitian
    for nu in range(nd):
        norm_sq = g.sum(g.trace(g.adj(eta_x[nu]) * eta_x[nu])).real
        herm_check = g.eval(eta_x[nu] - g.adj(eta_x[nu]))
        herm_norm = g.sum(g.trace(g.adj(herm_check) * herm_check)).real
        g.message(f"η_x[{nu}]: ||η||² = {norm_sq:.4f}, ||η - η†||² = {herm_norm:.2e}")

    # FFT to momentum space: η_x → η_k
    # This ensures η_k satisfies η(-k) = η(k)†
    fft = g.fft()
    eta_k = [g.eval(fft * eta_x[nu]) for nu in range(nd)]

    # Contract: P_μ(k) = Σ_ν D^{-1/2}_μν(k) · η_ν(k)
    P_k = [g.lattice(grid, eta_k[0].otype) for mu in range(nd)]
    for mu in range(nd):
        P_k[mu][:] = 0
        for nu in range(nd):
            P_k[mu] += g.eval(sqrtDinv_components[(mu, nu)] * eta_k[nu])

    # Debug: check P_k magnitude
    for mu in range(nd):
        norm_sq = g.sum(g.trace(g.adj(P_k[mu]) * P_k[mu])).real
        g.message(f"After D^{{-1/2}} contraction: ||P_k[{mu}]||² = {norm_sq:.4f}")

    # FFT^{-1} to position space: P_μ(k) → P_μ(x)
    fft_inv = g.adj(fft)
    mom = [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]

    # Debug: verify result is Hermitian
    for mu in range(nd):
        norm_sq = g.sum(g.trace(g.adj(mom[mu]) * mom[mu])).real
        herm_check = g.eval(mom[mu] - g.adj(mom[mu]))
        herm_norm = g.sum(g.trace(g.adj(herm_check) * herm_check)).real
        tr_mom = g.eval(g.trace(mom[mu]))
        trace_norm = g.sum(g.adj(tr_mom) * tr_mom).real
        g.message(f"P[{mu}]: ||P||² = {norm_sq:.4f}, ||P - P†||² = {herm_norm:.2e}, ||tr(P)||² = {trace_norm:.2e}")

    return mom


def compute_D_components(grid, M, eps):
    """
    Compute D_μν(k) components directly as scalar fields.

    D has eigenvalues:
      - 1/s² for transverse modes (3 eigenvalues)
      - 1/M² for longitudinal mode (1 eigenvalue)

    where s² = |k̂|² + ε² and |k̂|² = Σ_μ sin²(k_μ/2).

    D = (1/s²) P^T + (1/M²) P^L = (1/s²) I + (1/M² - 1/s²) P^L

    IMPORTANT: P^L must be normalized by |k̂|², not s², to be a true projector.
    P^L_{μν} = k̂_μ k̂*_ν / |k̂|²

    Returns a dict: D_components[(mu, nu)] = complex lattice
    """
    nd = grid.nd
    L = grid.fdimensions

    # Get coordinates
    tmp = g.complex(grid)
    coord_array = g.coordinates(tmp)
    n_sites = coord_array.shape[0]

    # Compute momenta (symmetric Brillouin zone)
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

    k_hat = sin_k_half * cos_k_half - 1j * sin_k_half**2
    k_hat_conj = sin_k_half * cos_k_half + 1j * sin_k_half**2

    # Eigenvalues
    inv_s2 = 1.0 / s2
    inv_M2 = 1.0 / (M**2)
    coeff = inv_M2 - inv_s2  # coefficient for P^L

    # Handle k=0 where |k̂|² = 0: P^L = 0 there
    k_hat_sq_safe = np.where(k_hat_sq > 1e-30, k_hat_sq, 1.0)

    # Build D components as separate complex fields
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


def fourier_kinetic_energy(mom, M, eps, grid):
    """
    Compute the Fourier-accelerated kinetic energy (Eq. 3.1):

    H_p = Σ_k tr_color(P_μ(-k) D^μν(k) P_ν(k))
        = Σ_k D^μν(k) · tr_color(P_μ(k)^† P_ν(k))

    Parameters:
        mom: list of 4 momentum lattices (Lie algebra elements in position space)
        M: gauge fixing mass parameter
        eps: IR regulator
        grid: lattice grid

    Returns:
        H_p: kinetic energy (real number)
    """
    nd = grid.nd

    # Get D components as scalar fields
    D_components = compute_D_components(grid, M, eps)

    # FFT momenta to momentum space
    fft = g.fft()
    P_k = [g.eval(fft * mom[mu]) for mu in range(nd)]

    # Compute H_p = Σ_k Σ_{μ,ν} D^μν(k) · tr(P_μ(k)^† P_ν(k))
    H_p = 0.0

    for mu in range(nd):
        for nu in range(nd):
            # tr(P_μ(k)^† P_ν(k)) at each k
            PdagP = g.eval(g.adj(P_k[mu]) * P_k[nu])
            tr_PdagP = g.trace(PdagP)

            # Sum: D^μν(k) · tr(P_μ^† P_ν)
            contribution = g.eval(D_components[(mu, nu)] * tr_PdagP)

            # Sum over all k
            H_p += g.sum(contribution)

    # H_p should be real (imaginary part from numerical errors)
    return H_p.real


def test_fourier_hmc():
    """Test the Fourier HMC momentum generation and Hamiltonian."""

    size = 4
    grid = g.grid([size, size, size, size], g.double)
    rng = g.random("test-fourier-hmc")

    # Parameters
    M = 1.0
    eps = 1e-6

    g.message("Building Fourier kernel D...")
    D = build_fourier_kernel(grid, M, eps)

    # Create gauge field (just use unit config for testing)
    U = g.qcd.gauge.unit(grid)

    # Generate Fourier-accelerated momenta
    g.message("Generating Fourier-accelerated momenta...")
    mom = generate_fourier_momenta(rng, grid, M, eps, U)

    g.message(f"Momentum type: {mom[0].otype}")
    g.message(f"Momentum at origin [0]: {mom[0][0,0,0,0]}")
    g.message(f"Momentum at [1,0,0,0]: {mom[0][1,0,0,0]}")

    # Check norm of momenta
    for mu in range(4):
        norm_sq = g.sum(g.trace(g.adj(mom[mu]) * mom[mu])).real
        g.message(f"||P_{mu}||^2 = {norm_sq}")

    # Compute kinetic energy
    g.message("Computing Fourier kinetic energy...")
    H_p = fourier_kinetic_energy(mom, M, eps, grid)
    g.message(f"Fourier kinetic energy H_p = {H_p}")

    # Compare with standard kinetic energy
    a0 = g.qcd.scalar.action.mass_term()
    H_standard = a0(mom)
    g.message(f"Standard kinetic energy = {H_standard}")

    # The Fourier kinetic energy should be O(1) per degree of freedom
    # if the momenta are properly normalized
    ndof = size**4 * 4 * 8  # sites * directions * SU(3) generators
    g.message(f"H_p / ndof = {H_p / ndof}")

    g.message("Test completed.")

    return mom, D, H_p


if __name__ == "__main__":
    test_fourier_hmc()
