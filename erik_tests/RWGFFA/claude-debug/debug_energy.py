"""Debug energy magnitudes."""
import sys
sys.path.insert(0, "/Users/ell579/Documents/Physics/lattice/gpt/erik_tests/RWGFFA")
import gpt as g
from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

grid = g.grid([4, 4, 4, 4], g.double)
U = g.qcd.gauge.unit(grid)
M = 1.0
eps = 0.01
nd = grid.nd
V = grid.gsites

D_components = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)

# Generate momenta
rng = g.random("debug")
eta_x = g.group.cartesian(U)
rng.normal_element(eta_x)

g.message("=== Raw Gaussian noise ===")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(eta_x[mu]) * eta_x[mu])).real
    g.message(f"||η_x[{mu}]||² = {norm:.2f}")

fft = g.fft()
eta_k = [g.eval(fft * eta_x[mu]) for mu in range(nd)]
g.message("\n=== After FFT ===")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(eta_k[mu]) * eta_k[mu])).real
    g.message(f"||η_k[{mu}]||² = {norm:.2f}")

P_k = [g.lattice(grid, eta_k[0].otype) for mu in range(nd)]
for mu in range(nd):
    P_k[mu][:] = 0
    for nu in range(nd):
        P_k[mu] += g.eval(sqrtDinv[(mu, nu)] * eta_k[nu])

g.message("\n=== After D^{-1/2} ===")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(P_k[mu]) * P_k[mu])).real
    g.message(f"||P_k[{mu}]||² = {norm:.2f}")

fft_inv = g.adj(fft)
pos_mom = [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]

g.message("\n=== Position space momenta ===")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(pos_mom[mu]) * pos_mom[mu])).real
    g.message(f"||P_x[{mu}]||² = {norm:.2f}")

# Kinetic energies
g.message("\n=== Kinetic energies ===")

# Standard
a0 = g.qcd.scalar.action.mass_term()
H_std = a0(pos_mom)
g.message(f"Standard H_p = (1/2) Σ tr(P†P) = {H_std:.2f}")

# Fourier with V factor
P_k2 = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]
H_fourier = 0.0
for mu in range(nd):
    for nu in range(nd):
        PdagP = g.eval(g.adj(P_k2[mu]) * P_k2[nu])
        tr_PdagP = g.trace(PdagP)
        contribution = g.eval(D_components[(mu, nu)] * tr_PdagP)
        H_fourier += g.sum(contribution)
H_fourier = H_fourier.real * V

g.message(f"Fourier H_p = V * Σ_k D tr(P†P) = {H_fourier:.2f}")

# Check velocity magnitude
v_k = [g.lattice(grid, P_k2[0].otype) for mu in range(nd)]
for mu in range(nd):
    v_k[mu][:] = 0
    for nu in range(nd):
        v_k[mu] += g.eval(D_components[(mu, nu)] * P_k2[nu])
velocity = [g.eval(2.0 * (fft_inv * v_k[mu])) for mu in range(nd)]

g.message("\n=== Velocity magnitudes ===")
for mu in range(nd):
    norm = g.sum(g.trace(g.adj(velocity[mu]) * velocity[mu])).real
    g.message(f"||v[{mu}]||² = {norm:.2f}")
