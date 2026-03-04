"""Debug velocity type issue."""
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

fft = g.fft()
eta_k = [g.eval(fft * eta_x[mu]) for mu in range(nd)]
P_k = [g.lattice(grid, eta_k[0].otype) for mu in range(nd)]
for mu in range(nd):
    P_k[mu][:] = 0
    for nu in range(nd):
        P_k[mu] += g.eval(sqrtDinv[(mu, nu)] * eta_k[nu])
fft_inv = g.adj(fft)
pos_mom = [g.eval(fft_inv * P_k[mu]) for mu in range(nd)]

g.message(f"pos_mom type: {pos_mom[0].otype}")
g.message(f"eta_x type: {eta_x[0].otype}")

# Compute velocity
P_k2 = [g.eval(fft * pos_mom[mu]) for mu in range(nd)]
g.message(f"P_k2 type: {P_k2[0].otype}")

v_k = [g.lattice(grid, P_k2[0].otype) for mu in range(nd)]
for mu in range(nd):
    v_k[mu][:] = 0
    for nu in range(nd):
        v_k[mu] += g.eval(D_components[(mu, nu)] * P_k2[nu])

g.message(f"v_k type: {v_k[0].otype}")

velocity = [g.eval(2.0 * (fft_inv * v_k[mu])) for mu in range(nd)]
g.message(f"velocity type: {velocity[0].otype}")

# Check what standard action gradient returns
a0 = g.qcd.scalar.action.mass_term()
std_velocity = a0.gradient(pos_mom, pos_mom)
g.message(f"standard velocity type: {std_velocity[0].otype}")
