import gpt as g
import numpy as np

# Grid setup
size = 4
grid = g.grid([size, size, size, size], g.double)
V = grid.gsites
nd = grid.nd

rng = g.random('gf-test-v2')

beta = 6.0
tau = 1.0
n_md_steps = 10

print(f'Grid: {size}^4, beta={beta}')

# === Standard HMC thermalization ===
print('Running 100 HMC trajectories from cold start...')
U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)
wilson_action = g.qcd.gauge.action.wilson(beta)
sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

a0 = g.qcd.scalar.action.mass_term()

for it in range(100):
    accrej = metro(U)
    rng.normal_element(P)
    h0 = a0(P) + wilson_action(U)

    ip = sympl.update_p(P, lambda: wilson_action.gradient(U, U))
    iq = sympl.update_q(U, lambda: P)
    mdint = sympl.leap_frog(n_md_steps, ip, iq)
    mdint(tau)

    h1 = a0(P) + wilson_action(U)
    accepted = accrej(h1, h0)

    if it % 20 == 0:
        print(f'  [{it}] plaq = {g.qcd.gauge.plaquette(U):.6f}, dH = {h1-h0:.2e}')

print(f'Thermalized: plaq = {g.qcd.gauge.plaquette(U):.6f}')

# === Gauge Fixing Functions ===
def ftg(U, eps):
    B = U[0] - g.cshift(U[0], 0, -1)
    for mu in range(1, nd):
        B += U[mu] - g.cshift(U[mu], mu, -1)
    U_prime = []
    for mu in range(nd):
        U_mu_prime = g(
            g.matrix.exp(-eps * g.qcd.gauge.project.traceless_anti_hermitian(B))
            * U[mu]
            * g.matrix.exp(+eps * g.qcd.gauge.project.traceless_anti_hermitian(g.cshift(B, mu, +1)))
        )
        U_prime.append(U_mu_prime)
    return U_prime

def ft0(U):
    return ftg(U, eps=-5e-2)

fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic

# === Save original ===
U_original = g.copy(U)
plaq_original = g.qcd.gauge.plaquette(U)
print(f'\nOriginal plaquette: {plaq_original:.10f}')

# === Setup dft for inverse ===
dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,
    ft0,
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.optimize.non_linear_cg(
        maxiter=100, eps=1e-15, step=1e-1, line_search=ls2, beta=fr
    ),
)

# === 10 forward GF steps ===
print('\n=== 10 FORWARD gauge fixing steps ===')
for i in range(10):
    Z = ft0(U)
    for mu in range(nd):
        U[mu] @= Z[mu]
    print(f'  Step {i+1}: plaq = {g.qcd.gauge.plaquette(U):.10f}')

# === 10 reverse GF steps ===
print('\n=== 10 REVERSE gauge fixing steps ===')
for i in range(10):
    Z = dft.inverse(U)
    for mu in range(nd):
        U[mu] @= Z[mu]
    print(f'  Step {i+1}: plaq = {g.qcd.gauge.plaquette(U):.10f}')

# === Compare link by link ===
print(f'\n=== Result: link-by-link comparison ===')
for mu in range(nd):
    diff = g.eval(U[mu] - U_original[mu])
    diff_norm = g.sum(g.trace(g.adj(diff) * diff)).real
    orig_norm = g.sum(g.trace(g.adj(U_original[mu]) * U_original[mu])).real
    print(f'||U[{mu}] - U_orig[{mu}]|| = {np.sqrt(diff_norm):.6e}, relative = {np.sqrt(diff_norm/orig_norm):.6e}')
