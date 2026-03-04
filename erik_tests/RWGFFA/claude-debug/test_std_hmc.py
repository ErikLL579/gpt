"""Test standard HMC - matching working code pattern."""
import gpt as g

grid = g.grid([4, 4, 4, 4], g.double)
rng = g.random("test")
beta = 6.0
tau = 0.5

# Create fields once (like working code)
U = g.qcd.gauge.unit(grid)
P = g.group.cartesian(U)

wilson = g.qcd.gauge.action.wilson(beta)
std_action = g.qcd.scalar.action.mass_term()
sympl = g.algorithms.integrator.symplectic

g.message("=== Standard HMC ===")

for n_steps in [10, 20, 40]:
    # Reset U in-place (like working code)
    for mu in range(4):
        U[mu] @= g.qcd.gauge.unit(grid)[mu]

    # Fill P with Gaussian (like working code)
    rng.normal_element(P)

    # Create integrator
    ip = sympl.update_p(P, lambda: wilson.gradient(U, U))
    iq = sympl.update_q(U, lambda: std_action.gradient(P, P))
    mdint = sympl.leap_frog(n_steps, ip, iq)

    H0 = std_action(P) + wilson(U)
    mdint(tau)
    H1 = std_action(P) + wilson(U)

    dt = tau / n_steps
    g.message(f"n_steps={n_steps:3d}, dt={dt:.5f}: dH = {H1-H0:.6e}")

g.message("Success!")
