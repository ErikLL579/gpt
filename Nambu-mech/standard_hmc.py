#!/usr/bin/env python3
#
# Standard HMC baseline for the Nambu HMC correctness test.
# Same lattice, beta, trajectory length and measurement output as
# nambu_markov.py; plain leapfrog with H = sum p^2/2 + S_beta(U).
#
import gpt as g
import numpy as np
import time

# parameters
beta = g.default.get_float("--beta", 1.0)
L = g.default.get_int("--L", 4)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 20)
n_therm = g.default.get_int("--ntherm", 10)
n_traj = g.default.get_int("--ntraj", 50)
seed = g.default.get("--seed", "standard-hmc")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Standard HMC baseline
  lattice = {grid.fdimensions}
  beta    = {beta}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm}, ntraj = {n_traj}
"""
)

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

P = g.group.cartesian(U)

a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

ip = sympl.update_p(P, lambda: a_S.gradient(U, U))
iq = sympl.update_q(U, lambda: a_kin.gradient(P, P))
mdint = sympl.leap_frog(n_steps, ip, iq)


def hmc(tau):
    rng.normal_element(P)
    accrej = metro(U)
    h0 = a_kin(P) + a_S(U)
    mdint(tau)
    h1 = a_kin(P) + a_S(U)
    return [accrej(h1, h0), h1 - h0]


g.message(f"Thermalizing {n_therm} trajectories")
t0 = time.time()
for i in range(n_therm):
    hmc(tau)
g.message(
    f"Thermalized in {time.time() - t0:.1f} s: plaquette = {g.qcd.gauge.plaquette(U):.6f}"
)

history = []
plaq = []
t0 = time.time()
for i in range(n_traj):
    history += [hmc(tau)]
    plaq += [g.qcd.gauge.plaquette(U)]
    g.message(
        f"Trajectory {i}: plaquette = {plaq[-1]:.6f}, dH = {history[-1][1]:+.4e}, "
        f"accept = {int(history[-1][0])}"
    )
t1 = time.time()

history = np.array(history)
plaq = np.array(plaq)
g.message("=" * 70)
g.message(f"Trajectories        = {n_traj}")
g.message(f"Wall-clock time     = {t1 - t0:.1f} s total, {(t1 - t0) / n_traj:.2f} s / trajectory")
g.message(f"Acceptance rate     = {np.mean(history[:, 0]):.2f}")
g.message(f"<|dH|>              = {np.mean(np.abs(history[:, 1])):.4e}")
g.message(f"<plaquette>         = {np.mean(plaq):.6f} (naive err {np.std(plaq) / len(plaq) ** 0.5:.6f})")
g.message(f"<1 - plaquette>     = {1.0 - np.mean(plaq):.6f}  (paper Table I convention)")
