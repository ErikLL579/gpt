#!/usr/bin/env python3
#
# Nambu HMC Markov chain with topological charge in G (arXiv:2409.18958)
#
# PRURP trajectories with Metropolis accept/reject on
#   H(p,U,r) = sum p^2/2 + sum r^2/2 + S_beta(U)
# with auxiliary Hamiltonian
#   G(p,U,r) = c_p sum p^2/2 + c_r sum r^2/2 + kappa * Q(U),
# Q = clover-leaf topological charge, force via reverse-mode AD.
# p and r are drawn fresh from a unit Gaussian each trajectory.
#
import gpt as g
import numpy as np
import time
from gpt.ad import reverse as rad

# parameters
beta = g.default.get_float("--beta", 1.0)
L = g.default.get_int("--L", 4)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
kappa = g.default.get_float("--kappa", 10.0)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 6)
n_therm = g.default.get_int("--ntherm", 10)
n_traj = g.default.get_int("--ntraj", 50)
seed = g.default.get("--seed", "nambu-topo-markov")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Nambu HMC Markov chain with topological-charge G
  lattice = {grid.fdimensions}
  beta    = {beta}
  c_p     = {c_p}
  c_r     = {c_r}
  kappa   = {kappa}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm}, ntraj = {n_traj}
"""
)

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

nd = len(U)
P = g.group.cartesian(U)
R = g.group.cartesian(U)

a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

# topological charge as differentiable functional via reverse-mode AD
aU = [rad.node(g.copy(u)) for u in U]
q_func = g.qcd.gauge.differentiable_topology(aU).functional(*aU)


# component-wise product in the adjoint index:  (A, B) -> sum_a a_a b_a T_a
def cw_list(A, B):
    out = []
    for x, y in zip(A, B):
        cx = x.otype.coordinates(x)
        cy = y.otype.coordinates(y)
        o = g.lattice(x)
        x.otype.coordinates(o, [g(a * b) for a, b in zip(cx, cy)])
        out.append(o)
    return out


# PRURP: P(t/2) R(t/2) U(t) R(t/2) P(t/2) via nested leapfrog
def frc_P():
    F_S = a_S.gradient(U, U)
    F_Q = q_func.gradient(U, U)
    F = [g(c_r * fs - kappa * fq) for fs, fq in zip(F_S, F_Q)]
    return cw_list(F, R)


def frc_R():
    F_S = a_S.gradient(U, U)
    F_Q = q_func.gradient(U, U)
    F = [g(kappa * fq - c_p * fs) for fs, fq in zip(F_S, F_Q)]
    return cw_list(F, P)


def vel_U():
    return [g((c_r - c_p) * o) for o in cw_list(P, R)]


ip_P = sympl.update_p(P, frc_P, tag="P")
ip_R = sympl.update_p(R, frc_R, tag="R")
iq_U = sympl.update_q(U, vel_U, tag="U")
mdint = sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


def nambu_hmc(tau):
    rng.normal_element(P)
    rng.normal_element(R)
    accrej = metro(U)
    h0 = hamiltonian_H()
    mdint(tau)
    h1 = hamiltonian_H()
    return [accrej(h1, h0), h1 - h0]


# thermalization
g.message(f"Thermalizing {n_therm} trajectories")
t0 = time.time()
for i in range(n_therm):
    nambu_hmc(tau)
g.message(
    f"Thermalized in {time.time() - t0:.1f} s: plaquette = {g.qcd.gauge.plaquette(U):.6f}"
)

# production
history = []
plaq = []
t0 = time.time()
for i in range(n_traj):
    history += [nambu_hmc(tau)]
    plaq += [g.qcd.gauge.plaquette(U)]
    g.message(
        f"Trajectory {i}: plaquette = {plaq[-1]:.6f}, dH = {history[-1][1]:+.4e}, "
        f"accept = {int(history[-1][0])}, Q = {q_func(U):+.6f}"
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
