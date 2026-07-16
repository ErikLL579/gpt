#!/usr/bin/env python3
#
# Nambu HMC with stout-smeared clover topological charge in G, with G
# LINEAR in r (arXiv:2409.18958 eq. 30 form): production run for the
# topological tunneling comparison (counterpart of hmc_topo_run.py and
# nambu_obc5d_linear_topo_run.py, same measurements and log format;
# analysis: tau_int.py). Scaling/conservation validated in
# nambu_smq_linear_scaling.py (July 16 2026: reversibility ~1e-15,
# dH ~ eps^1.98, dG ~ eps^1.99).
#
#   H = sum p^2/2 + sum r^2/2 + S_beta(U)      (Metropolis on H)
#   G = gamma * sum r_a + kappa * Q_sm(U)
#   Q_sm = clover Q (g.qcd.gauge.differentiable_topology) on n_sm
#          stout-smeared links (rho); gradient through the smearing via
#          differentiable_functional.transformed + stout jacobian
#
# Equations of motion (paper eqs. 16, 18 with dG/dr_a = gamma,
# dG/dp_a = 0, F_Q = e_a Q_sm):
#   Udot   = -sum_a (gamma p_a) T_a U        (standard-HMC-like velocity)
#   frc_P  = gamma * F_S - kappa * F_Q o r
#   frc_R  = kappa * F_Q o p
# gamma = 1, kappa = 0 reduces exactly to standard HMC.
#
# The topological drive enters ONLY through G, so the marginal for U is
# exactly exp(-S_beta(U)) for any kappa. Wilson-flowed Q and E measured
# every --nmeas trajectories; checkpoints every --nsave; auto-resume.
#
# Tune --nsteps so acceptance lands in 65-85%.
#
import gpt as g
import numpy as np
import time
from gpt.ad import reverse as rad
from topo_measure import flow_and_measure, measurement_line, latest_checkpoint, save_checkpoint

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 8)
gamma = g.default.get_float("--gamma", 1.0)
kappa = g.default.get_float("--kappa", 1.0)
rho = g.default.get_float("--rho", 0.12)
n_sm = g.default.get_int("--nsm", 2)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 50)
n_therm = g.default.get_int("--ntherm", 100)
n_traj = g.default.get_int("--ntraj", 100000)
n_meas = g.default.get_int("--nmeas", 1)
n_save = g.default.get_int("--nsave", 100)
t_flow = g.default.get_float("--tflow", 2.0)
eps_flow = g.default.get_float("--epsflow", 0.1)
root = g.default.get("--root", ".")
seed = g.default.get("--seed", "nambu-smq-topo")

grid = g.grid([L, L, L, L], g.double)

# resume from latest checkpoint if present
U = g.qcd.gauge.unit(grid)  # cold start at weak coupling
it0 = 0
latest = latest_checkpoint(root, n_traj)
if latest is not None:
    g.copy(U, g.load(f"{root}/ckpoint_lat.{latest}"))
    it0 = latest + 1
rng = g.random(f"{seed}-{it0}")

if it0 == 0:
    # kick infinitesimally off the exact identity: the stout jacobian in
    # the Q_sm force is 0/0-singular (Cayley-Hamilton coefficients) at
    # exactly unit links
    kick = g.group.cartesian(U)
    rng.normal_element(kick)
    for mu in range(len(U)):
        U[mu] @= g(g.matrix.exp(g(1e-2 * kick[mu])) * U[mu])

g.message(
    f"""
Nambu HMC with smeared-Q in G (linear in r): topological tunneling run
  lattice = {grid.fdimensions}
  beta    = {beta}
  gamma   = {gamma}, kappa = {kappa}
  rho     = {rho}, n_sm = {n_sm}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm} (skipped on resume), ntraj = {n_traj}
  nmeas   = {n_meas}, tflow = {t_flow} (eps_flow = {eps_flow}, radius = {np.sqrt(8 * t_flow):.2f})
  nsave   = {n_save}, root = {root}
  resume  = {"trajectory " + str(it0) if it0 > 0 else "fresh start"}
"""
)

nd = len(U)
P = g.group.cartesian(U)
R = g.group.cartesian(U)

a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

# smeared clover topological charge as differentiable functional:
# Q_sm(U) = Q_clover(stout^n_sm(U))
aU = [rad.node(g.copy(u)) for u in U]
q_func = g.qcd.gauge.differentiable_topology(aU).functional(*aU)
sm = g.qcd.gauge.smear.stout(rho=rho)
for _ in range(n_sm):
    q_func = q_func.transformed(sm)


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


# PRURP (paper eq. 19): nested GPT leapfrogs
def frc_P():
    F_S = a_S.gradient(U, U)
    F_Q = q_func.gradient(U, U)
    F_Qr = cw_list(F_Q, R)
    return [g(gamma * fs - kappa * fqr) for fs, fqr in zip(F_S, F_Qr)]


def frc_R():
    F_Q = q_func.gradient(U, U)
    return [g(kappa * fqp) for fqp in cw_list(F_Q, P)]


def vel_U():
    return [g(gamma * p) for p in P]


ip_P = sympl.update_p(P, frc_P, tag="P")
ip_R = sympl.update_p(R, frc_R, tag="R")
iq_U = sympl.update_q(U, vel_U, tag="U")
mdint = sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


def nambu_hmc(tau, accept_reject=True):
    rng.normal_element(P)
    rng.normal_element(R)
    if not accept_reject:
        # thermalization from cold start: integrate without Metropolis
        mdint(tau)
        return [1.0, 0.0]
    accrej = metro(U)
    h0 = hamiltonian_H()
    mdint(tau)
    h1 = hamiltonian_H()
    return [accrej(h1, h0), h1 - h0]


if it0 == 0:
    g.message(f"Thermalizing {n_therm} trajectories (no accept/reject)")
    t0 = time.time()
    for i in range(n_therm):
        nambu_hmc(tau, accept_reject=False)
    g.message(
        f"Thermalized in {time.time() - t0:.1f} s: plaquette = {g.qcd.gauge.plaquette(U):.6f}"
    )

history = []
n_acc = 0
t0 = time.time()
for i in range(it0, n_traj):
    history += [nambu_hmc(tau)]
    n_acc += history[-1][0]
    g.message(
        f"Trajectory {i}: plaquette = {g.qcd.gauge.plaquette(U):.6f}, "
        f"dH = {history[-1][1]:+.4e}, accept = {n_acc / len(history):.3f}, "
        f"Qsm = {q_func(U):+.4f}"
    )
    if i % n_meas == 0:
        m = flow_and_measure(U, t_flow, eps_flow)
        g.message(measurement_line(i, m))
    if i % n_save == 0:
        save_checkpoint(root, i, U)
    if len(history) % 100 == 0:
        h = np.array(history)
        g.message(
            f"Running stats after {len(history)} trajectories: "
            f"acceptance = {np.mean(h[:, 0]):.2f}, <|dH|> = {np.mean(np.abs(h[:, 1])):.4e}, "
            f"{(time.time() - t0) / len(history):.2f} s / trajectory"
        )
