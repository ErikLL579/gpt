#!/usr/bin/env python3
#
# Nambu HMC with 5D OBC scaffold in G: production run for the
# topological tunneling comparison (counterpart of hmc_topo_run.py,
# same measurements and log format; analysis: tau_int.py).
#
#   H = sum_all p^2/2 + sum_all r^2/2 + S_beta(U)      (Metropolis on H)
#   G = c_p sum_all p^2/2 + c_r sum_all r^2/2 + kappa5 * S5(U, V, W)
#   S5 = sum_{x,mu} Re Tr [ U_mu(x) V5(x+mu) W_mu(x)^dag V5(x)^dag ]
#
# Edge layer W and verticals V5 carry no weight in H (Haar marginal,
# exact Gibbs refresh each trajectory) -> 4D marginal is exactly
# exp(-S_beta(U)) for any kappa5. Wilson-flowed Q and E measured every
# --nmeas trajectories; checkpoints every --nsave; auto-resume.
#
# Tune --nsteps so acceptance lands in 65-85%.
#
import gpt as g
import numpy as np
import time
import hashlib
from gpt.ad import reverse as rad
from topo_measure import flow_and_measure, measurement_line, latest_checkpoint, save_checkpoint

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 8)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
kappa5 = g.default.get_float("--kappa5", 1.0)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 50)
n_therm = g.default.get_int("--ntherm", 100)
n_traj = g.default.get_int("--ntraj", 100000)
n_meas = g.default.get_int("--nmeas", 1)
n_save = g.default.get_int("--nsave", 100)
t_flow = g.default.get_float("--tflow", 2.0)
eps_flow = g.default.get_float("--epsflow", 0.1)
root = g.default.get("--root", ".")
seed = g.default.get("--seed", "nambu-obc5d-topo")

grid = g.grid([L, L, L, L], g.double)

# resume from latest checkpoint if present
U = g.qcd.gauge.unit(grid)  # cold start at weak coupling
it0 = 0
latest = latest_checkpoint(root, n_traj)
if latest is not None:
    g.copy(U, g.load(f"{root}/ckpoint_lat.{latest}"))
    it0 = latest + 1
rng = g.random(f"{seed}-{it0}")
np_rng = np.random.default_rng(
    int(hashlib.sha256(f"{seed}-{it0}".encode()).hexdigest()[:16], 16)
)

g.message(
    f"""
Nambu HMC with 5D OBC scaffold: topological tunneling run
  lattice = {grid.fdimensions}
  beta    = {beta}
  c_p     = {c_p}, c_r = {c_r}, kappa5 = {kappa5}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm} (skipped on resume), ntraj = {n_traj}
  nmeas   = {n_meas}, tflow = {t_flow} (eps_flow = {eps_flow}, radius = {np.sqrt(8 * t_flow):.2f})
  nsave   = {n_save}, root = {root}
  resume  = {"trajectory " + str(it0) if it0 > 0 else "fresh start"}
"""
)

V = g.lattice(U[0])
W = [g.lattice(U[0]) for mu in range(4)]

nd = len(U)


# exact Haar draw on SU(3): QR of complex Gaussian, column phase fix,
# det^(1/3) projection U(3) -> SU(3)
def haar_su3(field):
    coords = g.coordinates(field)
    n = len(coords)
    A = (np_rng.standard_normal((n, 3, 3)) + 1j * np_rng.standard_normal((n, 3, 3))) / np.sqrt(2.0)
    Q, Rm = np.linalg.qr(A)
    d = np.diagonal(Rm, axis1=1, axis2=2)
    Q = Q * (d / np.abs(d))[:, None, :]
    Q = Q / np.linalg.det(Q)[:, None, None] ** (1.0 / 3.0)
    field[coords] = Q.astype(grid.precision.complex_dtype)


def refresh_scaffold():
    haar_su3(V)
    for mu in range(nd):
        haar_su3(W[mu])


refresh_scaffold()

# big lists: 9 link species, each with its own (p, r)
Ubig = U + [V] + W
Pbig = g.group.cartesian(Ubig)
Rbig = g.group.cartesian(Ubig)

a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

# rung coupling S5 as differentiable functional via reverse-mode AD
aU = [rad.node(g.copy(u)) for u in U]
aV = rad.node(g.copy(V))
aW = [rad.node(g.copy(w)) for w in W]

s5_node = None
for mu in range(nd):
    t = g.sum(g.trace(aU[mu] * g.cshift(aV, mu, 1) * g.adj(aW[mu]) * g.adj(aV)))
    s5_node = t if s5_node is None else s5_node + t
s5_node = (s5_node + g.adj(s5_node)) * 0.5  # Re part
s5_func = s5_node.functional(*(aU + [aV] + aW))

zero_alg = g.group.cartesian(U[0])
zero_alg[:] = 0


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


# PRURP over all 9 species; e_a H = 0 for the scaffold links
def frc_P():
    F_S = a_S.gradient(U, U) + [zero_alg] * 5
    F_G = s5_func.gradient(Ubig, Ubig)
    F = [g(c_r * fs - kappa5 * fg) for fs, fg in zip(F_S, F_G)]
    return cw_list(F, Rbig)


def frc_R():
    F_S = a_S.gradient(U, U) + [zero_alg] * 5
    F_G = s5_func.gradient(Ubig, Ubig)
    F = [g(kappa5 * fg - c_p * fs) for fs, fg in zip(F_S, F_G)]
    return cw_list(F, Pbig)


def vel_U():
    return [g((c_r - c_p) * o) for o in cw_list(Pbig, Rbig)]


ip_P = sympl.update_p(Pbig, frc_P, tag="P")
ip_R = sympl.update_p(Rbig, frc_R, tag="R")
iq_U = sympl.update_q(Ubig, vel_U, tag="U")
mdint = sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


def hamiltonian_H():
    return a_kin(Pbig) + a_kin(Rbig) + a_S(U)


def nambu_hmc(tau, accept_reject=True):
    refresh_scaffold()  # exact Gibbs step: V, W ~ Haar
    rng.normal_element(Pbig)
    rng.normal_element(Rbig)
    if not accept_reject:
        # thermalization from cold start: integrate without Metropolis
        mdint(tau)
        return [1.0, 0.0]
    accrej = metro(Ubig)
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
t0 = time.time()
for i in range(it0, n_traj):
    history += [nambu_hmc(tau)]
    g.message(
        f"Trajectory {i}: plaquette = {g.qcd.gauge.plaquette(U):.6f}, "
        f"dH = {history[-1][1]:+.4e}, accept = {int(history[-1][0])}, "
        f"S5 = {s5_func(Ubig):+.4f}"
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
