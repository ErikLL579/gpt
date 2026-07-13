#!/usr/bin/env python3
#
# Nambu HMC Markov chain with a 5th-dimension OBC scaffold in G
# (arXiv:2409.18958 + open-boundary extra dimension).
#
# Physical 4D links U_mu(x); edge-layer links W_mu(x); vertical links
# V5(x). Single rung of mu5 plaquettes, no wrap-around -> OBC in the
# 5th direction. Scaffold carries no weight in H (Haar marginal), so
# the 4D marginal is exactly exp(-S_beta(U)) for any kappa5.
#
#   H = sum_all p^2/2 + sum_all r^2/2 + S_beta(U)      (Metropolis on H)
#   G = c_p sum_all p^2/2 + c_r sum_all r^2/2 + kappa5 * S5(U, V, W)
#   S5 = sum_{x,mu} Re Tr [ U_mu(x) V5(x+mu) W_mu(x)^dag V5(x)^dag ]
#
# Per trajectory: exact Gibbs refresh of V, W (uniform Haar draw via QR)
# and of all momenta (Gaussian); PRURP trajectory over all 9 species;
# Metropolis accept/reject on dH.
#
import gpt as g
import numpy as np
import time
import hashlib
from gpt.ad import reverse as rad

# parameters
beta = g.default.get_float("--beta", 1.0)
L = g.default.get_int("--L", 4)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
kappa5 = g.default.get_float("--kappa5", 1.0)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 6)
n_therm = g.default.get_int("--ntherm", 10)
n_traj = g.default.get_int("--ntraj", 50)
seed = g.default.get("--seed", "nambu-obc5d-markov")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)
np_rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))

g.message(
    f"""
Nambu HMC Markov chain with 5D OBC scaffold in G
  lattice = {grid.fdimensions}
  beta    = {beta}
  c_p     = {c_p}
  c_r     = {c_r}
  kappa5  = {kappa5}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm}, ntraj = {n_traj}
"""
)

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)
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

# topological charge of U, measurement only
q_args = [rad.node(g.copy(u)) for u in U]
q_func = g.qcd.gauge.differentiable_topology(q_args).functional(*q_args)

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


def nambu_hmc(tau):
    refresh_scaffold()  # exact Gibbs step: V, W ~ Haar
    rng.normal_element(Pbig)
    rng.normal_element(Rbig)
    accrej = metro(Ubig)
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
        f"accept = {int(history[-1][0])}, Q = {q_func(U):+.6f}, "
        f"S5 = {s5_func(Ubig):+.4f}"
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
