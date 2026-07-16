#!/usr/bin/env python3
#
# Nambu HMC with stout-smeared clover topological charge in G, with G
# LINEAR in r (arXiv:2409.18958 eq. 30 form). The directed-drive
# counterpart of nambu_obc5d_linear_scaling.py: same kinetic structure
# and equations of motion, but the exotic term in G is the smeared
# topological charge instead of the 5D scaffold rung coupling.
#
#   H = sum p^2/2 + sum r^2/2 + S_beta(U)
#   G = gamma * sum r_a + kappa * Q_sm(U)
#   Q_sm = clover Q (g.qcd.gauge.differentiable_topology) evaluated on
#          n_sm stout-smeared links (rho), chain rule through the
#          smearing via differentiable_functional.transformed + stout
#          jacobian
#
# Equations of motion (paper eqs. 16, 18 with dG/dr_a = gamma,
# dG/dp_a = 0, F_Q = e_a Q_sm):
#   Udot   = -sum_a (gamma p_a) T_a U        (standard-HMC-like velocity)
#   pdot_a = kappa (e_a Q_sm) r_a - gamma (e_a S_beta)
#   rdot_a = -kappa (e_a Q_sm) p_a
# update_p convention (dst <- dst - eps*frc):
#   frc_P = gamma * F_S - kappa * F_Q o r
#   frc_R = kappa * F_Q o p
#
# gamma = 1, kappa = 0 reduces exactly to standard HMC. G has no p
# dependence, so reversibility (G(-p) = G(p)) is automatic.
#
# Tests: AD gradient check through the smearing; reversibility;
#        |dH|, |dG| ~ eps^2.
#
import gpt as g
import numpy as np
from gpt.ad import reverse as rad

# parameters
beta = g.default.get_float("--beta", 1.0)
L = g.default.get_int("--L", 4)
gamma = g.default.get_float("--gamma", 1.0)
kappa = g.default.get_float("--kappa", 10.0)
rho = g.default.get_float("--rho", 0.12)
n_sm = g.default.get_int("--nsm", 2)
tau = g.default.get_float("--tau", 1.0)
n_therm = g.default.get_int("--ntherm", 20)
seed = g.default.get("--seed", "nambu-smq-linear-test")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Nambu HMC with smeared-Q in G (linear in r): scaling test
  lattice = {grid.fdimensions}
  beta    = {beta}
  gamma   = {gamma}
  kappa   = {kappa}
  rho     = {rho}, n_sm = {n_sm}
  tau     = {tau}
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

# smeared clover topological charge as differentiable functional:
# Q_sm(U) = Q_clover(stout^n_sm(U)), gradient through the smearing via
# transformed() + stout jacobian
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


# sum_{fields, x, a} of the adjoint coefficients (the linear-in-r term of G)
def lin_sum(A):
    s = 0.0
    for x in A:
        for c in x.otype.coordinates(x):
            s += complex(g.sum(c)).real
    return s


def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


def hamiltonian_G():
    return gamma * lin_sum(R) + kappa * q_func(U)


# PRURP (paper eq. 19): nested leapfrogs, no new integrator code
def make_prurp(n_steps):
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

    return sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


################################################################################
# Test 0: smeared-Q functional sanity and AD gradient check
################################################################################
g.message("=" * 70)
g.message("Test 0: smeared-Q value and gradient through the stout chain rule")

q_raw = g.qcd.gauge.topological_charge_5LI(U)
g.message(f"  Q_5LI (raw links)        = {q_raw:.6f}")
g.message(f"  Q_sm  (clover, {n_sm} stout) = {q_func(U):.6f}")
q_func.assert_gradient_error(rng, U, U, 1e-3, 1e-7)
g.message("Test 0 passed")

################################################################################
# thermalization of U with standard HMC
################################################################################
g.message("=" * 70)
g.message(f"Thermalizing {n_therm} standard HMC trajectories at beta = {beta}")

P4 = g.group.cartesian(U)
ip = sympl.update_p(P4, lambda: a_S.gradient(U, U))
iq = sympl.update_q(U, lambda: a_kin.gradient(P4, P4))
mdint_std = sympl.leap_frog(20, ip, iq)
metro = g.algorithms.markov.metropolis(rng)

for i in range(n_therm):
    rng.normal_element(P4)
    accrej = metro(U)
    h0 = a_kin(P4) + a_S(U)
    mdint_std(tau)
    h1 = a_kin(P4) + a_S(U)
    accrej(h1, h0)

g.message(f"Thermalized: plaquette = {g.qcd.gauge.plaquette(U):.6f}")
U0 = g.copy(U)

################################################################################
# Test 1: reversibility
################################################################################
g.message("=" * 70)
g.message("Test 1: reversibility of PRURP with smeared-Q linear-in-r G")

n_steps = 20
mdint = make_prurp(n_steps)

g.copy(U, U0)
rng.normal_element(P)
rng.normal_element(R)
P0 = g.copy(P)
R0 = g.copy(R)

mdint(tau)
for i in range(len(P)):
    P[i] @= g(-P[i])
mdint(tau)

err_U = max((g.norm2(U[i] - U0[i]) / g.norm2(U0[i])) ** 0.5 for i in range(nd))
err_P = max((g.norm2(g(-P[i]) - P0[i]) / g.norm2(P0[i])) ** 0.5 for i in range(nd))
err_R = max((g.norm2(R[i] - R0[i]) / g.norm2(R0[i])) ** 0.5 for i in range(nd))
g.message(f"  reversibility error: U = {err_U:.2e}, P = {err_P:.2e}, R = {err_R:.2e}")
assert max(err_U, err_P, err_R) < 1e-10

g.message("Test 1 passed")

################################################################################
# Test 2: dH and dG scaling with step size
################################################################################
g.message("=" * 70)
g.message("Test 2: |dH|, |dG| vs eps at fixed trajectory length")

n_meas = 3
steps_list = [5, 10, 20, 40, 80]
eps_list = []
dH_list = []
dG_list = []

# same momenta draws for all step sizes
draws = []
for m in range(n_meas):
    rng.normal_element(P)
    rng.normal_element(R)
    draws.append((g.copy(P), g.copy(R)))

for n_steps in steps_list:
    mdint = make_prurp(n_steps)
    dHs, dGs = [], []
    for m in range(n_meas):
        g.copy(U, U0)
        g.copy(P, draws[m][0])
        g.copy(R, draws[m][1])
        h0 = hamiltonian_H()
        g0 = hamiltonian_G()
        mdint(tau)
        dHs.append(abs(hamiltonian_H() - h0))
        dGs.append(abs(hamiltonian_G() - g0))
    eps = tau / n_steps
    eps_list.append(eps)
    dH_list.append(np.mean(dHs))
    dG_list.append(np.mean(dGs))
    g.message(
        f"  N = {n_steps:3d}  eps = {eps:.4f}   |dH| = {dH_list[-1]:.6e}   |dG| = {dG_list[-1]:.6e}"
    )

# fit excludes the coarsest step (outside asymptotic regime)
slope_H = np.polyfit(np.log(eps_list[1:]), np.log(dH_list[1:]), 1)[0]
slope_G = np.polyfit(np.log(eps_list[1:]), np.log(dG_list[1:]), 1)[0]
g.message(f"  fitted slope: dH ~ eps^{slope_H:.3f}, dG ~ eps^{slope_G:.3f}  (expect 2)")
assert abs(slope_H - 2.0) < 0.3
assert abs(slope_G - 2.0) < 0.3

g.message("Test 2 passed")
g.message("=" * 70)
g.message("All tests passed")
