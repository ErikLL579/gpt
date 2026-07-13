#!/usr/bin/env python3
#
# Nambu HMC with topological charge in G (arXiv:2409.18958):
#   H(p,U,r) = sum p^2/2 + sum r^2/2 + S_beta(U)
#   G(p,U,r) = c_p sum p^2/2 + c_r sum r^2/2 + kappa * Q(U)
# where Q is the clover-leaf topological charge
# (g.qcd.gauge.differentiable_topology), differentiated with GPT's
# reverse-mode AD:  F_Q = Q.functional(*aU).gradient(U, U).
#
# Evolution equations for general g3(U) in G (F_G = kappa * F_Q):
#   Udot_a = (c_r - c_p) p_a r_a                (unchanged)
#   pdot_a = -(c_r F_S - F_G)_a r_a   ->  frc_P = cw(c_r F_S - kappa F_Q, R)
#   rdot_a = -(F_G - c_p F_S)_a p_a   ->  frc_R = cw(kappa F_Q - c_p F_S, P)
# (update_p convention: dst <- dst - eps * frc)
#
# Tests: reversibility of PRURP; |dH|, |dG| ~ eps^2 at fixed tau.
#
import gpt as g
import numpy as np
from gpt.ad import reverse as rad

# parameters
beta = g.default.get_float("--beta", 1.0)
L = g.default.get_int("--L", 4)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
kappa = g.default.get_float("--kappa", 10.0)
tau = g.default.get_float("--tau", 1.0)
n_therm = g.default.get_int("--ntherm", 20)
seed = g.default.get("--seed", "nambu-topo-test")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Nambu HMC with topological-charge G: scaling test
  lattice = {grid.fdimensions}
  beta    = {beta}
  c_p     = {c_p}
  c_r     = {c_r}
  kappa   = {kappa}
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


def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


def hamiltonian_G():
    return c_p * a_kin(P) + c_r * a_kin(R) + kappa * q_func(U)


# PRURP: P(t/2) R(t/2) U(t) R(t/2) P(t/2) via nested leapfrog
def make_prurp(n_steps):
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

    return sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


################################################################################
# thermalization with standard HMC
################################################################################
g.message("=" * 70)
g.message(f"Thermalizing {n_therm} standard HMC trajectories at beta = {beta}")

ip = sympl.update_p(P, lambda: a_S.gradient(U, U))
iq = sympl.update_q(U, lambda: a_kin.gradient(P, P))
mdint_std = sympl.leap_frog(20, ip, iq)
metro = g.algorithms.markov.metropolis(rng)

for i in range(n_therm):
    rng.normal_element(P)
    accrej = metro(U)
    h0 = a_kin(P) + a_S(U)
    mdint_std(tau)
    h1 = a_kin(P) + a_S(U)
    accrej(h1, h0)

g.message(f"Thermalized: plaquette = {g.qcd.gauge.plaquette(U):.6f}")
g.message(f"Q (clover) = {q_func(U):.6f}, kappa*Q = {kappa * q_func(U):.6f}")
U0 = g.copy(U)

################################################################################
# Test 1: reversibility
################################################################################
g.message("=" * 70)
g.message("Test 1: reversibility of PRURP with topological-charge G")

n_steps = 20
mdint = make_prurp(n_steps)

g.copy(U, U0)
rng.normal_element(P)
rng.normal_element(R)
P0 = g.copy(P)
R0 = g.copy(R)

mdint(tau)
for mu in range(nd):
    P[mu] @= g(-P[mu])
mdint(tau)

err_U = max((g.norm2(U[mu] - U0[mu]) / g.norm2(U0[mu])) ** 0.5 for mu in range(nd))
err_P = max((g.norm2(g(-P[mu]) - P0[mu]) / g.norm2(P0[mu])) ** 0.5 for mu in range(nd))
err_R = max((g.norm2(R[mu] - R0[mu]) / g.norm2(R0[mu])) ** 0.5 for mu in range(nd))
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

# same momenta draws for all step sizes so dH(eps), dG(eps) are
# deterministic functions and the fit is not contaminated by noise
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
