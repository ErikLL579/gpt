#!/usr/bin/env python3
#
# Nambu HMC with the dislocation-gated nucleation amplifier in G
# (arXiv:2409.18958 quadratic scheme mixed with a linear-in-r term,
# "idle + supercharger"): scaling/conservation test.
#
#   H = sum p^2/2 + sum r^2/2 + S_beta(U)      (Metropolis on H)
#   G = gamma sum r_a + c_p sum p^2/2 + c_r sum r^2/2 + g3(U)
#   g3 = lam * D_sm^2,   D_sm = Q_sm(n_sm1) - Q_sm(n_sm2)
#   Q_sm(n) = clover Q (g.qcd.gauge.differentiable_topology) on n
#             stout-smeared links; gradient via transformed + stout chain
#
# D_sm is the differentiable proxy of the dislocation detector
# |Qclover - Q5LI| (measured on hmc_60.log: 10.6x baseline AT tunneling
# events, 7-9x elevated 3 measurements BEFORE — predictive). Depth
# pair (1, 3) chosen for cost (AD force ~ total chain depth); the
# shallow charge sits in the UV-noise regime, so the equilibrium D
# floor is larger than deeper pairs' — see the production script
# header for the measured tradeoff.
#
# Gate structure: F_G = 2 lam D_sm (F_Q1 - F_Q2). In equilibrium
# D_sm ~ 0.07 and the per-component floor is small (measured on the
# beta=6 8^4 checkpoint: |F_G/F_H| p99.9 = 5.7e-2 at lam=1), so nothing
# arms and gamma=1 idles as plain HMC. A forming dislocation raises
# D_sm AND localizes |grad D_sm| at the lump: those components enter the
# window s = F_G/F_H in (c_p, c_r), arm with rate sqrt(AB), and the
# nucleation site gets directed energy. Spurious floor arming is
# rate-suppressed (in-window => rate ~ |F_H,a| which must be ~ |F_G,a|
# ~ floor).
#
# EOM (update_p convention dst <- dst - eps*frc):
#   Udot_a = gamma p_a + (c_r - c_p) p_a r_a
#   frc_P  = gamma F_H + cw(c_r F_H - F_G, R)
#   frc_R  = cw(F_G - c_p F_H, P)
#
# Tests: composite gradient (chain rule 2 lam D (F1 - F2)) vs finite
# differences; gate floor statistics; reversibility; |dH|, |dG| ~ eps^2.
#
import gpt as g
import numpy as np
from gpt.ad import reverse as rad
from gpt.core.group import differentiable_functional

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 4)
gamma = g.default.get_float("--gamma", 1.0)  # 1 = idle + supercharger
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
lam = g.default.get_float("--lam", 1.75)
rho = g.default.get_float("--rho", 0.12)
n_sm1 = g.default.get_int("--nsm1", 1)
n_sm2 = g.default.get_int("--nsm2", 3)
tau = g.default.get_float("--tau", 0.5)
n_therm = g.default.get_int("--ntherm", 100)
seed = g.default.get("--seed", "nambu-disloc-test")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Nambu HMC with dislocation-gated amplifier G: scaling test
  lattice = {grid.fdimensions}
  beta    = {beta}
  gamma   = {gamma}  (linear-in-r idle velocity; 1 = HMC idle)
  c_p     = {c_p}, c_r = {c_r}, lam = {lam}  (window in s = F_G/F_H: ({c_p}, {c_r}))
  rho     = {rho}, n_sm1 = {n_sm1}, n_sm2 = {n_sm2}
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

# smeared clover topological charges at the two depths
sm = g.qcd.gauge.smear.stout(rho=rho)


def make_qfunc(n):
    aU = [rad.node(g.copy(u)) for u in U]
    qf = g.qcd.gauge.differentiable_topology(aU).functional(*aU)
    for _ in range(n):
        qf = qf.transformed(sm)
    return qf


class disloc_functional(differentiable_functional):
    # g3(U) = lam * (Q_sm(n1) - Q_sm(n2))^2, gradient by exact chain
    # rule 2 lam D (F_Q1 - F_Q2) through the validated Q_sm gradients
    def __init__(self, q1, q2, lam):
        self.q1 = q1
        self.q2 = q2
        self.lam = lam

    def d_value(self, fields):
        return self.q1(fields) - self.q2(fields)

    def __call__(self, fields):
        d = self.d_value(fields)
        return self.lam * d * d

    def gradient(self, fields, dfields):
        assert dfields is fields or dfields == fields
        d = self.d_value(fields)
        F1 = self.q1.gradient(fields, fields)
        F2 = self.q2.gradient(fields, fields)
        return [g(2.0 * self.lam * d * (f1 - f2)) for f1, f2 in zip(F1, F2)]


a_g3 = disloc_functional(make_qfunc(n_sm1), make_qfunc(n_sm2), lam)

sympl = g.algorithms.integrator.symplectic


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
    return gamma * lin_sum(R) + c_p * a_kin(P) + c_r * a_kin(R) + a_g3(U)


# per-component gate statistics: D value, fraction of components with
# AB > 0 and growth rates, and the |s| tail (lam tuning diagnostic)
def gate_stats():
    F_H = a_S.gradient(U, U)
    F_G = a_g3.gradient(U, U)
    h, q = [], []
    for fh, fg in zip(F_H, F_G):
        for ch, cg in zip(fh.otype.coordinates(fh), fg.otype.coordinates(fg)):
            h.append(ch[:].real.flatten())
            q.append(cg[:].real.flatten())
    h, q = np.concatenate(h), np.concatenate(q)
    AB = (c_r * h - q) * (q - c_p * h)
    amp = AB > 0
    rates = np.sqrt(AB[amp]) if amp.any() else np.array([0.0])
    ok = np.abs(h) > 1e-12
    s_tail = np.percentile(np.abs(q[ok] / h[ok]), 99.9)
    return a_g3.d_value(U), amp.mean(), rates.mean(), s_tail


# forces recomputed only when U changes: frc_P and frc_R are called
# back-to-back at the same U in the nested leapfrog, and the reverse-AD
# stout chains dominate the cost. Fingerprint = inner products with
# fixed random probe fields (norm2 is BLIND to unitary U: |U|^2 = 3V
# identically); deterministic in U, so rejects/copies/reversal are
# handled without invalidation bookkeeping
_fc = {"key": None}
_fc_probe = [g.lattice(u) for u in U]
for _w in _fc_probe:
    rng.element(_w)


def forces():
    key = tuple(complex(g.sum(g.trace(g.adj(w) * u))) for w, u in zip(_fc_probe, U))
    if _fc["key"] != key:
        _fc["key"] = key
        _fc["F_H"] = a_S.gradient(U, U)
        _fc["F_G"] = a_g3.gradient(U, U)
    return _fc["F_H"], _fc["F_G"]


def make_prurp(n_steps):
    def frc_P():
        F_H, F_G = forces()
        F = [g(c_r * fh - fg) for fh, fg in zip(F_H, F_G)]
        return [g(gamma * fh + fr) for fh, fr in zip(F_H, cw_list(F, R))]

    def frc_R():
        F_H, F_G = forces()
        F = [g(fg - c_p * fh) for fh, fg in zip(F_H, F_G)]
        return cw_list(F, P)

    def vel_U():
        return [g(gamma * p + (c_r - c_p) * o) for p, o in zip(P, cw_list(P, R))]

    ip_P = sympl.update_p(P, frc_P, tag="P")
    ip_R = sympl.update_p(R, frc_R, tag="R")
    iq_U = sympl.update_q(U, vel_U, tag="U")

    return sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


################################################################################
# thermalization of U with standard HMC (also moves U off any special point
# before the stout jacobian is evaluated)
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
# Test 0: composite g3 gradient and gate statistics
################################################################################
g.message("=" * 70)
g.message("Test 0: g3 = lam*D_sm^2 gradient via chain rule through both stout chains")

d0, armed, rate_mean, s_tail = gate_stats()
g.message(f"  Q_sm({n_sm1}) = {a_g3.q1(U):+.6f}, Q_sm({n_sm2}) = {a_g3.q2(U):+.6f}")
g.message(f"  D_sm = {d0:+.6f}, g3 = {a_g3(U):.6e}")
g.message(f"  gate: armed = {100 * armed:.3f}%, rate mean = {rate_mean:.4e}, |s| p99.9 = {s_tail:.3e}")
a_g3.assert_gradient_error(rng, U, U, 1e-3, 1e-5)
g.message("Test 0 passed")

################################################################################
# Test 1: reversibility
################################################################################
g.message("=" * 70)
g.message("Test 1: reversibility of PRURP with dislocation-gated G")

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

n_meas = 5
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

# fit excludes the two coarsest steps (outside asymptotic regime: armed
# components amplify the truncation error by e^{2 sqrt(AB) tau}, which
# is also why the test thermalizes properly and runs at tau = 0.5 —
# an under-thermalized 4^4 config has the gate wide open, D ~ -0.24,
# and dH never reaches its eps^2 regime)
slope_H = np.polyfit(np.log(eps_list[2:]), np.log(dH_list[2:]), 1)[0]
slope_G = np.polyfit(np.log(eps_list[2:]), np.log(dG_list[2:]), 1)[0]
g.message(f"  fitted slope: dH ~ eps^{slope_H:.3f}, dG ~ eps^{slope_G:.3f}  (expect 2)")
assert abs(slope_H - 2.0) < 0.3
assert abs(slope_G - 2.0) < 0.3

g.message("Test 2 passed")
g.message("=" * 70)
g.message("All tests passed")
