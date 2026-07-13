#!/usr/bin/env python3
#
# Nambu HMC integrator test (arXiv:2409.18958, Erik Lundstrum)
#
# Phase space per link: {U(x,mu), p_a(x,mu), r_a(x,mu)}
#   H(p,U,r) = sum p^2/2 + sum r^2/2 + S_beta(U)
#   G(p,U,r) = c_p sum p^2/2 + c_r sum r^2/2 + S_{lam*beta}(U)
#
# Nambu evolution equations (paper eqs. 16-18) for these separable H, G:
#   Udot_a = dH/dp_a dG/dr_a - dH/dr_a dG/dp_a = (c_r - c_p) p_a r_a
#   pdot_a = e_aG dH/dr_a - e_aH dG/dr_a      = (lam - c_r) (e_aS) r_a
#   rdot_a = e_aH dG/dp_a - e_aG dH/dp_a      = (c_p - lam) (e_aS) p_a
# where e_aS is the Wilson force and lam = beta_G / beta (Wilson force is
# linear in beta, so e_aG = lam * e_aS).
#
# PRURP integrator (paper eq. 19) realized as a two-level nested leapfrog:
#   leap_frog(N, ip_P, leap_frog(1, ip_R, iq_U))
#     = P(tau/2) R(tau/2) U(tau) R(tau/2) P(tau/2) per step.
#
# Products p_a r_a etc. are component-wise in the adjoint index, taken in
# GPT's internal generator basis via otype.coordinates().
#
# Tests:
#   0. basis consistency: <H_kin>/dof = 1/2, coordinates roundtrip
#   1. frozen-r reduction: c_p=0, lam=0, c_r=1, r_a=1 must reproduce
#      standard leapfrog HMC trajectories
#   2. reversibility of PRURP
#   3. |dH|, |dG| ~ eps^2 scaling at fixed trajectory length
#
import gpt as g
import numpy as np

# parameters
beta = g.default.get_float("--beta", 5.6)
L = g.default.get_int("--L", 4)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
lam = g.default.get_float("--lam", 0.5)  # beta_G = lam * beta
tau = g.default.get_float("--tau", 1.0)  # trajectory length for tests
n_therm = g.default.get_int("--ntherm", 20)
seed = g.default.get("--seed", "nambu-test")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Nambu HMC integrator test
  lattice = {grid.fdimensions}
  beta    = {beta}
  c_p     = {c_p}
  c_r     = {c_r}
  lam     = {lam}  (beta_G = {lam * beta})
  tau     = {tau}
"""
)

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

nd = len(U)
otype_alg = U[0].otype.cartesian()
n_gen = len(otype_alg.generators(grid.precision.complex_dtype))

# actions
a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic


# component-wise product in the adjoint index:  (A, B) -> sum_a a_a b_a T_a
def cw(A, B):
    ca = A.otype.coordinates(A)
    cb = B.otype.coordinates(B)
    out = g.lattice(A)
    A.otype.coordinates(out, [g(x * y) for x, y in zip(ca, cb)])
    return out


def cw_list(A, B):
    return [cw(a, b) for a, b in zip(A, B)]


def scaled_list(c, A):
    return [g(c * a) for a in A]


# fields
P = g.group.cartesian(U)
R = g.group.cartesian(U)


def make_ones(template):
    # algebra field with all adjoint coordinates = 1
    one = g.complex(grid)
    one[:] = 1
    out = g.lattice(template)
    out.otype.coordinates(out, [one] * n_gen)
    return out


# Hamiltonians (kinetic terms use the same group inner product / basis
# as coordinates(), see mass_term + su_n.inner_product)
def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


def hamiltonian_G():
    return c_p * a_kin(P) + c_r * a_kin(R) + lam * a_S(U)


# PRURP integrator factory; coefficients passed in so tests can override
def make_prurp(n_steps, cp, cr, la):
    # update_p does dst <- dst - eps * frc(), update_q does U <- exp(eps*frc()) U
    def frc_P():
        F = a_S.gradient(U, U)
        return scaled_list(cr - la, cw_list(F, R))

    def frc_R():
        F = a_S.gradient(U, U)
        return scaled_list(la - cp, cw_list(F, P))

    def vel_U():
        return scaled_list(cr - cp, cw_list(P, R))

    ip_P = sympl.update_p(P, frc_P, tag="P")
    ip_R = sympl.update_p(R, frc_R, tag="R")
    iq_U = sympl.update_q(U, vel_U, tag="U")

    inner = sympl.leap_frog(1, ip_R, iq_U)
    return sympl.leap_frog(n_steps, ip_P, inner)


################################################################################
# Test 0: basis consistency
################################################################################
g.message("=" * 70)
g.message("Test 0: generator basis consistency")

rng.normal_element(P)
dof = grid.gsites * nd * n_gen
kin_per_dof = a_kin(P) / dof
g.message(f"  <p^2/2> per d.o.f. = {kin_per_dof:.6f}  (expect 0.5 +- {(0.5 / dof) ** 0.5:.4f})")

# roundtrip: decompose and reassemble
A = P[0]
tmp = g.lattice(A)
A.otype.coordinates(tmp, A.otype.coordinates(A))
err = (g.norm2(tmp - A) / g.norm2(A)) ** 0.5
g.message(f"  coordinates roundtrip error = {err:.2e}")
assert err < 1e-13

# cw(A, ones) = A
ones = make_ones(P[0])
err = (g.norm2(cw(A, ones) - A) / g.norm2(A)) ** 0.5
g.message(f"  cw(A, 1) - A error          = {err:.2e}")
assert err < 1e-13

# kinetic term equals sum of squared coordinates in the same basis
kin_coord = 0.0
for mu in range(nd):
    for c in P[mu].otype.coordinates(P[mu]):
        kin_coord += 0.5 * g.sum(g(c * c)).real
err = abs(kin_coord - a_kin(P)) / abs(a_kin(P))
g.message(f"  |sum_a p_a^2/2 - mass_term| (rel) = {err:.2e}")
assert err < 1e-12

g.message("Test 0 passed")

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
U0 = g.copy(U)

################################################################################
# Test 1: frozen-r reduction to standard HMC
################################################################################
g.message("=" * 70)
g.message("Test 1: frozen-r limit (c_p=0, lam=0, c_r=1, r_a=1) vs standard leapfrog")

n_steps = 20

# reference: standard leapfrog from (U0, P0)
rng.normal_element(P)
P0 = g.copy(P)
g.copy(U, U0)
h0 = a_kin(P) + a_S(U)
mdint_std(tau)
h1 = a_kin(P) + a_S(U)
U_ref = g.copy(U)
dH_ref = h1 - h0

# PRURP with inert r
g.copy(U, U0)
g.copy(P, P0)
for mu in range(nd):
    R[mu] @= ones
mdint_frozen = make_prurp(n_steps, cp=0.0, cr=1.0, la=0.0)
mdint_frozen(tau)

err = max(
    (g.norm2(U[mu] - U_ref[mu]) / g.norm2(U_ref[mu])) ** 0.5 for mu in range(nd)
)
dH_frozen = (a_kin(P) + a_S(U)) - h0
g.message(f"  max rel. deviation of U from standard HMC = {err:.2e}")
g.message(f"  dH standard = {dH_ref:.6e}, dH frozen-r PRURP = {dH_frozen:.6e}")
assert err < 1e-10

g.message("Test 1 passed")

################################################################################
# Test 2: reversibility of PRURP with full coefficients
################################################################################
g.message("=" * 70)
g.message(f"Test 2: reversibility (c_p={c_p}, c_r={c_r}, lam={lam})")

mdint = make_prurp(n_steps, cp=c_p, cr=c_r, la=lam)

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

g.message("Test 2 passed")

################################################################################
# Test 3: dH and dG scaling with step size
################################################################################
g.message("=" * 70)
g.message("Test 3: |dH|, |dG| vs eps at fixed trajectory length")

n_meas = 3
steps_list = [5, 10, 20, 40, 80]
eps_list = []
dH_list = []
dG_list = []

# use the same momenta draws for all step sizes so dH(eps), dG(eps) are
# deterministic functions and the fit is not contaminated by trajectory noise
draws = []
for m in range(n_meas):
    rng.normal_element(P)
    rng.normal_element(R)
    draws.append((g.copy(P), g.copy(R)))

for n_steps in steps_list:
    mdint = make_prurp(n_steps, cp=c_p, cr=c_r, la=lam)
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

# fit excludes the coarsest step, where dH is O(1) and outside the
# asymptotic eps^2 regime
slope_H = np.polyfit(np.log(eps_list[1:]), np.log(dH_list[1:]), 1)[0]
slope_G = np.polyfit(np.log(eps_list[1:]), np.log(dG_list[1:]), 1)[0]
g.message(f"  fitted slope: dH ~ eps^{slope_H:.3f}, dG ~ eps^{slope_G:.3f}  (expect 2)")
assert abs(slope_H - 2.0) < 0.3
assert abs(slope_G - 2.0) < 0.3

g.message("Test 3 passed")
g.message("=" * 70)
g.message("All tests passed")
