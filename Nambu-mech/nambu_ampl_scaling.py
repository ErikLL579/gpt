#!/usr/bin/env python3
#
# Nambu HMC with a scale-selective parametric amplifier in G
# (arXiv:2409.18958 quadratic scheme, g3 = smeared Wilson action),
# optionally mixed with a linear-in-r term ("idle + supercharger").
#
#   H = sum p^2/2 + sum r^2/2 + S_beta(U)
#   G = gamma sum r_a + c_p sum p^2/2 + c_r sum r^2/2 + lam * S_sm(U)
#   S_sm = Wilson action at the same beta evaluated on n_sm stout-smeared
#          links (rho); gradient through the smearing via
#          differentiable_functional.transformed + stout jacobian
#
# Mechanism: per adjoint component, (p_a, r_a) obey
#   pdot_a = -A_a r_a,  rdot_a = -B_a p_a
#   A = (c_r F_H - F_G)_a,  B = (F_G - c_p F_H)_a,  F_G = lam * F_sm
# Eigenvalues +-sqrt(AB): AB < 0 -> rotor (bounded precession);
# AB > 0 -> amplifier (exponential growth at rate sqrt(AB), (p,r) lock
# onto the unstable eigenvector, U-velocity (c_r-c_p) p_a r_a becomes
# directed). AB > 0 iff the force ratio rho_a = F_G,a/F_H,a lies in the
# window (c_p, c_r). With lam in the window: IR/smooth components have
# F_sm ~ F_H -> rho ~ lam -> amplified; UV components have F_sm ~ 0 ->
# rho ~ 0 -> rotors. Scale-selective energy concentration.
#
# EOM (update_p convention dst <- dst - eps*frc):
#   Udot_a = gamma p_a + (c_r - c_p) p_a r_a
#   frc_P  = gamma F_H + cw(c_r F_H - lam F_sm, R)
#   frc_R  = cw(lam F_sm - c_p F_H, P)
# gamma = 0 is the pure quadratic amplifier; gamma = 1 restores the full
# HMC ballistic velocity for every component (removing the p*r transport
# tax: prefactor c_r-c_p, <|pr|>/<|p|> = 0.64, rotor sign-flips at
# 2 sqrt|AB|) while the armed components keep the same +-sqrt(AB)
# hyperbolic instability on top. gamma = 1, lam = 0, c_p = c_r = 0 is
# exactly standard HMC.
#
# Tests: AD gradient check through smearing; window statistics on a
# thermalized config; reversibility; |dH|, |dG| ~ eps^2.
#
import gpt as g
import numpy as np

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 4)
gamma = g.default.get_float("--gamma", 0.0)  # 0 = pure quadratic
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
lam = g.default.get_float("--lam", 1.125)  # mid-window (c_p+c_r)/2
rho = g.default.get_float("--rho", 0.12)
n_sm = g.default.get_int("--nsm", 2)
tau = g.default.get_float("--tau", 1.0)
n_therm = g.default.get_int("--ntherm", 20)
seed = g.default.get("--seed", "nambu-ampl-test")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)

g.message(
    f"""
Nambu HMC with smeared-Wilson amplifier G: scaling test
  lattice = {grid.fdimensions}
  beta    = {beta}
  gamma   = {gamma}  (linear-in-r idle velocity; 0 = pure quadratic)
  c_p     = {c_p}, c_r = {c_r}, lam = {lam}  (window: c_p < lam < c_r -> amplifier)
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

# smeared Wilson action: S_sm(U) = S_beta(stout^n_sm(U)), gradient via
# stout jacobian chain rule
a_Ssm = g.qcd.gauge.action.wilson(beta)
sm = g.qcd.gauge.smear.stout(rho=rho)
for _ in range(n_sm):
    a_Ssm = a_Ssm.transformed(sm)

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
    return gamma * lin_sum(R) + c_p * a_kin(P) + c_r * a_kin(R) + lam * a_Ssm(U)


# per-component window statistics: fraction of (link, adjoint-index)
# components with AB > 0 (amplifier) and the distribution of growth rates
def window_stats():
    F_H = a_S.gradient(U, U)
    F_G = a_Ssm.gradient(U, U)
    n_tot = 0
    n_amp = 0
    rates = []
    for fh, fg in zip(F_H, F_G):
        for ch, cg in zip(fh.otype.coordinates(fh), fg.otype.coordinates(fg)):
            h = ch[:].real.flatten()
            q = lam * cg[:].real.flatten()
            AB = (c_r * h - q) * (q - c_p * h)
            n_tot += len(AB)
            n_amp += int((AB > 0).sum())
            if (AB > 0).any():
                rates.append(np.sqrt(AB[AB > 0]))
    rates = np.concatenate(rates) if rates else np.array([0.0])
    return n_amp / n_tot, rates.mean(), rates.max()


def make_prurp(n_steps):
    def frc_P():
        F_H = a_S.gradient(U, U)
        F_G = a_Ssm.gradient(U, U)
        F = [g(c_r * fh - lam * fg) for fh, fg in zip(F_H, F_G)]
        return [g(gamma * fh + fr) for fh, fr in zip(F_H, cw_list(F, R))]

    def frc_R():
        F_H = a_S.gradient(U, U)
        F_G = a_Ssm.gradient(U, U)
        F = [g(lam * fg - c_p * fh) for fh, fg in zip(F_H, F_G)]
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
# Test 0: smeared-action gradient and amplifier-window statistics
################################################################################
g.message("=" * 70)
g.message("Test 0: smeared-Wilson gradient through the stout chain rule")

g.message(f"  S(U)    = {a_S(U):.6f}")
g.message(f"  S_sm(U) = {a_Ssm(U):.6f}")
a_Ssm.assert_gradient_error(rng, U, U, 1e-3, 1e-7)

frac, rate_mean, rate_max = window_stats()
g.message(
    f"  amplifier window: {100 * frac:.1f}% of components have AB > 0, "
    f"growth rate mean = {rate_mean:.4f}, max = {rate_max:.4f} (MD-time units)"
)
g.message("Test 0 passed")

################################################################################
# Test 1: reversibility
################################################################################
g.message("=" * 70)
g.message("Test 1: reversibility of PRURP with amplifier G")

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
