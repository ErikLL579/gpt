#!/usr/bin/env python3
#
# Nambu HMC with a 5th-dimension OBC scaffold coupled through G,
# with G LINEAR in r (arXiv:2409.18958 eq. 30 form + open-boundary
# extra dimension). Counterpart of nambu_obc5d_scaling.py; only the
# auxiliary Hamiltonian and the equations of motion differ.
#
# Geometry: physical 4D links U_mu(x); one extra layer of edge links
# W_mu(x); vertical links V5(x) connecting the layers. The 5th dimension
# terminates above the edge layer (single rung of mu5 plaquettes, no
# wrap-around term) -> open boundary condition by construction.
#
#   H = sum_all p^2/2 + sum_all r^2/2 + S_beta(U)
#   G = gamma * sum_all r_a + kappa5 * S5(U, V, W)
#   S5 = sum_{x,mu} Re Tr [ U_mu(x) V5(x+mu) W_mu(x)^dag V5(x)^dag ]
#
# Equations of motion (paper eqs. 16, 18 with dG/dr_a = gamma,
# dG/dp_a = 0):
#   Udot   = -sum_a (gamma p_a) T_a U        (standard-HMC-like velocity)
#   pdot_a = kappa5 (e_a S5) r_a - gamma (e_a S_beta)
#   rdot_a = -kappa5 (e_a S5) p_a
# update_p convention (dst <- dst - eps*frc):
#   frc_P = gamma * F_S - kappa5 * F_G o r
#   frc_R = kappa5 * F_G o p
#
# gamma = 1, kappa5 = 0 reduces exactly to standard HMC ("minimal
# deformation", paper Sec. V). G has no p dependence, so reversibility
# (G(-p) = G(p)) is automatic.
#
# V, W carry no weight in H (flat = Haar marginal); their exact Gibbs
# refresh is a uniform Haar draw each trajectory (QR construction).
# All U<->scaffold coupling lives exclusively in G, so the 4D marginal
# is exactly exp(-S_beta(U)) for any kappa5.
#
# Tests: Haar draw quality; AD gradient check; reversibility;
#        |dH|, |dG| ~ eps^2.
#
import gpt as g
import numpy as np
import hashlib
from gpt.ad import reverse as rad

# parameters
beta = g.default.get_float("--beta", 1.0)
L = g.default.get_int("--L", 4)
gamma = g.default.get_float("--gamma", 1.0)
kappa5 = g.default.get_float("--kappa5", 1.0)
tau = g.default.get_float("--tau", 1.0)
n_therm = g.default.get_int("--ntherm", 20)
seed = g.default.get("--seed", "nambu-obc5d-linear-test")

grid = g.grid([L, L, L, L], g.double)
rng = g.random(seed)
np_rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))

g.message(
    f"""
Nambu HMC with 5D OBC scaffold in G (linear in r): scaling test
  lattice = {grid.fdimensions}
  beta    = {beta}
  gamma   = {gamma}
  kappa5  = {kappa5}
  tau     = {tau}
"""
)

# physical links, scaffold links
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

# rung coupling S5 as differentiable functional via reverse-mode AD
aU = [rad.node(g.copy(u)) for u in U]
aV = rad.node(g.copy(V))
aW = [rad.node(g.copy(w)) for w in W]

s5_node = None
for mu in range(nd):
    t = g.sum(g.trace(aU[mu] * g.cshift(aV, mu, 1) * g.adj(aW[mu]) * g.adj(aV)))
    s5_node = t if s5_node is None else s5_node + t
s5_node = (s5_node + g.adj(s5_node)) * 0.5  # Re part, cf. differentiable_P_and_R
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


# sum_{fields, x, a} of the adjoint coefficients (the linear-in-r term of G)
def lin_sum(A):
    s = 0.0
    for x in A:
        for c in x.otype.coordinates(x):
            s += complex(g.sum(c)).real
    return s


def hamiltonian_H():
    return a_kin(Pbig) + a_kin(Rbig) + a_S(U)


def hamiltonian_G():
    return gamma * lin_sum(Rbig) + kappa5 * s5_func(Ubig)


# PRURP over all 9 species; e_a H = 0 for the scaffold links
def make_prurp(n_steps):
    def frc_P():
        F_S = a_S.gradient(U, U) + [zero_alg] * 5
        F_G = s5_func.gradient(Ubig, Ubig)
        F_Gr = cw_list(F_G, Rbig)
        return [g(gamma * fs - kappa5 * fgr) for fs, fgr in zip(F_S, F_Gr)]

    def frc_R():
        F_G = s5_func.gradient(Ubig, Ubig)
        return [g(kappa5 * fgp) for fgp in cw_list(F_G, Pbig)]

    def vel_U():
        return [g(gamma * p) for p in Pbig]

    ip_P = sympl.update_p(Pbig, frc_P, tag="P")
    ip_R = sympl.update_p(Rbig, frc_R, tag="R")
    iq_U = sympl.update_q(Ubig, vel_U, tag="U")

    return sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


################################################################################
# Test 0: Haar draw quality and AD gradient check
################################################################################
g.message("=" * 70)
g.message("Test 0: Haar draw quality and gradient of S5")

I = g.identity(g.lattice(V))
err_unit = (g.norm2(V * g.adj(V) - I) / g.norm2(I)) ** 0.5
tr_mean = g.sum(g.trace(V)) / grid.gsites
tr2 = g.norm2(g.trace(V)) / grid.gsites
g.message(f"  unitarity error          = {err_unit:.2e}")
g.message(f"  <Tr V>                   = {tr_mean:.4f}  (Haar: 0)")
g.message(f"  <|Tr V|^2>               = {tr2:.4f}  (Haar: 1)")
assert err_unit < 1e-13
assert abs(tr2 - 1.0) < 0.2

g.message(f"  S5(U,V,W) = {s5_func(Ubig):.6f}")
s5_func.assert_gradient_error(rng, Ubig, Ubig, 1e-3, 1e-7)
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
Ubig0 = g.copy(Ubig)

################################################################################
# Test 1: reversibility
################################################################################
g.message("=" * 70)
g.message("Test 1: reversibility of PRURP with linear-in-r G")

n_steps = 20
mdint = make_prurp(n_steps)

g.copy(Ubig, Ubig0)
rng.normal_element(Pbig)
rng.normal_element(Rbig)
P0 = g.copy(Pbig)
R0 = g.copy(Rbig)

mdint(tau)
for i in range(len(Pbig)):
    Pbig[i] @= g(-Pbig[i])
mdint(tau)

err_U = max((g.norm2(Ubig[i] - Ubig0[i]) / g.norm2(Ubig0[i])) ** 0.5 for i in range(9))
err_P = max((g.norm2(g(-Pbig[i]) - P0[i]) / g.norm2(P0[i])) ** 0.5 for i in range(9))
err_R = max((g.norm2(Rbig[i] - R0[i]) / g.norm2(R0[i])) ** 0.5 for i in range(9))
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
    rng.normal_element(Pbig)
    rng.normal_element(Rbig)
    draws.append((g.copy(Pbig), g.copy(Rbig)))

for n_steps in steps_list:
    mdint = make_prurp(n_steps)
    dHs, dGs = [], []
    for m in range(n_meas):
        g.copy(Ubig, Ubig0)
        g.copy(Pbig, draws[m][0])
        g.copy(Rbig, draws[m][1])
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
