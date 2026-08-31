#!/usr/bin/env python3
#
# Multi-step gauge-fixing flow HMC with EXACT LOCAL Jacobians, 4D.
#
# Chain (Luscher trivialising-map form, arXiv:0907.5491 eq. 6.5):
#
#     W = V_0 -> V_1 = K_0(V_0) -> ... -> V_N = K_{N-1}(V_{N-1}) = U
#
# W is the dynamical variable.  The density in W is
#
#     p(W) ~ exp(-S_beta(U(W))) * prod_k |det J_k(V_k)|
#
# Every K_k is a *gauge transformation* and S_beta is gauge invariant, so
# S_beta(U(W)) = S_beta(W) exactly: the Wilson force is evaluated on W and
# needs NO chain rule.  Only the log-det forces are pulled back, since
# S_ldj,k lives at V_k which depends on W through K_0...K_{k-1}:
#
#     dS/dW = dS_beta/dW + sum_k (dV_k/dW)^T dS_ldj,k/dV_k
#
# computed by the O(N) backward recursion in log_det_force() below.
#
# Each step is masked on its own checkerboard (alternating), so each J_k is
# block diagonal over that step's active sites and log det J_k is exact and
# local.  The *composite* Jacobian is not block diagonal -- the stars of
# successive steps overlap -- but it is never needed.
#
import gpt as g
import numpy as np
import time
from gf_local_jacobian import gauge_fixing_step

nd, size, beta = 4, 4, 10.0
eps_gf = g.default.get_float("--eps_gf", 1e-2)
n_gf = g.default.get_int("--n_gf", 2)
pullback = not g.default.has("--no_pullback")   # control: drop J_k^T F to test sensitivity
tau = g.default.get_float("--tau", 0.25)
nsteps = g.default.get_int("--nsteps", 8)

grid = g.grid([size] * nd, g.double)
rng = g.random("t")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

# keep every step alive for the whole run: a garbage-collected
# differentiable_field_transformation corrupts later ones over the same U
gfs = [
    gauge_fixing_step(U, eps_gf, checkerboard=[g.even, g.odd][k % 2]) for k in range(n_gf)
]
alds = [gf.action_log_det_jacobian() for gf in gfs]

g.message(f"nd={nd}, L={size}, beta={beta}, eps_gf={eps_gf}, tau={tau}, nsteps={nsteps}")
g.message(f"pullback = {pullback}")
g.message(f"n_gf_steps = {n_gf}, checkerboards = {['even','odd'][:1]*0 + [['even','odd'][k%2] for k in range(n_gf)]}")
g.message(f"block size per active site: {gfs[0].n_block} x {gfs[0].n_block}")

W = g.copy(U)                      # dynamical variable
U_mom = g.group.cartesian(W)
a_mom = g.qcd.scalar.action.mass_term()
a_gauge = g.qcd.gauge.action.wilson(beta)


def forward_pass(W):
    """V_0 = W, V_{k+1} = K_k(V_k); returns [V_0, ..., V_N]."""
    V = [W]
    for k in range(n_gf):
        V.append(gfs[k](V[-1]))
    return V


def log_det_actions(W):
    V = forward_pass(W)
    return [alds[k](V[k]) for k in range(n_gf)]


def log_det_force():
    """dS_ldj/dW by Luscher backward recursion (O(N) in the number of steps)."""
    V = forward_pass(W)
    F = alds[n_gf - 1].gradient(V[n_gf - 1], V[n_gf - 1])
    for k in range(n_gf - 2, -1, -1):
        if pullback:
            F = gfs[k].dfm.jacobian(V[k], V[k + 1], F)  # pull back through K_k
        F_local = alds[k].gradient(V[k], V[k])
        F = [g(F[mu] + F_local[mu]) for mu in range(nd)]
    return F


sympl = g.algorithms.integrator.symplectic
iq = sympl.update_q(W, lambda: a_mom.gradient(U_mom, U_mom))
ip_g = sympl.update_p(U_mom, lambda: a_gauge.gradient(W, W))   # NO pullback
ip_l = sympl.update_p(U_mom, log_det_force)

# S_beta is blind to the maps, so pre-thermalise with plain Wilson HMC
for i in range(20):
    rng.normal_element(U_mom)
    sympl.leap_frog(20, ip_g, iq)(1.0)
g.message(f"pre-thermalised: plaq = {g.qcd.gauge.plaquette(W)}")

V = forward_pass(W)
for k in range(n_gf):
    g.message(f"  step {k}: log-det action = {alds[k](V[k]):.8f}")
g.message(f"plaq along the chain: {[f'{g.qcd.gauge.plaquette(v):.8f}' for v in V]}")

rng.normal_element(U_mom)
ld0 = log_det_actions(W)
s0, m0 = a_gauge(W), a_mom(U_mom)
g.message(f"H0: gauge = {s0:.8f}, log-det = {sum(ld0):.8f} {['%.6f'%x for x in ld0]}, mom = {m0:.8f}")

t0 = time.time()
sympl.leap_frog(nsteps, ip_l, sympl.leap_frog(1, ip_g, iq))(tau)
dt = time.time() - t0

ld1 = log_det_actions(W)
s1, m1 = a_gauge(W), a_mom(U_mom)
g.message(f"H1: gauge = {s1:.8f}, log-det = {sum(ld1):.8f} {['%.6f'%x for x in ld1]}, mom = {m1:.8f}")
g.message(f"dS_gauge = {s1-s0:+.6e}, dS_logdet = {sum(ld1)-sum(ld0):+.6e}, dS_mom = {m1-m0:+.6e}")
g.message(f"dH = {(s1+sum(ld1)+m1)-(s0+sum(ld0)+m0):+.6e}   [trajectory {dt:.1f} s]")
g.message(f"final plaq = {g.qcd.gauge.plaquette(W)}")
