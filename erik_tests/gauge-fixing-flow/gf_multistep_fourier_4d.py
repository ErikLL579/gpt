#!/usr/bin/env python3
#
# Multi-step gauge-fixing flow HMC with EXACT LOCAL Jacobians *and* the
# Zhao (2018) Fourier-accelerated kinetic term.  4D.
#
# This is gf_multistep_4d.py (the N-step Luscher chain) with the trivial
# kinetic term  S_p = 1/2 <pi,pi>  replaced by the FA one, exactly as in
# gf_fourier_4d.py:
#
#     S_p = V * sum_k D^{mu,nu}(k) tr(P_mu(k)^dag P_nu(k)),
#     D = (1/s^2) P^T + (1/M^2) P^L,   s^2 = |k^|^2 + eps_fourier^2
#
# sqrt(D) is the RWGFFA kernel (build_sqrt_fourier_kernel, imported unchanged)
# handed to GPT's built-in g.qcd.scalar.action.fourier_mass_term, which derives
# D = sqrt(D)^2 and D^{-1/2} = inv(sqrt(D)) and supplies draw/energy/velocity.
#
# Chain (Luscher trivialising-map form, arXiv:0907.5491 eq. 6.5):
#
#     W = V_0 -> V_1 = K_0(V_0) -> ... -> V_N = K_{N-1}(V_{N-1}) = U
#
# Every K_k is a gauge transformation and S_beta is gauge invariant, so
# S_beta(U(W)) = S_beta(W): the Wilson force is evaluated on W and needs NO
# chain rule.  Only the log-det forces are pulled back.
#
import gpt as g
import numpy as np
import time
from gf_local_jacobian import gauge_fixing_step
from fourier_kernel import build_sqrt_fourier_kernel

nd, size, beta = 4, 4, 10.0
eps_gf = g.default.get_float("--eps_gf", 0.08)
n_gf = g.default.get_int("--n_gf", 2)
pullback = not g.default.has("--no_pullback")   # control: drop J_k^T F
tau = g.default.get_float("--tau", 0.25)
nsteps = g.default.get_int("--nsteps", 8)
M = g.default.get_float("--M", 1.0)                       # longitudinal eigenvalue 1/M^2
eps_fourier = g.default.get_float("--eps_fourier", 0.5)   # IR regulator

grid = g.grid([size] * nd, g.double)
rng = g.random("t")

g.message(f"nd={nd}, L={size}, beta={beta}, eps_gf={eps_gf}, tau={tau}, nsteps={nsteps}")
g.message(f"n_gf_steps = {n_gf}, checkerboards = {[['even','odd'][k%2] for k in range(n_gf)]}")
g.message(f"pullback = {pullback}")
g.message(f"Fourier kernel: M={M}, eps_fourier={eps_fourier}")

# sqrt(D) split into the nd x nd complex fields fourier_mass_term expects
sqrtD_lat = build_sqrt_fourier_kernel(grid, M, eps_fourier)
coor = g.coordinates(sqrtD_lat)
arr = np.array(sqrtD_lat[coor]).reshape(len(coor), nd, nd)
sqrt_mass = [[g.complex(grid) for _ in range(nd)] for _ in range(nd)]
for mu in range(nd):
    for nu in range(nd):
        sqrt_mass[mu][nu][coor] = np.ascontiguousarray(arr[:, mu, nu])

a_mom = g.qcd.scalar.action.fourier_mass_term(sqrt_mass)
a_mom_std = g.qcd.scalar.action.mass_term()
g.message(f"kinetic term: {a_mom.__name__}")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

# keep every step alive for the whole run
gfs = [
    gauge_fixing_step(U, eps_gf, checkerboard=[g.even, g.odd][k % 2]) for k in range(n_gf)
]
alds = [gf.action_log_det_jacobian() for gf in gfs]
g.message(f"block size per active site: {gfs[0].n_block} x {gfs[0].n_block}")

W = g.copy(U)                      # dynamical variable
U_mom = g.group.cartesian(W)
a_gauge = g.qcd.gauge.action.wilson(beta)

# the velocity fed to update_q is a_mom.gradient; check it really is the
# derivative of the kinetic energy a_mom evaluates (cheap, no log-det involved)
_p = g.group.cartesian(W)
rng.normal_element(_p)
a_mom.assert_gradient_error(rng, _p, _p, 1e-3, 1e-8)


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
iq_std = sympl.update_q(W, lambda: a_mom_std.gradient(U_mom, U_mom))
for i in range(20):
    rng.normal_element(U_mom)
    sympl.leap_frog(20, ip_g, iq_std)(1.0)
g.message(f"pre-thermalised: plaq = {g.qcd.gauge.plaquette(W)}")

V = forward_pass(W)
for k in range(n_gf):
    g.message(f"  step {k}: log-det action = {alds[k](V[k]):.8f}")
g.message(f"plaq along the chain: {[f'{g.qcd.gauge.plaquette(v):.8f}' for v in V]}")

# draw momenta from exp(-pi^dag D pi)
a_mom.draw(U_mom, rng)
for mu in range(nd):
    g.message(f"  P[{mu}]: group defect = {g.group.defect(U_mom[mu]):.3e}")
g.message(f"drawn momenta: fourier KE = {a_mom(U_mom):.8f}, "
          f"trivial KE = {a_mom_std(U_mom):.8f}")

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
