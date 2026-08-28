#!/usr/bin/env python3
# H conservation for the local-Jacobian one-step gauge-fixing HMC in 4D.
#
# 4^4, block size 64x64 per active site (2*nd links x ng generators).
# Short trajectories (tau = 0.25 / 0.5) to keep the cost down: one log-det
# force is ~50 s at 4^4 on 8 threads, and a trajectory costs nsteps+1 of them.
import gpt as g
import numpy as np
import time
from gf_local_jacobian import gauge_fixing_step

nd, size, beta, eps_gf = 4, 4, 10.0, 1e-2
tau = g.default.get_float("--tau", 0.25)

grid = g.grid([size] * nd, g.double)
rng = g.random("t")
U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

gf = gauge_fixing_step(U, eps_gf, checkerboard=g.even)
g.message(f"nd = {nd}, L = {size}, beta = {beta}, eps_gf = {eps_gf}, tau = {tau}")
g.message(f"block size per active site: {gf.n_block} x {gf.n_block}")

Vgf = g.copy(U)
U_mom = g.group.cartesian(Vgf)
a_mom = g.qcd.scalar.action.mass_term()
a_gauge = g.qcd.gauge.action.wilson(beta)
a_ld = gf.action_log_det_jacobian()
sympl = g.algorithms.integrator.symplectic

iq = sympl.update_q(Vgf, lambda: a_mom.gradient(U_mom, U_mom))
ip_g = sympl.update_p(U_mom, lambda: a_gauge.gradient(Vgf, Vgf))
ip_l = sympl.update_p(U_mom, lambda: a_ld.gradient(Vgf, Vgf))


def H():
    return a_gauge(Vgf) + a_mom(U_mom) + a_ld(Vgf)


# ---------------------------------------------------------------- thermalise
# The map is a gauge transformation, so the plaquette sector is completely
# insensitive to the log-det term -- thermalise it with plain Wilson HMC
# (seconds instead of an hour), then run a couple of full trajectories to let
# the gauge-orbit direction, which the log-det *does* see, relax.
gauge_only = sympl.leap_frog(20, ip_g, iq)
for i in range(20):
    rng.normal_element(U_mom)
    gauge_only(1.0)
g.message(f"after gauge-only pre-thermalisation: plaq = {g.qcd.gauge.plaquette(Vgf)}")

full = sympl.leap_frog(8, ip_l, sympl.leap_frog(1, ip_g, iq))
for i in range(2):
    rng.normal_element(U_mom)
    t0 = time.time()
    full(tau)
    g.message(
        f"therm {i}: plaq = {g.qcd.gauge.plaquette(Vgf)}, "
        f"log-det action = {a_ld(Vgf):.6f}  [{time.time()-t0:.1f} s]"
    )

V0 = g.copy(Vgf)
rng.normal_element(U_mom)
P0 = g.copy(U_mom)
g.message(f"start: plaq = {g.qcd.gauge.plaquette(Vgf)}, log-det action = {a_ld(Vgf):.6f}")


def reset():
    for mu in range(nd):
        Vgf[mu] @= V0[mu]
        U_mom[mu] @= P0[mu]


# ------------------------------------------------------------ dH ~ eps^2 scan
res = []
for n in [2, 4, 8, 16]:
    reset()
    h0 = H()
    t0 = time.time()
    sympl.leap_frog(n, ip_l, sympl.leap_frog(1, ip_g, iq))(tau)
    dH = H() - h0
    res.append((tau / n, abs(dH)))
    g.message(f"nsteps = {n:3d}  eps = {tau/n:.4f}  dH = {dH:+.6e}  [{time.time()-t0:.1f} s]")

x = np.log([r[0] for r in res])
y = np.log([r[1] for r in res])
g.message(f"fitted slope: |dH| ~ eps^{np.polyfit(x, y, 1)[0]:.3f}   (expect 2)")
for i in range(len(res) - 1):
    g.message(f"  ratio {res[i][1]/res[i+1][1]:.2f} over eps halving "
              f"(local slope {np.log2(res[i][1]/res[i+1][1]):.2f})")

# ------------------------------------------------------------- reversibility
reset()
mdint = sympl.leap_frog(4, ip_l, sympl.leap_frog(1, ip_g, iq))
mdint(tau)
for mu in range(nd):
    U_mom[mu] @= -U_mom[mu]
mdint(tau)
err = sum(g.norm2(g(Vgf[mu] - V0[mu])) for mu in range(nd)) / sum(g.norm2(v) for v in V0)
g.message(f"reversibility error: {err:.3e}")
