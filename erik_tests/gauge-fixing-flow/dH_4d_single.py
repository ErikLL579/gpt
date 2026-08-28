#!/usr/bin/env python3
# Single short trajectory (tau = 0.25) of the local-Jacobian one-step
# gauge-fixing HMC in 4D (4^4, 64x64 block per active site).
import gpt as g
import numpy as np
import time
from gf_local_jacobian import gauge_fixing_step

nd, size, beta, eps_gf = 4, 4, 10.0, 1e-2
tau = g.default.get_float("--tau", 0.25)
nsteps = g.default.get_int("--nsteps", 8)

grid = g.grid([size] * nd, g.double)
rng = g.random("t")
U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

gf = gauge_fixing_step(U, eps_gf, checkerboard=g.even)
g.message(f"nd={nd}, L={size}, beta={beta}, eps_gf={eps_gf}, tau={tau}, nsteps={nsteps}")
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

# the map is a gauge transformation, so the plaquette sector cannot see the
# log-det -- pre-thermalise it with plain Wilson HMC (seconds, no log-det force)
for i in range(20):
    rng.normal_element(U_mom)
    sympl.leap_frog(20, ip_g, iq)(1.0)
g.message(f"pre-thermalised: plaq = {g.qcd.gauge.plaquette(Vgf)}")

rng.normal_element(U_mom)
s0, ld0, m0 = a_gauge(Vgf), a_ld(Vgf), a_mom(U_mom)
g.message(f"H0: gauge = {s0:.8f}, log-det = {ld0:.8f}, mom = {m0:.8f}")

t0 = time.time()
sympl.leap_frog(nsteps, ip_l, sympl.leap_frog(1, ip_g, iq))(tau)
dt = time.time() - t0

s1, ld1, m1 = a_gauge(Vgf), a_ld(Vgf), a_mom(U_mom)
g.message(f"H1: gauge = {s1:.8f}, log-det = {ld1:.8f}, mom = {m1:.8f}")
g.message(f"dS_gauge = {s1-s0:+.6e}, dS_logdet = {ld1-ld0:+.6e}, dS_mom = {m1-m0:+.6e}")
g.message(f"dH = {(s1+ld1+m1)-(s0+ld0+m0):+.6e}   [trajectory {dt:.1f} s]")
g.message(f"final plaq = {g.qcd.gauge.plaquette(Vgf)}")
