#!/usr/bin/env python3
# H conservation for the local-Jacobian one-step gauge-fixing HMC:
# thermalise, then run one trajectory of fixed tau at several step counts.
import gpt as g
import numpy as np
from gf_local_jacobian import gauge_fixing_step

nd, size, beta, eps_gf, tau = 2, 8, 10.0, 1e-2, 1.0
grid = g.grid([size] * nd, g.double)
rng = g.random("t")
U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

gf = gauge_fixing_step(U, eps_gf, checkerboard=g.even)
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


# thermalise
for i in range(4):
    rng.normal_element(U_mom)
    sympl.leap_frog(10, ip_l, sympl.leap_frog(1, ip_g, iq))(tau)
    g.message(f"therm {i}: plaq = {g.qcd.gauge.plaquette(Vgf)}")

V0 = g.copy(Vgf)
rng.normal_element(U_mom)
P0 = g.copy(U_mom)
g.message(f"start: plaq = {g.qcd.gauge.plaquette(Vgf)}")

res = []
for n in [4, 8, 16, 32]:
    for mu in range(nd):
        Vgf[mu] @= V0[mu]
        U_mom[mu] @= P0[mu]
    h0 = H()
    sympl.leap_frog(n, ip_l, sympl.leap_frog(1, ip_g, iq))(tau)
    dH = H() - h0
    res.append((tau / n, abs(dH)))
    g.message(f"nsteps = {n:3d}  eps = {tau/n:.4f}  dH = {dH:+.6e}")

x = np.log([r[0] for r in res])
y = np.log([r[1] for r in res])
g.message(f"fitted slope: |dH| ~ eps^{np.polyfit(x, y, 1)[0]:.3f}   (expect 2)")

# reversibility
for mu in range(nd):
    Vgf[mu] @= V0[mu]
    U_mom[mu] @= P0[mu]
mdint = sympl.leap_frog(16, ip_l, sympl.leap_frog(1, ip_g, iq))
mdint(tau)
for mu in range(nd):
    U_mom[mu] @= -U_mom[mu]
mdint(tau)
err = sum(g.norm2(g(Vgf[mu] - V0[mu])) for mu in range(nd)) / sum(g.norm2(v) for v in V0)
g.message(f"reversibility error: {err:.3e}")
