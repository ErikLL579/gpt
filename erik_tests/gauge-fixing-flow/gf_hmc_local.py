#!/usr/bin/env python3
#
# One-step gauge-fixing flow HMC, local-Jacobian version of Trying_AD.py.
#
# Differences from Trying_AD.py:
#   - checkerboard mask is ON (the Jacobian is only block diagonal when it is)
#   - log-det comes from the per-site block determinant, so there is no mom2 /
#     no stochastic momentum refresh, and no FGCR in the log-det path
# Everything else -- map, beta, tau, nsteps, eps, integrator -- is unchanged.
#
import gpt as g
import numpy as np
import time
from gf_local_jacobian import gauge_fixing_step

nd = g.default.get_int("--nd", 2)
size = g.default.get_int("--L", 4)
beta = g.default.get_float("--beta", 10.0)
eps_gf = g.default.get_float("--eps", 1e-2)
tau = g.default.get_float("--tau", 1.0)
nsteps = g.default.get_int("--nsteps", 10)
ntraj = g.default.get_int("--ntraj", 5)
ntherm = g.default.get_int("--ntherm", 0)

grid = g.grid([size] * nd, g.double)
rng = g.random("t")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)


def trace_U(V):
    # mean Re Tr U / Nc over all links (normalisation fixed for general nd)
    tot = sum(float(np.sum(u[:].real)) for u in g.eval(g.trace(V)))
    return tot / (nd * grid.gsites) / V[0].otype.shape[0]


gf = gauge_fixing_step(U, eps_gf, checkerboard=g.even)

# dynamical variable: the map is applied to Vgf to give the physical field.
# One masked step is a small deformation, so Vgf = U is a fine starting point;
# no non-linear inverse solve is needed to *start* the chain.
Vgf = g.copy(U)

U_mom = g.group.cartesian(Vgf)
action_gauge_mom = g.qcd.scalar.action.mass_term()
action_gauge = g.qcd.gauge.action.wilson(beta)
a_log_det = gf.action_log_det_jacobian()

metro = g.algorithms.markov.metropolis(rng)
sympl = g.algorithms.integrator.symplectic
log = sympl.log()

g.message(f"Lattice {grid.fdimensions}, beta={beta}, eps_gf={eps_gf}")
g.message(f"Block size per active site: {gf.n_block} x {gf.n_block}")
g.message(f"Initial plaquette      : {g.qcd.gauge.plaquette(Vgf)}")
g.message(f"Initial log-det action : {a_log_det(Vgf)}")


def hamiltonian(draw):
    if draw:
        rng.normal_element(U_mom)
    s = action_gauge(Vgf)
    ald = a_log_det(Vgf)
    h = s + action_gauge_mom(U_mom) + ald
    if draw:
        g.message(f"    Gluonic action {s}, log-det action {ald}")
    return h, s


def log_det_force():
    return log(lambda: a_log_det.gradient(Vgf, Vgf), "log det")()


def gauge_force():
    return log(lambda: action_gauge.gradient(Vgf, Vgf), "gauge")()


iq = sympl.update_q(Vgf, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom"))
ip_gauge = sympl.update_p(U_mom, gauge_force)
ip_log_det = sympl.update_p(U_mom, log_det_force)

mdint = sympl.leap_frog(nsteps, ip_log_det, sympl.leap_frog(1, ip_gauge, iq))
g.message(f"Integration scheme:\n{mdint}")

no_accept_reject = False


def hmc(tau):
    accrej = metro(Vgf)
    h0, s0 = hamiltonian(True)
    mdint(tau)
    h1, s1 = hamiltonian(False)
    if no_accept_reject:
        return [True, s1 - s0, h1 - h0]
    return [accrej(h1, h0), s1 - s0, h1 - h0]


accept, total = 0, 0
plaq_measurements, dH_measurements = [], []
start = time.time()

for it in range(ntraj):
    no_accept_reject = it < ntherm
    t0 = time.time()
    a, dS, dH = hmc(tau)
    accept += a
    total += 1

    plaq = g.qcd.gauge.plaquette(Vgf)
    plaq_measurements.append(plaq)
    dH_measurements.append(dH)

    g.message(
        f"traj {it}: plaq = {plaq:.8f}, linktrace(Vgf) = {trace_U(Vgf):.8f}, "
        f"linktrace(ft Vgf) = {trace_U(gf(Vgf)):.8f}, dS = {dS:.6e}, dH = {dH:.6e}, "
        f"acc = {accept/total:.2f}, {time.time()-t0:.1f} s"
    )

g.message(f"Total time: {time.time() - start:.1f} seconds")
d = np.array(dH_measurements)
g.message(f"<dH> = {d.mean():.6e}, <|dH|> = {np.abs(d).mean():.6e}, <exp(-dH)> = {np.exp(-d).mean():.6f}")
p = np.array(plaq_measurements)
g.message(f"Avg plaq = {p.mean():.8f}, std = {p.std():.8f}")
