import gpt as g
import sys, os
import numpy as np

import time
from gpt.ad import reverse as rad
import time

size = 8
grid = g.grid([size, size], g.double)
rng = g.random("t")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

num_steps = 1
def ftg(U, eps):
    global num_steps

    ##### dmuAmu ##############
    B = U[0] -  g.cshift(U[0], 0, -1)
    for mu in [1,]:
        B += U[mu] -  g.cshift(U[mu], mu, -1)

    #### masks for all even/odd sites ########
    grid_cb = grid.checkerboarded(g.redblack)
    one_cb = g.complex(grid_cb)
    one_cb[:] = 1

    masks = {}
    for p in [g.even, g.odd]:
        m = g.complex(grid)
        m[:] = 0
        one_cb.checkerboard(p)
        g.set_checkerboard(m, one_cb)
        masks[p] = m

    if num_steps // 2 == 0:
        mask, imask = masks[g.odd], masks[g.odd.inv()]
    else:
        mask, imask = masks[g.even], masks[g.even.inv()]

    num_steps += 1
    fm = g(mask + 1e-15 * imask)

    # Create group-typed mask: fm(x) * Identity
    # This preserves the su_n_fundamental_group otype when multiplying B
    fm_group = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
    g.identity(fm_group)
    fm_group @= fm * fm_group
    B *= fm_group

    # apply gtf
    U_prime = []
    for mu in [0,1]:
        U_mu_prime = g(
                g.matrix.exp(  - eps * g.qcd.gauge.project.traceless_anti_hermitian(B) )
                * U[mu] * g.matrix.exp(  + eps * g.qcd.gauge.project.traceless_anti_hermitian( g.cshift(B, mu, +1) ) )
        )
        U_prime.append(U_mu_prime)

    return U_prime


def ft0(U):
    return ftg(U, eps=1e-2)

fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic

dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,
    ft0,
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.optimize.non_linear_cg(
        maxiter=100, eps=1e-15, step=1e-1, line_search=ls2, beta=fr
    ),
)

dfm = dft.diffeomorphism()

# FT
Vgf = dft.inverse(U)

# conjugate momenta
U_mom = g.group.cartesian(Vgf)

action_gauge_mom = g.qcd.scalar.action.mass_term()
action_gauge = g.qcd.gauge.action.wilson(10.0)

metro = g.algorithms.markov.metropolis(rng)
sympl = g.algorithms.integrator.symplectic

log = sympl.log()

a_log_det = dft.action_log_det_jacobian()

mom = [g.group.cartesian(v) for v in Vgf]
mom_prime = g.copy(mom)
rng.normal_element(mom_prime)

mom2 = dfm.jacobian(Vgf, U, mom_prime)
ald = a_log_det(Vgf + mom2)
g.message("Action log det jac:", ald)

def hamiltonian(draw):
    global mom2
    if draw:
        rng.normal_element(U_mom)

        mom = [g.group.cartesian(v) for v in Vgf]
        mom_prime = g.copy(mom)
        rng.normal_element(mom_prime)

        W = dfm(Vgf)
        mom2 = dfm.jacobian(Vgf, W, mom_prime)
        ald = a_log_det(Vgf + mom2)

        s = action_gauge(Vgf)

        g.message("Gluonic action", s)
        g.message("Log-det action", ald)

        h = s + action_gauge_mom(U_mom) + ald

    else:

        ald = a_log_det(Vgf + mom2)
        s = action_gauge(Vgf)

        h = s + action_gauge_mom(U_mom) + ald
    return h, s


def log_det_force():
    x = log(lambda: a_log_det.gradient(Vgf + mom2, Vgf), "log det")()
    return x

def gauge_force():
    x = log(lambda: action_gauge.gradient(Vgf, Vgf), "gauge")()
    return x


iq = sympl.update_q(
    Vgf, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom")
)

ip_gauge = sympl.update_p(U_mom, gauge_force)

ip_log_det = sympl.update_p(U_mom, log_det_force)

# integrator
mdint = sympl.leap_frog(1, ip_log_det, sympl.leap_frog(1, ip_gauge, iq))


g.message(f"Integration scheme:\n{mdint}")

tau = 1.0

no_accept_reject = False

def hmc(tau):
    accrej = metro(Vgf)
    g.message("After metro")

    h0, s0 = hamiltonian(True)

    nsteps = 10
    for i in range(nsteps):
        mdint(tau/nsteps)

    g.message("After mdint(tau)")
    h1, s1 = hamiltonian(False)
    g.message("After H(false)")

    if no_accept_reject:
        return [True, s1 - s0, h1 - h0]
    else:
        return [accrej(h1, h0), s1 - s0, h1 - h0]

accept, total = 0, 0

plaq_measurements = []

start = time.time()

trajs = 5
for it in range(trajs):
    no_accept_reject = it < 3
    g.message(f"no_accept_reject = {no_accept_reject}")

    a, dS, dH = hmc(tau)
    accept += a
    total += 1

    plaq = g.qcd.gauge.plaquette(Vgf)
    plaq_measurements.append(plaq)

    g.message(f"HMC plaq = {plaq}, dS = {dS}, dH = {dH}, acceptance = {accept/total}")
    g.message(f"Timing:\n{log.time}")
    g.message(f"TRAJECTORY NUM {it}")

end = time.time()
g.message(f"Total time: {end - start} seconds")

p = np.array(plaq_measurements)

discard_samps = 1
g.message(f"Avg plaq = {np.mean(p[discard_samps:])}, std = {np.std(p[discard_samps:])}")
