import gpt as g
import sys, os
import numpy as np

from scipy.linalg import expm
import time
import matplotlib.pyplot as plt
from gpt.qcd.gauge.smear import local_stout  
from gpt.ad import reverse as rad
from gpt.qcd.gauge.smear.differentiable import dft_diffeomorphism
import time
import random


size = 16
grid = g.grid([size, size], g.double)
rng = g.random("t")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)



def trace_U(U):
    return sum(v for v in sum(u[:].real for u in g.eval(g.trace(U)))) / (size* size*2) / 3.

num_steps = 1
def ftg(U, eps):
    global num_steps

    ##### dmuAmu ##############
    #B = U[0] -  g.adj(g.cshift(U[0], 0, -1))
    B = U[0] -  g.cshift(U[0], 0, -1)
    for mu in [1,]:
        #B += U[mu] -  g.adj(g.cshift(U[mu], mu, -1))
        B += U[mu] -  g.cshift(U[mu], mu, -1)

   #### masks for all even/odd sites ########
    """
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
    #fm = mask + 1e-15 * imask
    
    ###### apply masks ########
    #B *= fm # causes issues with action log det routine
    #B = g(B*fm)
    """
    
    # apply gtf 
    U_prime = []
    for mu in [0,1,]:
        U_mu_prime = g(
                g.matrix.exp(  - eps * g.qcd.gauge.project.traceless_anti_hermitian(B) )  
                * U[mu] * g.matrix.exp(  + eps * g.qcd.gauge.project.traceless_anti_hermitian( g.cshift(B, mu, +1) ) ) 
        )
        U_prime.append(U_mu_prime)
    
    return U_prime


def ft0(U):
    return ftg(U, eps = -5e-2)
    
#dft = dft_diffeomorphism(U, ft0)
fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic

dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,
    ft0,
    #g.algorithms.inverter.fgmres(eps=1e-15, maxiter=30, restartlen=10),
    #g.algorithms.inverter.fgmres(eps=1e-15, maxiter=30, restartlen=10),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.optimize.non_linear_cg(
        maxiter=100, eps=1e-15, step=1e-1, line_search=ls2, beta=fr
    ),
)

dfm = dft.diffeomorphism()
ald = dft.action_log_det_jacobian()


# conjugate momenta
U_mom = g.group.cartesian(U)
#U_mom = g.group.cartesian(U)

action_gauge_mom = g.qcd.scalar.action.mass_term()
action_gauge = g.qcd.gauge.action.wilson(10.0)

metro = g.algorithms.markov.metropolis(rng)
inv = g.algorithms.inverter
sympl = g.algorithms.integrator.symplectic

log = sympl.log()

pure_gauge = True



def hamiltonian(draw):

    if draw:
        rng.normal_element(U_mom)
        
        return action_gauge(U) + action_gauge_mom(U_mom)

    else:
        
        return action_gauge(U) + action_gauge_mom(U_mom)


def gauge_force():
    #g.message("Compute gauge force")
    x = log(lambda: action_gauge.gradient(U, U), "gauge")()
    return x



iq = sympl.update_q(
    U, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom")
)

ip_gauge = sympl.update_p(U_mom, gauge_force)

mdint = sympl.leap_frog(3, ip_gauge, iq)


g.message(f"Integration scheme:\n{mdint}")

tau = 1.0
#nsteps = 20



no_accept_reject = False

def hmc(tau):
    accrej = metro(U)
    g.message("After metro")

    h0 = hamiltonian(True)
    
    nsteps = 13
    for i in range(nsteps):
        mdint(tau/nsteps)

    g.message("After mdint(tau)")
    h1 = hamiltonian(False)
    g.message("After H(false)")
    
    if no_accept_reject:
        return [True, h1 - h0]
    else:
        return [accrej(h1, h0), h1 - h0]
   
accept, total = 0, 0

plaq_measurements_gf = []
trace_measurements = []

for it in range(20000):
    #pure_gauge = it < 10
    no_accept_reject = it < 10
    g.message(pure_gauge, no_accept_reject)

    a, dH = hmc(tau)
    accept += a
    total += 1

    plaq = g.qcd.gauge.plaquette(U)
    plaq_measurements_gf.append(plaq)
    trace_measurements.append(trace_U(U))

    g.message("////////////// PRE GAUGE FIXING//////////////////////")
    g.message(f"link trace = {trace_U(U)}")

    
    if it > 50:
        # gauge-fixing steps
        if random.randint(0, 1) == 0:
            g.message("////////////// FORWARD GAUGE FIXING//////////////////////")
            for i in range(4):
                Z = ft0(U)
                for mu in [0,1]:
                    U[mu] @= Z[mu]
                g.message(f"link trace = {trace_U(U)}")
                
    
        else:
            g.message("////////////// REVERSE GAUGE FIXING//////////////////////")
            for i in range(4):
                Z = dft.inverse(U)
                for mu in [0,1]:
                    U[mu] @= Z[mu]

                g.message(f"link trace = {trace_U(U)}")        
    
    g.message(f"HMC plaq = {plaq}, dH = {dH}, acceptance = {accept/total}")
    g.message(f"Timing:\n{log.time}")
    g.message(f"TRAJECTORY NUM {it}")


np.save("plaq_measurements_gf.npy", np.asarray(plaq_measurements_gf))
np.save("trace_measurements_gf.npy", np.asarray(trace_measurements ))
