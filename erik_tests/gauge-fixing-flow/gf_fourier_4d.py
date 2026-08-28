#!/usr/bin/env python3
#
# Local-Jacobian one-step gauge-fixing flow HMC with the Zhao (2018)
# Fourier-accelerated kinetic term.
#
# Same setup as dH_4d_single.py (4^4, exact per-site 64x64 block log-det), with
# the trivial kinetic term  S_p = 1/2 <pi,pi>  replaced by
#
#     S_p = V * sum_k D^{mu,nu}(k) tr(P_mu(k)^dag P_nu(k)),
#     D = (1/s^2) P^T + (1/M^2) P^L,   s^2 = |k^|^2 + eps_fourier^2
#
# The kernel is the RWGFFA one -- build_sqrt_fourier_kernel from
# fourier_kernel.py, imported unchanged -- and it is handed to GPT's built-in
# g.qcd.scalar.action.fourier_mass_term, which takes sqrt(D) and derives
# D = sqrt(D)^2 and D^{-1/2} = inv(sqrt(D)) itself.  That built-in is the same
# construction as fourier_hmc.py's kinetic_energy/velocity/fill_fourier_momenta
# (g.adj(fft) and g.inv(fft) are literally the same function in GPT, and
# scale_unitary**2 == V), with two additions: it symmetrises the velocity onto
# the algebra, and its __call__/draw/gradient cannot disagree by construction.
#
import gpt as g
import numpy as np
import time
from gf_local_jacobian import gauge_fixing_step
from fourier_kernel import build_sqrt_fourier_kernel

nd, size, beta, eps_gf = 4, 4, 10.0, 1e-2
tau = g.default.get_float("--tau", 0.25)
nsteps = g.default.get_int("--nsteps", 8)
M = g.default.get_float("--M", 1.0)                       # longitudinal eigenvalue 1/M^2
eps_fourier = g.default.get_float("--eps_fourier", 0.5)   # IR regulator

grid = g.grid([size] * nd, g.double)
rng = g.random("t")

g.message(f"nd={nd}, L={size}, beta={beta}, eps_gf={eps_gf}, tau={tau}, nsteps={nsteps}")
g.message(f"Fourier kernel: M={M}, eps_fourier={eps_fourier}")

# sqrt(D) from the RWGFFA kernel, split into the nd x nd complex fields that
# fourier_mass_term expects
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

gf = gauge_fixing_step(U, eps_gf, checkerboard=g.even)
g.message(f"block size per active site: {gf.n_block} x {gf.n_block}")

Vgf = g.copy(U)
U_mom = g.group.cartesian(Vgf)

a_gauge = g.qcd.gauge.action.wilson(beta)
a_ld = gf.action_log_det_jacobian()

# the velocity fed to update_q is a_mom.gradient; check it really is the
# derivative of the kinetic energy a_mom evaluates (cheap, no log-det involved)
_p = g.group.cartesian(Vgf)
rng.normal_element(_p)
a_mom.assert_gradient_error(rng, _p, _p, 1e-3, 1e-8)

sympl = g.algorithms.integrator.symplectic
iq = sympl.update_q(Vgf, lambda: a_mom.gradient(U_mom, U_mom))
ip_g = sympl.update_p(U_mom, lambda: a_gauge.gradient(Vgf, Vgf))
ip_l = sympl.update_p(U_mom, lambda: a_ld.gradient(Vgf, Vgf))

# the map is a gauge transformation, so the plaquette sector cannot see the
# log-det -- pre-thermalise with plain Wilson HMC (no log-det force, seconds)
iq_std = sympl.update_q(Vgf, lambda: a_mom_std.gradient(U_mom, U_mom))
for i in range(20):
    rng.normal_element(U_mom)
    sympl.leap_frog(20, ip_g, iq_std)(1.0)
g.message(f"pre-thermalised: plaq = {g.qcd.gauge.plaquette(Vgf)}")

# draw momenta from exp(-pi^dag D pi)
a_mom.draw(U_mom, rng)
for mu in range(nd):
    g.message(f"  P[{mu}]: group defect = {g.group.defect(U_mom[mu]):.3e}")
g.message(f"drawn momenta: fourier KE = {a_mom(U_mom):.8f}, "
          f"trivial KE = {a_mom_std(U_mom):.8f}")

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
