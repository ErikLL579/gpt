#!/usr/bin/env python3
#
# Nambu HMC with the scale-selective smeared-Wilson amplifier in G:
# production run for the topological tunneling comparison (counterpart
# of hmc_topo_run.py / nambu_obc5d_linear_topo_run.py /
# nambu_smq_topo_run.py, same measurements and log format; analysis:
# tau_int.py). Scaling/conservation validated in nambu_ampl_scaling.py
# (July 16 2026: gradient 7.2e-11, reversibility ~5e-15, dH ~ eps^2.30,
# dG ~ eps^2.28).
#
#   H = sum p^2/2 + sum r^2/2 + S_beta(U)      (Metropolis on H)
#   G = c_p sum p^2/2 + c_r sum r^2/2 + lam * S_sm(U)
#   S_sm = Wilson action at the same beta on n_sm stout-smeared links
#
# Per adjoint component the (p,r) pair is a parametric amplifier
# (exponential energy concentration, directed velocity) iff the force
# ratio s = F_sm/F_H lies in the window (c_p/lam, c_r/lam); smooth/IR
# components are armed, UV components precess as bounded rotors.
# Window candidates from the fair-cost scan (wscan_b6.log, 8^4 beta=6):
#   conservative: --c_p 0.75  --c_r 1.5   --lam 1.125   (7.6% armed)
#   middle:       --c_p 0.685 --c_r 1.643 --lam 1.369  (11.9% armed)
#   aggressive:   --c_p 0.581 --c_r 1.936 --lam 1.453  (17.4% armed)
#
# EOM (update_p convention dst <- dst - eps*frc):
#   Udot_a = (c_r - c_p) p_a r_a
#   frc_P  = cw(c_r F_H - lam F_sm, R)
#   frc_R  = cw(lam F_sm - c_p F_H, P)
#
# The amplifier term enters ONLY through G, so the marginal for U is
# exactly exp(-S_beta(U)). Wilson-flowed Q and E measured every --nmeas
# trajectories; checkpoints every --nsave; auto-resume. Per-trajectory
# log includes the window occupancy and mean growth rate (the
# mechanism's vital signs).
#
# Tune --nsteps so acceptance lands in 65-85%.
#
import gpt as g
import numpy as np
import time
from topo_measure import flow_and_measure, measurement_line, latest_checkpoint, save_checkpoint

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 8)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
lam = g.default.get_float("--lam", 1.125)
rho = g.default.get_float("--rho", 0.12)
n_sm = g.default.get_int("--nsm", 2)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 50)
n_therm = g.default.get_int("--ntherm", 100)
n_traj = g.default.get_int("--ntraj", 100000)
n_meas = g.default.get_int("--nmeas", 1)
n_save = g.default.get_int("--nsave", 100)
t_flow = g.default.get_float("--tflow", 2.0)
eps_flow = g.default.get_float("--epsflow", 0.1)
root = g.default.get("--root", ".")
seed = g.default.get("--seed", "nambu-ampl-topo")

grid = g.grid([L, L, L, L], g.double)

# resume from latest checkpoint if present
U = g.qcd.gauge.unit(grid)  # cold start at weak coupling
it0 = 0
latest = latest_checkpoint(root, n_traj)
if latest is not None:
    g.copy(U, g.load(f"{root}/ckpoint_lat.{latest}"))
    it0 = latest + 1
rng = g.random(f"{seed}-{it0}")

if it0 == 0:
    # kick infinitesimally off the exact identity: the stout jacobian in
    # the smeared force is 0/0-singular (Cayley-Hamilton coefficients)
    # at exactly unit links
    kick = g.group.cartesian(U)
    rng.normal_element(kick)
    for mu in range(len(U)):
        U[mu] @= g(g.matrix.exp(g(1e-2 * kick[mu])) * U[mu])

g.message(
    f"""
Nambu HMC with smeared-Wilson amplifier G: topological tunneling run
  lattice = {grid.fdimensions}
  beta    = {beta}
  c_p     = {c_p}, c_r = {c_r}, lam = {lam}  (window in s: [{c_p / lam:.3f}, {c_r / lam:.3f}])
  rho     = {rho}, n_sm = {n_sm}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm} (skipped on resume), ntraj = {n_traj}
  nmeas   = {n_meas}, tflow = {t_flow} (eps_flow = {eps_flow}, radius = {np.sqrt(8 * t_flow):.2f})
  nsave   = {n_save}, root = {root}
  resume  = {"trajectory " + str(it0) if it0 > 0 else "fresh start"}
"""
)

nd = len(U)
P = g.group.cartesian(U)
R = g.group.cartesian(U)

a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

# smeared Wilson action: S_sm(U) = S_beta(stout^n_sm(U)), gradient via
# stout jacobian chain rule
a_Ssm = g.qcd.gauge.action.wilson(beta)
sm = g.qcd.gauge.smear.stout(rho=rho)
for _ in range(n_sm):
    a_Ssm = a_Ssm.transformed(sm)

sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)


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


# PRURP (paper eq. 19): nested GPT leapfrogs
def frc_P():
    F_H = a_S.gradient(U, U)
    F_G = a_Ssm.gradient(U, U)
    F = [g(c_r * fh - lam * fg) for fh, fg in zip(F_H, F_G)]
    return cw_list(F, R)


def frc_R():
    F_H = a_S.gradient(U, U)
    F_G = a_Ssm.gradient(U, U)
    F = [g(lam * fg - c_p * fh) for fh, fg in zip(F_H, F_G)]
    return cw_list(F, P)


def vel_U():
    return [g((c_r - c_p) * o) for o in cw_list(P, R)]


ip_P = sympl.update_p(P, frc_P, tag="P")
ip_R = sympl.update_p(R, frc_R, tag="R")
iq_U = sympl.update_q(U, vel_U, tag="U")
mdint = sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


# amplifier-window vital signs: fraction of components with AB > 0 and
# their mean growth rate sqrt(AB)
def window_vitals():
    F_H = a_S.gradient(U, U)
    F_G = a_Ssm.gradient(U, U)
    n_tot = 0
    n_amp = 0
    rate_sum = 0.0
    for fh, fg in zip(F_H, F_G):
        for ch, cg in zip(fh.otype.coordinates(fh), fg.otype.coordinates(fg)):
            h = ch[:].real.flatten()
            q = lam * cg[:].real.flatten()
            AB = (c_r * h - q) * (q - c_p * h)
            n_tot += len(AB)
            n_amp += int((AB > 0).sum())
            rate_sum += np.sqrt(AB[AB > 0]).sum()
    return n_amp / n_tot, rate_sum / max(n_amp, 1)


def nambu_hmc(tau, accept_reject=True):
    rng.normal_element(P)
    rng.normal_element(R)
    if not accept_reject:
        # thermalization from cold start: integrate without Metropolis
        mdint(tau)
        return [1.0, 0.0]
    accrej = metro(U)
    h0 = hamiltonian_H()
    mdint(tau)
    h1 = hamiltonian_H()
    return [accrej(h1, h0), h1 - h0]


if it0 == 0:
    g.message(f"Thermalizing {n_therm} trajectories (no accept/reject)")
    t0 = time.time()
    for i in range(n_therm):
        nambu_hmc(tau, accept_reject=False)
    g.message(
        f"Thermalized in {time.time() - t0:.1f} s: plaquette = {g.qcd.gauge.plaquette(U):.6f}"
    )

history = []
n_acc = 0
t0 = time.time()
for i in range(it0, n_traj):
    history += [nambu_hmc(tau)]
    n_acc += history[-1][0]
    armed, rate = window_vitals()
    g.message(
        f"Trajectory {i}: plaquette = {g.qcd.gauge.plaquette(U):.6f}, "
        f"dH = {history[-1][1]:+.4e}, accept = {n_acc / len(history):.3f}, "
        f"armed = {100 * armed:.2f}%, rate = {rate:.4f}"
    )
    if i % n_meas == 0:
        m = flow_and_measure(U, t_flow, eps_flow)
        g.message(measurement_line(i, m))
    if i % n_save == 0:
        save_checkpoint(root, i, U)
    if len(history) % 100 == 0:
        h = np.array(history)
        g.message(
            f"Running stats after {len(history)} trajectories: "
            f"acceptance = {np.mean(h[:, 0]):.2f}, <|dH|> = {np.mean(np.abs(h[:, 1])):.4e}, "
            f"{(time.time() - t0) / len(history):.2f} s / trajectory"
        )
