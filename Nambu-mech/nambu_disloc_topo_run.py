#!/usr/bin/env python3
#
# Nambu HMC with the dislocation-gated nucleation amplifier in G:
# production run for the topological tunneling comparison (counterpart
# of hmc_topo_run.py / nambu_ampl_topo_run.py, same measurements and
# log format; analysis: tau_int.py). Scaling/conservation validated in
# nambu_disloc_scaling.py.
#
#   H = sum p^2/2 + sum r^2/2 + S_beta(U)      (Metropolis on H)
#   G = gamma sum r_a + c_p sum p^2/2 + c_r sum r^2/2 + lam * D_sm^2
#   D_sm = Q_sm(n_sm1) - Q_sm(n_sm2)   (smeared clover charges, stout rho)
#
# D_sm is the differentiable proxy of the dislocation detector
# |Qclover - Q5LI| (hmc_60.log: 10.6x baseline AT tunneling events,
# 7-9x elevated BEFORE — predictive). Depth pair (1, 3) chosen for
# cost: the AD force scales with total chain depth (~0.9 s/force set
# at 8^4 on 8 Mac threads vs 2.4 s for the (4, 10) pair). Tradeoff:
# the shallow charge is in the UV-noise regime (sign-flips between
# nsm = 1 and 2 on quiet configs), so the equilibrium D floor is ~9x
# larger than the (4, 10) pair's — the gate idles less cleanly; watch
# the D column for floor drift.
#
# Gate mechanism (F_G = 2 lam D_sm (F_Q1 - F_Q2)):
#   equilibrium (measured on beta=6 8^4 ckpoint_lat.0, July 20 2026):
#     D_sm ~ 0.07, |s| p99.9 = 5.7e-2 * lam; lam = 1.75 default =>
#     tail at 0.1, 7x below the window edge c_p = 0.75 => armed ~ 0,
#     and with gamma = 1 the idle dynamics is plain HMC (unarmed p*r
#     products are bounded rotors, no transport tax on the bulk)
#   nucleation attempt: D_sm rises and |grad D_sm| localizes at the
#     lump; those components enter s = F_G/F_H in (c_p, c_r), arm at
#     rate sqrt(AB), and the site gets directed energy. Floor arming is
#     rate-suppressed (in-window at the floor forces |F_H,a| ~ floor).
#
# EOM (update_p convention dst <- dst - eps*frc):
#   Udot_a = gamma p_a + (c_r - c_p) p_a r_a
#   frc_P  = gamma F_H + cw(c_r F_H - F_G, R)
#   frc_R  = cw(F_G - c_p F_H, P)
#
# The amplifier enters ONLY through G, so the marginal for U is exactly
# exp(-S_beta(U)). Wilson-flowed Q5LI/Qclover/E measured every --nmeas
# trajectories (the flowed |Qclover - Q5LI| in the log IS the proven
# detector — free cross-check of the D_sm gate); checkpoints + resume.
#
# Vital signs per trajectory: D = D_sm, armed%, rate mean, |s| p99.9.
#   - armed should sit at ~0 in equilibrium and spike WITH |D|
#   - |s| p99.9 is the lam knob: raise lam until attempt trajectories
#     (elevated |D|) reach the window, keep equilibrium tail below ~0.1
#   - THE diagnostic for this variant is acceptance conditioned on
#     elevated |D| (from the log): if attempt trajectories are
#     selectively rejected, the chain looks healthy but vetoes every
#     crossing. Shorten tau before touching eps if that happens.
#
# Seed from an equilibrated HMC checkpoint (cp .../hmc_60/ckpoint_lat.7X000
# $ROOT/ckpoint_lat.0) — skips --ntherm, every sample is equilibrium.
# Tune --nsteps for 65-85% acceptance; gamma = 1 should give dH ~ HMC's.
#
import gpt as g
import numpy as np
import time
from gpt.ad import reverse as rad
from gpt.core.group import differentiable_functional
from topo_measure import flow_and_measure, measurement_line, latest_checkpoint, save_checkpoint

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 8)
gamma = g.default.get_float("--gamma", 1.0)  # 1 = idle + supercharger
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
lam = g.default.get_float("--lam", 1.75)
rho = g.default.get_float("--rho", 0.12)
n_sm1 = g.default.get_int("--nsm1", 1)
n_sm2 = g.default.get_int("--nsm2", 3)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 50)
n_therm = g.default.get_int("--ntherm", 100)
n_traj = g.default.get_int("--ntraj", 100000)
n_meas = g.default.get_int("--nmeas", 1)
n_save = g.default.get_int("--nsave", 100)
t_flow = g.default.get_float("--tflow", 2.0)
eps_flow = g.default.get_float("--epsflow", 0.1)
root = g.default.get("--root", ".")
seed = g.default.get("--seed", "nambu-disloc-topo")

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
Nambu HMC with dislocation-gated amplifier G: topological tunneling run
  lattice = {grid.fdimensions}
  beta    = {beta}
  gamma   = {gamma}  (linear-in-r idle velocity; 1 = HMC idle)
  c_p     = {c_p}, c_r = {c_r}, lam = {lam}  (window in s = F_G/F_H: ({c_p}, {c_r}))
  rho     = {rho}, n_sm1 = {n_sm1}, n_sm2 = {n_sm2}
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

# smeared clover topological charges at the two depths
sm = g.qcd.gauge.smear.stout(rho=rho)


def make_qfunc(n):
    aU = [rad.node(g.copy(u)) for u in U]
    qf = g.qcd.gauge.differentiable_topology(aU).functional(*aU)
    for _ in range(n):
        qf = qf.transformed(sm)
    return qf


class disloc_functional(differentiable_functional):
    # g3(U) = lam * (Q_sm(n1) - Q_sm(n2))^2, gradient by exact chain
    # rule 2 lam D (F_Q1 - F_Q2) through the validated Q_sm gradients
    def __init__(self, q1, q2, lam):
        self.q1 = q1
        self.q2 = q2
        self.lam = lam

    def d_value(self, fields):
        return self.q1(fields) - self.q2(fields)

    def __call__(self, fields):
        d = self.d_value(fields)
        return self.lam * d * d

    def gradient(self, fields, dfields):
        assert dfields is fields or dfields == fields
        d = self.d_value(fields)
        F1 = self.q1.gradient(fields, fields)
        F2 = self.q2.gradient(fields, fields)
        return [g(2.0 * self.lam * d * (f1 - f2)) for f1, f2 in zip(F1, F2)]


a_g3 = disloc_functional(make_qfunc(n_sm1), make_qfunc(n_sm2), lam)

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


# forces recomputed only when U changes: frc_P and frc_R are called
# back-to-back at the same U in the nested leapfrog, and the reverse-AD
# stout chains dominate the cost. Fingerprint = inner products with
# fixed random probe fields (norm2 is BLIND to unitary U: |U|^2 = 3V
# identically); deterministic in U, so rejects/copies/reversal are
# handled without invalidation bookkeeping
_fc = {"key": None}
_fc_probe = [g.lattice(u) for u in U]
for _w in _fc_probe:
    rng.element(_w)


def forces():
    key = tuple(complex(g.sum(g.trace(g.adj(w) * u))) for w, u in zip(_fc_probe, U))
    if _fc["key"] != key:
        _fc["key"] = key
        _fc["F_H"] = a_S.gradient(U, U)
        _fc["F_G"] = a_g3.gradient(U, U)
    return _fc["F_H"], _fc["F_G"]


# PRURP (paper eq. 19): nested GPT leapfrogs
def frc_P():
    F_H, F_G = forces()
    F = [g(c_r * fh - fg) for fh, fg in zip(F_H, F_G)]
    return [g(gamma * fh + fr) for fh, fr in zip(F_H, cw_list(F, R))]


def frc_R():
    F_H, F_G = forces()
    F = [g(fg - c_p * fh) for fh, fg in zip(F_H, F_G)]
    return cw_list(F, P)


def vel_U():
    return [g(gamma * p + (c_r - c_p) * o) for p, o in zip(P, cw_list(P, R))]


ip_P = sympl.update_p(P, frc_P, tag="P")
ip_R = sympl.update_p(R, frc_R, tag="R")
iq_U = sympl.update_q(U, vel_U, tag="U")
mdint = sympl.leap_frog(n_steps, ip_P, sympl.leap_frog(1, ip_R, iq_U))


def hamiltonian_H():
    return a_kin(P) + a_kin(R) + a_S(U)


# gate vital signs: D_sm, fraction of components with AB > 0, their mean
# growth rate, and the |s| tail (lam tuning diagnostic)
def gate_vitals():
    F_H = a_S.gradient(U, U)
    F_G = a_g3.gradient(U, U)
    h, q = [], []
    for fh, fg in zip(F_H, F_G):
        for ch, cg in zip(fh.otype.coordinates(fh), fg.otype.coordinates(fg)):
            h.append(ch[:].real.flatten())
            q.append(cg[:].real.flatten())
    h, q = np.concatenate(h), np.concatenate(q)
    AB = (c_r * h - q) * (q - c_p * h)
    amp = AB > 0
    rate = np.sqrt(AB[amp]).mean() if amp.any() else 0.0
    ok = np.abs(h) > 1e-12
    s_tail = np.percentile(np.abs(q[ok] / h[ok]), 99.9)
    return a_g3.d_value(U), amp.mean(), rate, s_tail


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
    d, armed, rate, s_tail = gate_vitals()
    g.message(
        f"Trajectory {i}: plaquette = {g.qcd.gauge.plaquette(U):.6f}, "
        f"dH = {history[-1][1]:+.4e}, accept = {n_acc / len(history):.3f}, "
        f"D = {d:+.6f}, armed = {100 * armed:.3f}%, rate = {rate:.4f}, stail = {s_tail:.3e}"
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
