#!/usr/bin/env python3
#
# Standard HMC production run for the topological tunneling comparison.
# Wilson gauge action, leapfrog, Metropolis; Wilson-flowed topological
# charge and energy density measured every --nmeas trajectories;
# checkpoints every --nsave trajectories, auto-resumes from the latest
# checkpoint in --root.
#
# Counterpart: nambu_obc5d_topo_run.py (same measurements, same log
# format). Analysis: tau_int.py.
#
# Tune --nsteps so acceptance lands in 65-85%, or pass --autotune 1 to
# have the trajectory length tau tuned automatically (see below).
#
# Autotune (--autotune 1): at fixed --nsteps, adjust tau every
# --tune-window trajectories (non-overlapping windows) for the first
# --tune-ntraj trajectories, so the window-averaged acceptance lands in
# [--tune-lo, --tune-hi]. tau is multiplied by (1 -+ --tune-step) per
# window while out of band. Once --tune-ntraj is reached, tau is frozen
# for the remainder of the run (ordinary fixed-parameter HMC). The
# trajectories inside the tuning window are real Metropolis trajectories
# (not discarded), but tau varies within that stretch, so treat it like
# extended thermalization when computing tau_int/autocorrelations -
# start analysis at trajectory ntherm + tune-ntraj to be safe. tau's
# current value/tuning status is persisted to <root>/tau_state.txt so a
# walltime-killed and resumed job picks up exactly where it left off.
#
import gpt as g
import numpy as np
import time
from topo_measure import (
    flow_and_measure,
    measurement_line,
    latest_checkpoint,
    save_checkpoint,
    load_tau_state,
    save_tau_state,
)

# parameters
beta = g.default.get_float("--beta", 6.0)
L = g.default.get_int("--L", 8)
tau = g.default.get_float("--tau", 2.0)
n_steps = g.default.get_int("--nsteps", 50)
n_therm = g.default.get_int("--ntherm", 100)
n_traj = g.default.get_int("--ntraj", 100000)
n_meas = g.default.get_int("--nmeas", 1)
n_save = g.default.get_int("--nsave", 100)
t_flow = g.default.get_float("--tflow", 2.0)
eps_flow = g.default.get_float("--epsflow", 0.1)
root = g.default.get("--root", ".")
seed = g.default.get("--seed", "hmc-topo")

autotune = bool(g.default.get_int("--autotune", 0))
tune_window = g.default.get_int("--tune-window", 50)
tune_ntraj = g.default.get_int("--tune-ntraj", 300)
tune_lo = g.default.get_float("--tune-lo", 0.80)
tune_hi = g.default.get_float("--tune-hi", 0.90)
tune_step = g.default.get_float("--tune-step", 0.10)

tuning_done = not autotune
if autotune:
    state = load_tau_state(root)
    if state is not None:
        tau, tuning_done = state

grid = g.grid([L, L, L, L], g.double)

# resume from latest checkpoint if present
U = g.qcd.gauge.unit(grid)  # cold start at weak coupling
it0 = 0
latest = latest_checkpoint(root, n_traj)
if latest is not None:
    g.copy(U, g.load(f"{root}/ckpoint_lat.{latest}"))
    it0 = latest + 1
rng = g.random(f"{seed}-{it0}")

g.message(
    f"""
Standard HMC topological tunneling run
  lattice = {grid.fdimensions}
  beta    = {beta}
  tau     = {tau}, MD steps = {n_steps}  (eps = {tau / n_steps})
  ntherm  = {n_therm} (skipped on resume), ntraj = {n_traj}
  nmeas   = {n_meas}, tflow = {t_flow} (eps_flow = {eps_flow}, radius = {np.sqrt(8 * t_flow):.2f})
  nsave   = {n_save}, root = {root}
  resume  = {"trajectory " + str(it0) if it0 > 0 else "fresh start"}
  autotune = {autotune}{f" (window={tune_window}, ntraj={tune_ntraj}, target=[{tune_lo},{tune_hi}], step={tune_step}, tuning_done={tuning_done}, tau={tau:.4f})" if autotune else ""}
"""
)

P = g.group.cartesian(U)

a_kin = g.qcd.scalar.action.mass_term()
a_S = g.qcd.gauge.action.wilson(beta)

sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

ip = sympl.update_p(P, lambda: a_S.gradient(U, U))
iq = sympl.update_q(U, lambda: a_kin.gradient(P, P))
mdint = sympl.leap_frog(n_steps, ip, iq)


def hmc(tau, accept_reject=True):
    rng.normal_element(P)
    if not accept_reject:
        # thermalization from cold start: integrate without Metropolis
        mdint(tau)
        return [1.0, 0.0]
    accrej = metro(U)
    h0 = a_kin(P) + a_S(U)
    mdint(tau)
    h1 = a_kin(P) + a_S(U)
    return [accrej(h1, h0), h1 - h0]


if it0 == 0:
    g.message(f"Thermalizing {n_therm} trajectories (no accept/reject)")
    t0 = time.time()
    for i in range(n_therm):
        hmc(tau, accept_reject=False)
    g.message(
        f"Thermalized in {time.time() - t0:.1f} s: plaquette = {g.qcd.gauge.plaquette(U):.6f}"
    )

history = []
tune_hist = []
n_acc = 0
t0 = time.time()
for i in range(it0, n_traj):
    history += [hmc(tau)]
    n_acc += history[-1][0]
    g.message(
        f"Trajectory {i}: plaquette = {g.qcd.gauge.plaquette(U):.6f}, "
        f"dH = {history[-1][1]:+.4e}, accept = {n_acc / len(history):.3f}, tau = {tau:.4f}"
    )
    if autotune and not tuning_done:
        if i < tune_ntraj:
            tune_hist.append(history[-1][0])
            if len(tune_hist) >= tune_window:
                a = np.mean(tune_hist)
                if a < tune_lo:
                    tau *= 1 - tune_step
                elif a > tune_hi:
                    tau *= 1 + tune_step
                tune_hist = []
                g.message(f"[autotune] trajectory {i}: window acceptance = {a:.3f}, tau -> {tau:.4f}")
                save_tau_state(root, tau, tuning_done)
        if i + 1 >= tune_ntraj:
            tuning_done = True
            g.message(f"[autotune] tuning finished at trajectory {i}: tau frozen at {tau:.4f}")
            save_tau_state(root, tau, tuning_done)
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
