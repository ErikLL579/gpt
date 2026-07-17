#!/usr/bin/env python3
#
# Amplifier-window diagnostics for the smeared-Wilson Nambu HMC
# (nambu_ampl_scaling.py): per-component histogram of the
# smeared-to-raw force ratio
#
#   s_a = F_sm,a / F_H,a
#
# on stored gauge configurations, for several stout smearing depths.
# A component is in the amplifier window iff AB > 0 with
#   A = c_r F_H - lam F_sm,  B = lam F_sm - c_p F_H
# i.e. iff s_a in (c_p/lam, c_r/lam). The histogram shows where the UV
# bulk (s ~ 0) and the smooth tail (s ~ 1) sit, so the window edges can
# be placed in the valley between them.
#
# Usage (on stored checkpoints, e.g. on Perlmutter):
#   python3 window_histogram.py --root $SCRATCH/nambu/hmc_60 \
#       --configs 70000,71000,72000 --beta 6.0 --nsmlist 1,2,3,4
#
# Output per config and smearing depth: window occupancy, growth-rate
# stats (units of MD time; e-folds per trajectory = rate * tau), s
# percentiles, and an ASCII histogram of s. "HIST" lines are
# machine-parseable: HIST <nsm> <bin_center> <count>.
#
import gpt as g
import numpy as np

# parameters
root = g.default.get("--root", ".")
configs = [int(x) for x in g.default.get("--configs", "0").split(",")]
beta = g.default.get_float("--beta", 6.0)
c_p = g.default.get_float("--c_p", 0.75)
c_r = g.default.get_float("--c_r", 1.5)
lam = g.default.get_float("--lam", 1.125)
rho = g.default.get_float("--rho", 0.12)
nsm_list = [int(x) for x in g.default.get("--nsmlist", "1,2,3,4").split(",")]
tau = g.default.get_float("--tau", 2.0)

s_lo, s_hi, n_bins = -0.5, 2.0, 50

g.message(
    f"""
Amplifier-window histogram
  root    = {root}, configs = {configs}
  beta    = {beta}
  c_p     = {c_p}, c_r = {c_r}, lam = {lam}
  window in s = F_sm/F_H: ({c_p / lam:.3f}, {c_r / lam:.3f})
  rho     = {rho}, nsm list = {nsm_list}
  tau     = {tau} (for e-folds per trajectory)
"""
)

a_S = g.qcd.gauge.action.wilson(beta)
sm = g.qcd.gauge.smear.stout(rho=rho)


def components(F):
    # concatenate the 8 adjoint coefficient fields of all 4 links
    out = []
    for f in F:
        for c in f.otype.coordinates(f):
            out.append(c[:].real.flatten())
    return np.concatenate(out)


def analyze(U, nsm, hist_accum):
    a_Ssm = g.qcd.gauge.action.wilson(beta)
    for _ in range(nsm):
        a_Ssm = a_Ssm.transformed(sm)

    h = components(a_S.gradient(U, U))
    q = lam * components(a_Ssm.gradient(U, U))

    A = c_r * h - q
    B = q - c_p * h
    AB = A * B
    amp = AB > 0
    rates = np.sqrt(AB[amp])

    # ratio histogram (guard tiny F_H; window occupancy above is exact)
    ok = np.abs(h) > 1e-12
    s = q[ok] / h[ok] / lam
    counts, edges = np.histogram(np.clip(s, s_lo, s_hi), bins=n_bins, range=(s_lo, s_hi))
    hist_accum += counts

    pct = np.percentile(s, [5, 25, 50, 75, 95])
    g.message(
        f"  nsm = {nsm}: window occupancy = {100 * amp.mean():.2f}%  "
        f"rate mean/max = {rates.mean() if len(rates) else 0:.4f}/{rates.max() if len(rates) else 0:.4f}  "
        f"e-folds/traj mean = {tau * (rates.mean() if len(rates) else 0):.2f}  "
        f"s pct[5,25,50,75,95] = [{', '.join(f'{x:.3f}' for x in pct)}]"
    )
    return edges


def print_hist(nsm, counts, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = counts.max() if counts.max() > 0 else 1
    g.message(f"  s-histogram, nsm = {nsm} (window = [{c_p / lam:.3f}, {c_r / lam:.3f}]):")
    for c, n in zip(centers, counts):
        bar = "#" * int(round(60 * n / peak))
        mark = "|" if c_p / lam <= c <= c_r / lam else " "
        g.message(f"  HIST {nsm} {c:+.3f} {n:8d} {mark}{bar}")


hists = {nsm: np.zeros(n_bins, dtype=int) for nsm in nsm_list}
edges = None

for n in configs:
    fn = f"{root}/ckpoint_lat.{n}"
    g.message(f"Config {fn}")
    U = g.load(fn)
    g.message(f"  plaquette = {g.qcd.gauge.plaquette(U):.6f}")
    for nsm in nsm_list:
        edges = analyze(U, nsm, hists[nsm])

g.message("=" * 70)
g.message(f"Aggregated over {len(configs)} config(s):")
for nsm in nsm_list:
    print_hist(nsm, hists[nsm], edges)

g.message("Done")
