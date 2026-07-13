#!/usr/bin/env python3
#
# Integrated autocorrelation time comparison for the topological
# tunneling runs (hmc_topo_run.py vs nambu_obc5d_topo_run.py).
#
# Parses "Measurement i: Q5LI = ..., Qclover = ..., E = ..., t2E = ..."
# and "Trajectory i: plaquette = ..." lines. tau_int via Madras-Sokal
# with automatic windowing (smallest W with W >= c * tau_int(W)),
# error = tau_int * sqrt(2 (2W + 1) / N). Also reports tunneling rate
# (fraction of measurements where round(Q) changes) and Q histogram
# integerness. Means with binned drop-one-bin jackknife.
#
# Usage: python3 tau_int.py hmc.log nhmc.log [c=6.0]
#
import re
import sys
import numpy as np


def parse_log(fn):
    data = {"Q5LI": [], "Qclover": [], "E": [], "t2E": [], "plaq": []}
    pat_m = re.compile(
        r"Measurement \d+: Q5LI = ([0-9.eE+-]+), Qclover = ([0-9.eE+-]+), "
        r"E = ([0-9.eE+-]+), t2E = ([0-9.eE+-]+)"
    )
    pat_t = re.compile(r"Trajectory \d+: plaquette = ([0-9.eE+-]+)")
    with open(fn) as f:
        for line in f:
            m = pat_m.search(line)
            if m:
                data["Q5LI"].append(float(m.group(1)))
                data["Qclover"].append(float(m.group(2)))
                data["E"].append(float(m.group(3)))
                data["t2E"].append(float(m.group(4)))
                continue
            m = pat_t.search(line)
            if m:
                data["plaq"].append(float(m.group(1)))
    return {k: np.array(v) for k, v in data.items()}


def tau_int(x, c=6.0):
    # Madras-Sokal integrated autocorrelation time with automatic window
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 20 or x.var() == 0:
        return np.nan, np.nan, 0
    xm = x - x.mean()
    # autocovariance via FFT
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xm, m)
    acov = np.fft.irfft(f * np.conj(f), m)[:n].real / np.arange(n, 0, -1)
    rho = acov / acov[0]
    tau = 0.5
    for W in range(1, n // 2):
        tau += rho[W]
        if W >= c * tau:
            err = tau * np.sqrt(2.0 * (2 * W + 1) / n)
            return tau, err, W
    return tau, np.nan, n // 2


def jackknife_binned(x, bin_size):
    n_bins = len(x) // bin_size
    bins = x[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    jk = (bins.sum() - bins) / (n_bins - 1)
    err = np.sqrt((n_bins - 1) / n_bins * ((jk - jk.mean()) ** 2).sum())
    return bins.mean(), err


def tunneling_rate(Q):
    s = np.rint(Q)
    return np.mean(s[1:] != s[:-1]) if len(s) > 1 else np.nan


def integerness(Q):
    return np.mean(np.abs(Q - np.rint(Q))) if len(Q) else np.nan


def analyze(name, d, c):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    n_meas = len(d["Q5LI"])
    print(f"  measurements = {n_meas}, trajectories = {len(d['plaq'])}")
    if n_meas == 0:
        return {}
    print(f"  Q integerness <|Q - round(Q)|>: 5LI = {integerness(d['Q5LI']):.4f}, "
          f"clover = {integerness(d['Qclover']):.4f}")
    print(f"  tunneling rate (round(Q5LI) changes) = {tunneling_rate(d['Q5LI']):.4f}")
    out = {}
    print(f"  {'observable':>10} {'mean':>12} {'tau_int':>10} {'err':>8} {'window':>7}")
    for key in ["Q5LI", "Qclover", "t2E", "plaq"]:
        x = d[key]
        if len(x) == 0:
            continue
        tau, terr, W = tau_int(x, c)
        bs = max(1, int(2 * tau)) if np.isfinite(tau) else 16
        mean, merr = jackknife_binned(x, bs) if len(x) // bs >= 10 else (x.mean(), np.nan)
        out[key] = (mean, merr, tau, terr)
        print(f"  {key:>10} {mean:>12.6f} {tau:>10.2f} {terr:>8.2f} {W:>7}")
    # tau_int of Q^2 (sector mobility even when <Q> = 0)
    tau, terr, W = tau_int(d["Q5LI"] ** 2, c)
    print(f"  {'Q5LI^2':>10} {'':>12} {tau:>10.2f} {terr:>8.2f} {W:>7}")
    out["Q2"] = (None, None, tau, terr)
    return out


c = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
f_hmc = sys.argv[1] if len(sys.argv) > 1 else "hmc_topo.log"
f_nhmc = sys.argv[2] if len(sys.argv) > 2 else "nambu_obc5d_topo.log"

r_hmc = analyze(f"Standard HMC ({f_hmc})", parse_log(f_hmc), c)
r_nhmc = analyze(f"Nambu HMC + 5D OBC ({f_nhmc})", parse_log(f_nhmc), c)

print(f"\n{'=' * 72}\nComparison (NHMC / HMC)\n{'=' * 72}")
for key in ["Q5LI", "Qclover", "Q2", "t2E", "plaq"]:
    if key in r_hmc and key in r_nhmc:
        t1, t2 = r_hmc[key][2], r_nhmc[key][2]
        if np.isfinite(t1) and np.isfinite(t2) and t1 > 0:
            print(f"  tau_int({key}): HMC = {t1:.2f}, NHMC = {t2:.2f}, ratio = {t2 / t1:.3f}")
for key in ["plaq", "t2E"]:
    if key in r_hmc and key in r_nhmc:
        m1, e1 = r_hmc[key][0], r_hmc[key][1]
        m2, e2 = r_nhmc[key][0], r_nhmc[key][1]
        if e1 and e2 and np.isfinite(e1) and np.isfinite(e2):
            sig = abs(m2 - m1) / np.sqrt(e1**2 + e2**2)
            print(f"  <{key}> consistency: {m1:.6f}({e1:.6f}) vs {m2:.6f}({e2:.6f}) -> {sig:.2f} sigma")
