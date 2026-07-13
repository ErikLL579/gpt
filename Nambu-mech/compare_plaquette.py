#!/usr/bin/env python3
#
# Binned jackknife comparison of plaquette measurements from
# nambu_markov.py and standard_hmc.py log files.
#
# Errors: bin adjacent trajectories, drop-one-bin jackknife over the bins;
# sweep bin size to check the error plateaus (resolves autocorrelation).
#
import re
import sys
import numpy as np


def parse_log(fn):
    plaq = []
    pat = re.compile(r"Trajectory \d+: plaquette = ([0-9.eE+-]+)")
    with open(fn) as f:
        for line in f:
            m = pat.search(line)
            if m:
                plaq.append(float(m.group(1)))
    return np.array(plaq)


def jackknife_binned(x, bin_size):
    n_bins = len(x) // bin_size
    bins = x[: n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    total = bins.sum()
    jk = (total - bins) / (n_bins - 1)  # drop-one-bin means
    mean = bins.mean()
    err = np.sqrt((n_bins - 1) / n_bins * ((jk - jk.mean()) ** 2).sum())
    return mean, err, n_bins


def analyze(name, x):
    print(f"\n{name}: {len(x)} trajectories")
    print(f"  naive error (no binning): {x.std(ddof=1) / len(x) ** 0.5:.6f}")
    print(f"  {'bin':>4} {'n_bins':>7} {'mean':>10} {'jk err':>10}")
    results = {}
    for b in [1, 2, 4, 8, 16, 32, 64]:
        if len(x) // b < 10:
            break
        mean, err, n_bins = jackknife_binned(x, b)
        results[b] = (mean, err)
        print(f"  {b:>4} {n_bins:>7} {mean:>10.6f} {err:>10.6f}")
    return results


f_nambu = sys.argv[1] if len(sys.argv) > 1 else "nambu_1k.log"
f_hmc = sys.argv[2] if len(sys.argv) > 2 else "hmc_1k.log"
bin_final = int(sys.argv[3]) if len(sys.argv) > 3 else 16

nambu = parse_log(f_nambu)
hmc = parse_log(f_hmc)

r_nambu = analyze("Nambu HMC", nambu)
r_hmc = analyze("Standard HMC", hmc)

mn, en = r_nambu[bin_final]
mh, eh = r_hmc[bin_final]
diff = mn - mh
sigma = (en**2 + eh**2) ** 0.5

print(f"\nComparison at bin size {bin_final}:")
print(f"  Nambu    <plaq> = {mn:.6f} +- {en:.6f}   (<1-plaq> = {1 - mn:.6f})")
print(f"  Standard <plaq> = {mh:.6f} +- {eh:.6f}   (<1-plaq> = {1 - mh:.6f})")
print(f"  difference = {diff:+.6f} +- {sigma:.6f}  ({abs(diff) / sigma:.2f} sigma)")
