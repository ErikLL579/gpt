"""
Simple test: D * sqrt(D^{-1}) * sqrt(D^{-1}) = I
"""
import gpt as g
import numpy as np

from fourier_hmc import compute_D_components, compute_sqrt_Dinv_components

size = 4
grid = g.grid([size, size, size, size], g.double)
nd = grid.nd

M = 1.0
eps = 0.5

g.message(f"Building kernels with M={M}, eps={eps}...")

D = compute_D_components(grid, M, eps)
sqrtDinv = compute_sqrt_Dinv_components(grid, M, eps)

# Compute (D * sqrtDinv * sqrtDinv)_{mu,nu} = sum_rho,sigma D_{mu,rho} * sqrtDinv_{rho,sigma} * sqrtDinv_{sigma,nu}
# This should equal delta_{mu,nu}

g.message("\nComputing D * D^{-1/2} * D^{-1/2}...")

result = {}
for mu in range(nd):
    for nu in range(nd):
        # (D @ sqrtDinv @ sqrtDinv)_{mu,nu} = sum_{rho,sigma} D_{mu,rho} * sqrtDinv_{rho,sigma} * sqrtDinv_{sigma,nu}
        tmp = g.lattice(D[(0,0)])
        tmp[:] = 0
        for rho in range(nd):
            for sigma in range(nd):
                tmp += g.eval(D[(mu, rho)] * sqrtDinv[(rho, sigma)] * sqrtDinv[(sigma, nu)])
        result[(mu, nu)] = tmp

# Check diagonal elements (should be 1)
g.message("\nDiagonal elements (should be 1):")
for mu in range(nd):
    val = g.sum(result[(mu, mu)]) / grid.gsites
    g.message(f"  [{mu},{mu}]: avg = {val}")

# Check off-diagonal elements (should be 0)
g.message("\nOff-diagonal elements (should be 0):")
for mu in range(nd):
    for nu in range(nd):
        if mu != nu:
            val = g.sum(result[(mu, nu)]) / grid.gsites
            g.message(f"  [{mu},{nu}]: avg = {val}")

# Check max deviation from identity
g.message("\nMax deviation from identity at each site:")
for mu in range(nd):
    for nu in range(nd):
        if mu == nu:
            diff = g.eval(result[(mu, nu)] - 1.0)
        else:
            diff = result[(mu, nu)]

        # Get max absolute value
        diff_sq = g.eval(g.adj(diff) * diff)
        max_diff_sq = g.sum(diff_sq).real
        rms = np.sqrt(max_diff_sq / grid.gsites)
        g.message(f"  [{mu},{nu}]: RMS deviation = {rms:.2e}")
