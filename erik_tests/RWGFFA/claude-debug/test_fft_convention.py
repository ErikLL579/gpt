"""
Test GPT's FFT convention to understand normalization.
"""
import gpt as g
import numpy as np

grid = g.grid([4, 4, 4, 4], g.double)
V = grid.gsites

g.message(f"Volume V = {V}")

# Create a simple test field: constant field = 1
f = g.complex(grid)
f[:] = 1.0

fft = g.fft()
fft_inv = g.adj(fft)

# Forward FFT of constant 1 should give V*delta(k,0)
f_k = g.eval(fft * f)

# Sum of |f_k|^2
sum_fk2 = g.sum(g.adj(f_k) * f_k).real
sum_f2 = g.sum(g.adj(f) * f).real
g.message(f"For constant f(x) = 1:")
g.message(f"  Σ|f(x)|² = {sum_f2}")
g.message(f"  Σ|f_k|² = {sum_fk2}")
g.message(f"  Ratio Σ|f_k|²/Σ|f|² = {sum_fk2/sum_f2}")

# Inverse FFT
f_back = g.eval(fft_inv * f_k)
sum_fback2 = g.sum(g.adj(f_back) * f_back).real
g.message(f"  After FFT then IFFT: Σ|f_back|² = {sum_fback2}")
g.message(f"  f_back/f ratio = {sum_fback2/sum_f2}")

# Now test with Gaussian noise - this is more informative
g.message("\n=== Gaussian noise test ===")
rng = g.random("test-fft")
eta = g.complex(grid)
rng.cnormal(eta)

eta_k = g.eval(fft * eta)
eta_back = g.eval(fft_inv * eta_k)

var_eta = g.sum(g.adj(eta) * eta).real / V
var_eta_k = g.sum(g.adj(eta_k) * eta_k).real / V
var_eta_back = g.sum(g.adj(eta_back) * eta_back).real / V

g.message(f"Var[η(x)] per site = {var_eta:.6f}")
g.message(f"Var[η(k)] per site = {var_eta_k:.6f}")
g.message(f"Var[η_back(x)] per site = {var_eta_back:.6f}")
g.message(f"Ratio Var[η(k)]/Var[η(x)] = {var_eta_k/var_eta:.6f}")
g.message(f"Ratio Var[η_back]/Var[η] = {var_eta_back/var_eta:.6f}")

# Parseval check
g.message(f"\nParseval theorem check:")
g.message(f"  Σ|η(x)|² = {V*var_eta:.6f}")
g.message(f"  Σ|η(k)|² = {V*var_eta_k:.6f}")
g.message(f"  If unitary FFT: these should be equal")
g.message(f"  If physics FFT: Σ|f_k|² = V * Σ|f|²")

# Check FFT * IFFT = identity or V*identity
diff = g.eval(eta_back - eta)
diff_norm = g.sum(g.adj(diff) * diff).real
g.message(f"\n||FFT^(-1)(FFT(η)) - η||² = {diff_norm:.2e}")

# Check if IFFT(FFT(f)) = f or V*f
ratio_back = g.eval(eta_back / g.component.real(eta))
# This is messy for complex, let's just check the scaling
g.message(f"\nScaling check:")
g.message(f"  If FFT*IFFT = 1: η_back should equal η")
g.message(f"  ||η||² = {V*var_eta:.6f}, ||η_back||² = {V*var_eta_back:.6f}")
