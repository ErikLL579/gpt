# RWGFFA - Random Walk Gauge Fixing with Fourier Acceleration

This directory contains implementations of Fourier-accelerated HMC for lattice QCD with optional Landau gauge fixing, based on Zhao (2018).

## Project Status

The Fourier kernel implementation has been debugged and verified. Key fix applied (January 2025): the longitudinal projector P^L must be normalized by |k̂|², NOT by s² = |k̂|² + ε². The ε regulator only appears in eigenvalues, not in projector normalization.

### Verified (February 2025)
- **Kernel identity**: D × D^{-1/2} × D^{-1/2} = I verified at ALL 256 lattice sites on 4^4 grid to machine precision (max deviation ~7.8e-16)
- **Velocity in Lie algebra**: v = IFFT[D·FFT[P]] is Hermitian and traceless to machine precision (~3e-16 relative deviation)

### Open Issues
- Gauge fixing reversibility needs theoretical investigation. The forward/reverse gauge fixing mechanism using `dft.inverse()` requires further study.

## File Overview

### Core Files (required for running tests)
- `fourier_kernel.py` - Builds D_{μν}(k), sqrt(D), and D^{-1/2} as nd×nd matrix lattices
- `fourier_hmc.py` - Provides `compute_D_components()` and `compute_sqrt_Dinv_components()` as scalar field dictionaries, plus momentum generation and kinetic energy functions

### Test Files
- `fourier_hmc_test_nogf.py` - Fourier HMC without gauge fixing (for validating kernel correctness)
- `fourier_hmc_gf_test.py` - Fourier HMC with 50/50 forward/reverse gauge fixing
- `gf_test.py` - Tests gauge fixing reversibility: 100 HMC trajectories, then 10 forward + 10 reverse GF steps, compares link-by-link

### Debug Files (in `claude-debug-mom-draw/`)
- `test_kernel_simple.py` - Tests D × D^{-1/2} × D^{-1/2} = I
- `test_projector.py` - Tests P^L × P^L = P^L projector property
- `fourier_kernel_fixed.py` - Reference implementation with correct projector

## Physics Background

### Fourier Acceleration (Zhao 2018)
The Fourier kernel D_{μν}(k) accelerates HMC by preconditioning the kinetic term:
- Standard HMC: H_p = tr(P† P)
- Fourier HMC: H_p = Σ_k tr(P†(k) D(k) P(k))

D has eigenvalue decomposition:
- D = (1/s²) P^T + (1/M²) P^L
- s² = |k̂|² + ε² where |k̂|² = Σ_μ sin²(k_μ/2)
- P^T = transverse projector (3 modes), P^L = longitudinal projector (1 mode)
- M = gauge fixing mass parameter, ε = IR regulator

### Critical Formula for Projectors
```
k̂_μ = e^{-i k_μ/2} sin(k_μ/2) = sin(k_μ/2) cos(k_μ/2) - i sin²(k_μ/2)

P^L_{μν} = k̂_μ k̂*_ν / |k̂|²    (NOT / s²!)

P^T_{μν} = δ_{μν} - P^L_{μν}
```

IMPORTANT: P^L must satisfy P^L × P^L = P^L. This only works if normalized by |k̂|², not s².

### Momentum Generation
To sample P from exp(-P† D P):
1. Generate η ~ N(0,1) in position space (Hermitian, traceless)
2. FFT to momentum space: η(k)
3. Apply D^{-1/2}: P(k) = D^{-1/2}(k) · η(k)
4. IFFT back: P(x)

D^{-1/2} = s P^T + M P^L = s I + (M - s) P^L

### Gauge Fixing
The 50/50 algorithm randomly applies forward or reverse gauge fixing after each trajectory:
- Forward: ftg(U, eps) applies gauge transformation toward Landau gauge
- Reverse: Uses differentiable_field_transformation.inverse() to undo gauge fixing

## GPT Library Reference

### Grid and Field Creation
```python
grid = g.grid([Lx, Ly, Lz, Lt], g.double)  # Create 4D lattice
V = grid.gsites      # Total number of sites
nd = grid.nd         # Number of dimensions (4)
L = grid.fdimensions # [Lx, Ly, Lz, Lt]

# Scalar fields
phi = g.complex(grid)           # Complex scalar at each site
phi = g.lattice(grid, g.ot_complex_additive_group())  # Equivalent

# Matrix fields
D = g.mcomplex(grid, nd)        # nd×nd complex matrix at each site

# Gauge fields
U = g.qcd.gauge.unit(grid)      # Unit gauge field (list of 4 SU(3) matrices)
P = g.group.cartesian(U)        # Lie algebra elements (momenta)
```

### Coordinates and Field Assignment
```python
coord_array = g.coordinates(phi)  # Shape (n_sites, nd), gives [x,y,z,t] for each site
phi[coord_array] = values         # Assign numpy array to lattice
phi[x, y, z, t]                   # Access single site (returns gpt tensor)
phi[x, y, z, t].array             # Get numpy array from tensor
```

### FFT
```python
fft = g.fft()
phi_k = g.eval(fft * phi)        # Forward FFT
phi_x = g.eval(g.adj(fft) * phi_k)  # Inverse FFT
```

### Field Operations
```python
g.eval(expr)                     # Evaluate expression, return new lattice
g.copy(phi)                      # Deep copy
phi @= g.eval(expr)              # In-place assignment
g.adj(phi)                       # Hermitian conjugate
g.trace(M)                       # Trace (returns scalar field for matrix field)
g.sum(phi)                       # Sum over all sites (returns Python scalar)
g.cshift(phi, mu, +1)            # Shift field in direction mu by +1
```

### Gauge Operations
```python
g.qcd.gauge.plaquette(U)                    # Average plaquette
g.qcd.gauge.energy_density(U)               # Energy density
g.qcd.gauge.smear.wilson_flow(U, epsilon)   # One Wilson flow step
g.matrix.exp(A)                             # Matrix exponential
g.qcd.gauge.project.traceless_anti_hermitian(A)  # Project to su(N)
```

### Actions and HMC
```python
wilson = g.qcd.gauge.action.wilson(beta)
S = wilson(U)                              # Evaluate action
dS = wilson.gradient(U, U)                 # Gradient w.r.t. U

sympl = g.algorithms.integrator.symplectic
ip = sympl.update_p(P, lambda: wilson.gradient(U, U))  # Momentum update
iq = sympl.update_q(U, lambda: velocity_func(P))       # Position update
mdint = sympl.leap_frog(n_steps, ip, iq)
mdint(tau)                                 # Run trajectory

metro = g.algorithms.markov.metropolis(rng)
accrej = metro(U)                          # Checkpoint U
accepted = accrej(H_final, H_initial)      # Accept/reject, restores U if rejected
```

### Random Numbers
```python
rng = g.random("seed-string")
rng.normal_element(P)            # Fill P with Gaussian Lie algebra elements
```

### Gauge Fixing Infrastructure
```python
fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic

dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,
    ft0,  # Forward transformation function
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10),
    g.algorithms.optimize.non_linear_cg(maxiter=100, eps=1e-15, step=1e-1, line_search=ls2, beta=fr),
)

Z = dft.inverse(U)  # Inverse gauge transformation
```

## Testing and Verification

### Kernel identity test (all sites)
Run from RWGFFA directory with PYTHONPATH set:
```bash
python3 claude-debug-mom-draw/test_kernel_simple.py
```
This tests D × D^{-1/2} × D^{-1/2} = I at every lattice site. Expected: max deviation ~1e-16.

### Velocity Hermiticity test
The velocity v = IFFT[D·FFT[P]] must be in the Lie algebra (Hermitian, traceless). Check with:
```python
v = velocity(P)
for mu in range(nd):
    diff = g.eval(v[mu] - g.adj(v[mu]))
    diff_norm = g.sum(g.trace(g.adj(diff) * diff)).real
    # Should be ~1e-29 (machine precision squared)
```

### Gauge fixing reversibility test
```bash
python3 gf_test.py
```
Runs 100 HMC trajectories, then 10 forward GF steps + 10 reverse GF steps. Compares ||U[mu] - U_orig[mu]|| for each direction.

## Common Parameters

- `M = 1.0` - Gauge fixing mass (longitudinal mode eigenvalue is 1/M²)
- `eps_fourier = 0.5` - IR regulator (transverse eigenvalue at k=0 is 1/eps²)
- `beta = 10.0` - Wilson action coupling
- `tau = 0.8-1.0` - MD trajectory length
- `n_md_steps = 12` - Leapfrog steps per trajectory
- `n_gf_steps = 4` - Gauge fixing iterations per direction

## Known Issues

- Local GPT build on this Mac has a bug in `g.matrix.exp()` for certain inputs ("only 0-dimensional arrays can be converted to Python scalars"). Tests run on remote machine.
