# GPT Knowledge Compilation

Compiled from all CLAUDE.md files and auto-memory across the GPT project.

---

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [Environment & Build](#environment--build)
3. [GPT Library Reference](#gpt-library-reference)
4. [Automatic Differentiation (AD) System](#automatic-differentiation-ad-system)
5. [Differentiable Field Transformation](#differentiable-field-transformation)
6. [Checkerboard Masking with AD](#checkerboard-masking-with-ad)
7. [Research: RWGFFA](#research-rwgffa)
8. [Research: Gauge-Fixing Flow (AD-based HMC)](#research-gauge-fixing-flow-ad-based-hmc)
9. [Multi-Step Gauge-Fixing HMC](#multi-step-gauge-fixing-hmc)
10. [Bug Fixes Applied](#bug-fixes-applied)
11. [Git & Deployment Setup](#git--deployment-setup)

---

## Repository Overview

GPT is a Python library for lattice QCD built on top of Grid. This is a local development/research copy.

- `lib/gpt/` — Core GPT Python library
- `lib/cgpt/` — C++ backend (Grid interface)
- `applications/` — Example applications (HMC, etc.)
- `tests/` — Test suite
- `scripts/bootstrap/` — Build scripts for different platforms
- `erik_tests/` — Erik's research code and experiments
  - `erik_tests/RWGFFA/` — Fourier-accelerated HMC with Landau gauge fixing
  - `erik_tests/gauge-fixing-flow/` — AD-based gauge-fixing flow HMC

---

## Environment & Build

### Setup
```bash
source gpt.env  # Required before running anything
```

### Build Scripts
- `scripts/bootstrap/erikmac` — Erik's Mac build (OpenMP via Homebrew LLVM)
- `scripts/bootstrap/macos-sequoia.clang.no-mpi` — macOS Sequoia without MPI

### Build Notes
- Local Mac: built with OpenMP via Homebrew LLVM
- Use `export OMP_NUM_THREADS=N` to set thread count
- Remote server: `mrvectorspaces@qcdn0:~/QCD/gpt`

### Running Tests
```bash
source gpt.env
cd tests
python3 <test_file>.py
```

---

## GPT Library Reference

### Grid and Fields
```python
import gpt as g
import numpy as np

grid = g.grid([L, L, L, L], g.double)  # 4D lattice
V = grid.gsites      # Total number of sites
nd = grid.nd         # Number of dimensions (4)
L = grid.fdimensions # [Lx, Ly, Lz, Lt]

U = g.qcd.gauge.unit(grid)             # Unit gauge config (list of 4 SU(3) matrices)
P = g.group.cartesian(U)               # Lie algebra momenta
rng = g.random("seed")
rng.normal_element(P)                  # Gaussian momenta

# Scalar fields
phi = g.complex(grid)
phi = g.lattice(grid, g.ot_complex_additive_group())  # Equivalent

# Matrix fields
D = g.mcomplex(grid, nd)        # nd×nd complex matrix at each site
```

### Common Operations
```python
g.eval(expr)                 # Evaluate lazy expression
g.copy(field)                # Deep copy
g.adj(field)                 # Hermitian conjugate
g.trace(M)                   # Matrix trace
g.sum(field)                 # Sum over lattice
g.cshift(field, mu, +/-1)    # Shift in direction mu

fft = g.fft()
field_k = g.eval(fft * field)         # Forward FFT
field_x = g.eval(g.adj(fft) * field_k) # Inverse FFT
```

### Coordinates and Field Assignment
```python
coord_array = g.coordinates(phi)  # Shape (n_sites, nd), gives [x,y,z,t] for each site
phi[coord_array] = values         # Assign numpy array to lattice
phi[x, y, z, t]                   # Access single site (returns gpt tensor)
phi[x, y, z, t].array             # Get numpy array from tensor
```

### Gauge Physics
```python
g.qcd.gauge.plaquette(U)                          # Average plaquette
g.qcd.gauge.action.wilson(beta)                   # Wilson action
g.qcd.gauge.smear.wilson_flow(U, epsilon=0.01)    # Wilson flow step
g.qcd.gauge.energy_density(U)                     # Energy density
g.matrix.exp(A)                                   # Matrix exponential
g.qcd.gauge.project.traceless_anti_hermitian(A)   # Project to su(N)
```

### HMC
```python
action = g.qcd.gauge.action.wilson(beta)
sympl = g.algorithms.integrator.symplectic
metro = g.algorithms.markov.metropolis(rng)

ip = sympl.update_p(P, lambda: action.gradient(U, U))
iq = sympl.update_q(U, lambda: velocity(P))
mdint = sympl.leap_frog(n_steps, ip, iq)

accrej = metro(U)       # Checkpoint
mdint(tau)              # Run trajectory
accepted = accrej(H1, H0)  # Accept/reject
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

---

## Automatic Differentiation (AD) System

GPT has a full AD system in `lib/gpt/ad/` with both reverse-mode and forward-mode implementations.

### Reverse-Mode AD (`gpt.ad.reverse`)

Location: `lib/gpt/ad/reverse/`

Core class: `node_base` (aliased as `node`) in `lib/gpt/ad/reverse/node.py`.

**Node construction:**
```python
from gpt.ad import reverse as rad

x = rad.node(field_value, with_gradient=True)
y = rad.node(other_field)
z = x * y + x       # z._children includes x, y; z._forward and z._backward are set
```

**Node attributes:**
- `value` — computed forward value (lattice field, tensor, or scalar)
- `gradient` — accumulated gradient (set during backward pass)
- `_forward` — callable to compute value from children
- `_backward` — callable to propagate gradients to children
- `_children` — tuple of child nodes in the DAG
- `_container` — type descriptor (complex, tensor, or lattice+grid+otype)
- `with_gradient` — whether this node participates in gradient computation
- `_tag` — optional debug label

**Evaluation and gradient computation:**
```python
result = z(with_gradients=True)    # Forward + backward pass
# z.value now holds the result
# x.gradient, y.gradient hold dz/dx, dz/dy

func = z.functional(x, y)          # Returns a differentiable_functional
func(fields)                       # Evaluate
func.gradient(fields, dfields)     # Compute gradients
```

**Supported operations** (all with correct backward rules):
- Arithmetic: `+`, `-`, `*`, `/` (division requires scalar denominator), unary `-`
- `g.ad.reverse.foundation.cshift(x, mu, disp)` — lattice shift; backward = shift in opposite direction
- `g.ad.reverse.foundation.adj(x)` — Hermitian conjugate; backward = adj(dz)
- `g.ad.reverse.foundation.trace(x)` — matrix trace; backward = identity(x) * dz
- `g.ad.reverse.foundation.sum(x)` — lattice sum; backward = identity(x) * dz
- `g.ad.reverse.foundation.inner_product(x, y)` — adj(x)*y
- `g.ad.reverse.foundation.matrix.exp.function(x)` — matrix exponential (Taylor + scaling/squaring)
- `g.norm2(x)` — works on nodes, returns scalar node

**Container system** (`lib/gpt/ad/reverse/util.py`):
Tracks types through the graph — `container(complex)`, `container(g.lattice, grid, otype)`, `container(g.tensor, otype)`. Handles automatic trace insertion and sum reduction when types change (e.g., field * field -> scalar requires trace + sum).

**Memory management:**
Forward and backward passes track when intermediates are last used and free them. Enable verbose logging with `g.default.is_verbose("ad_memory")`.

### Forward-Mode AD (`gpt.ad.forward`)

Location: `lib/gpt/ad/forward/`

Uses formal power series in infinitesimal symbols for forward-mode differentiation and Taylor expansion.

```python
import gpt.ad.forward as fad

eps = fad.infinitesimal("eps")
On = fad.landau(eps**2)              # Truncate at O(eps^2)
s = fad.series(field, On)            # Series: field + O(eps^2)
s[eps] = g(1j * dfield * field)      # Set first-order term

result = ft(s)                       # Apply any function to the series
deriv = result[eps]                  # Extract first derivative
```

**Key classes:**
- `infinitesimal` — symbolic variable (supports products like eps_a * eps_b for higher-order)
- `landau` — truncation order specification (O(eps^n))
- `series` — power series with lattice-valued coefficients; supports `+`, `-`, `*`, `/`

**Foundation operations** (`lib/gpt/ad/forward/foundation/`):
All lattice operations (`cshift`, `trace`, `adj`, `sum`, `inner_product`) distribute over series terms automatically. Also includes `matrix.det(series)` using Taylor expansion of determinant.

---

## Differentiable Field Transformation

Location: `lib/gpt/qcd/gauge/smear/differentiable.py`

```python
dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,                  # Reference gauge field (sets up AD graph)
    ft,                 # Transformation function: U -> U' (must work on AD nodes)
    inverter_force,     # Linear solver for force computation (e.g., FGCR)
    inverter_action,    # Linear solver for action evaluation
    optimizer,          # Non-linear optimizer for inverse (e.g., non_linear_cg)
)
```

**Methods:**
- `dft.inverse(U)` — Solve ft(V) = U for V using non-linear CG (minimizes ||ft(V) - U||^2)
- `dft.diffeomorphism()` — Returns `dft_diffeomorphism` object with:
  - `dfm(fields)` — apply transformation
  - `dfm.jacobian(fields, fields_prime, dfields)` — Jacobian-vector product J*v via reverse-mode AD
  - `dfm.adjoint_jacobian(fields, dfields)` — J^dag * v via forward-mode AD
- `dft.action_log_det_jacobian()` — Returns `dft_action_log_det_jacobian`, a `differentiable_functional` that computes S = -log|det J| stochastically

**How log-det Jacobian works (`dft_action_log_det_jacobian`):**
Constructs a bilinear form <left | J^dag | right> as an AD graph, then:
1. Evaluation: solves J^dag J * x = mom for x using iterative solver, returns <mom|x>
2. Gradient: differentiates the bilinear form w.r.t. gauge fields using reverse-mode AD, returns anti-Hermitian projection

**Base classes** (in `lib/gpt/core/group/`):
- `diffeomorphism` — abstract: `__call__(fields)`, `jacobian(fields, fields_prime, dfields)`
- `local_diffeomorphism` — adds `log_det_jacobian(fields)` (site-local determinant)
- `differentiable_functional` — abstract: `__call__(fields)`, `gradient(fields, dfields)`, plus `approximate_gradient()` for numerical verification

**Reference test** (`tests/qcd/gauge.py`, lines ~328-390): Canonical example using `differentiable_stout(rho=0.01)` as the transformation. Validates:
- Forward map matches reference stout smearing (~1e-25)
- `dfm.jacobian()` matches reference Jacobian (~1e-25)
- `ald(U + mom2)` evaluates the log-det action
- `ald.assert_gradient_error(rng, U + mom2, U, 1e-3, 1e-7)` validates gradients
- `ald.draw(U + mom2, rng)` samples from the action, agrees with direct evaluation to ~1e-8
- `dft.inverse(Uft)` recovers original config to ~1e-25

**Common usage pattern across gauge-fixing scripts:**
```python
from gpt.ad import reverse as rad
dft = g.qcd.gauge.smear.differentiable_field_transformation(U, ft0, inverter_force, inverter_action, optimizer)
Vgf = dft.inverse(U)              # Solve for gauge-fixed config
dfm = dft.diffeomorphism()         # Get diffeomorphism object
a_log_det = dft.action_log_det_jacobian()  # Log-det Jacobian as action
# HMC Hamiltonian: H = S_gauge(Vgf) + S_mom(P) + S_logdet(Vgf, mom)
```

---

## Checkerboard Masking with AD

### Problem
Applying a scalar checkerboard mask `fm` (a `g.complex` field) to a matrix field `B` via `B *= fm` changes the otype from `ot_matrix_su_n_fundamental_group(3)` to `ot_matrix_color(3)`. This causes a container mismatch in the AD framework's `__sub__` (which uses strict `==` instead of `accumulate_compatible`).

### Solution — Group-typed mask
Create the mask as a `su_n_fundamental_group(3)` matrix (diagonal: `fm * Identity`), so `group * group = group` preserves the otype:
```python
fm_group = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
g.identity(fm_group)
fm_group @= fm * fm_group   # @= copies data, preserving fm_group's otype
B *= fm_group                # group * group = group, otype preserved
```

### Required framework fix
In `lib/gpt/ad/reverse/node.py`, `__mul__` backward (~line 178): `g.adj(y.value)` returns a lazy `g.expr`, which fails when `z.gradient` is a node (nested forward/reverse mode in `action_log_det_jacobian`). Fix:
```python
# Original:
# x.gradient += z.gradient * g.adj(y.value)
# y.gradient += g.adj(x.value) * z.gradient
# Fixed:
x.gradient += z.gradient * g.eval(g.adj(y.value))
y.gradient += g.eval(g.adj(x.value)) * z.gradient
```

### What was tried and didn't work
1. `B *= fm` (plain scalar mask) — container mismatch
2. `fm = rad.node(fm); B *= fm` — same container mismatch
3. `fm = rad.node(fm, with_gradient=False); B *= fm` — same
4. Group-typed mask WITHOUT `node.py` fix — `g.adj(y.value)` returns lazy expr, crashes in nested node backward pass
5. Wrapping `fm_group = rad.node(fm_group, with_gradient=False)` — same backward pass crash
6. Various `g.eval()` wrappings on user side — doesn't help because crash is inside framework

### Remaining numerical issues
- FGCR solver: true residual often much larger than computed residual (e.g., true: 4e-4 vs computed: 4e-25), suggesting ill-conditioning
- Overflow in `matrix.exp` on trajectory 2 during force computation — likely needs tuning (smaller step size, more MD steps)
- `non_linear_cg` converges in 1 iteration with f(x)=0 — transformation is very weak with mask

---

## Research: RWGFFA

**Location:** `erik_tests/RWGFFA/`

Implements Fourier-accelerated HMC for lattice QCD with optional Landau gauge fixing, based on Zhao (2018).

### Project Status
The Fourier kernel implementation has been debugged and verified. Key fix (January 2025): the longitudinal projector P^L must be normalized by |k_hat|^2, NOT by s^2 = |k_hat|^2 + epsilon^2. The epsilon regulator only appears in eigenvalues, not in projector normalization.

**Verified (February 2025):**
- Kernel identity: D * D^{-1/2} * D^{-1/2} = I verified at ALL 256 lattice sites on 4^4 grid to machine precision (max deviation ~7.8e-16)
- Velocity in Lie algebra: v = IFFT[D*FFT[P]] is Hermitian and traceless to machine precision (~3e-16)

**Open Issues:**
- Gauge fixing reversibility needs theoretical investigation

### Files
- `fourier_kernel.py` — Builds D_{mu,nu}(k), sqrt(D), and D^{-1/2} as nd x nd matrix lattices
- `fourier_hmc.py` — `compute_D_components()` and `compute_sqrt_Dinv_components()` as scalar field dictionaries, plus momentum generation and kinetic energy functions
- `fourier_hmc_test_nogf.py` — Fourier HMC without gauge fixing (kernel validation)
- `fourier_hmc_gf_test.py` — Fourier HMC with 50/50 forward/reverse gauge fixing
- `gf_test.py` — Tests gauge fixing reversibility
- `claude-debug/` — Various debugging scripts
- `claude-debug-mom-draw/` — Kernel verification scripts

### Physics: Fourier Acceleration (Zhao 2018)
The Fourier kernel D_{mu,nu}(k) accelerates HMC by preconditioning the kinetic term:
- Standard HMC: H_p = tr(P^dag P)
- Fourier HMC: H_p = sum_k tr(P^dag(k) D(k) P(k))

D has eigenvalue decomposition:
- D = (1/s^2) P^T + (1/M^2) P^L
- s^2 = |k_hat|^2 + epsilon^2 where |k_hat|^2 = sum_mu sin^2(k_mu/2)
- P^T = transverse projector (3 modes), P^L = longitudinal projector (1 mode)
- M = gauge fixing mass parameter, epsilon = IR regulator

Critical formula for projectors:
```
k_hat_mu = e^{-i k_mu/2} sin(k_mu/2)
P^L_{mu,nu} = k_hat_mu * k_hat*_nu / |k_hat|^2    (NOT / s^2!)
P^T_{mu,nu} = delta_{mu,nu} - P^L_{mu,nu}
```

P^L must satisfy P^L * P^L = P^L. This only works if normalized by |k_hat|^2, not s^2.

### Momentum Generation
To sample P from exp(-P^dag D P):
1. Generate eta ~ N(0,1) in position space (Hermitian, traceless)
2. FFT to momentum space: eta(k)
3. Apply D^{-1/2}: P(k) = D^{-1/2}(k) * eta(k)
4. IFFT back: P(x)

D^{-1/2} = s * P^T + M * P^L = s * I + (M - s) * P^L

### Common Parameters
- `M = 1.0` — Gauge fixing mass
- `eps_fourier = 0.5` — IR regulator
- `beta = 10.0` — Wilson action coupling
- `tau = 0.8-1.0` — MD trajectory length
- `n_md_steps = 12` — Leapfrog steps per trajectory
- `n_gf_steps = 4` — Gauge fixing iterations per direction

---

## Research: Gauge-Fixing Flow (AD-based HMC)

**Location:** `erik_tests/gauge-fixing-flow/`

HMC in gauge-fixed variables using GPT's automatic differentiation and `differentiable_field_transformation` infrastructure. The gauge-fixing map and its Jacobian are handled automatically via AD.

### Python Scripts
- `Trying_AD.py` — 2D (4x4) gauge-fixed HMC, single transformation layer, no masking. 1000 trajectories.
- `multi_step_AD.py` — 2D (4x4) gauge-fixed HMC, masking commented out. 1000 trajectories.
- `Gauge-fixing-FT-code-autodiff.py` — 4D (4^4) version with checkerboard masking code (has issues).
- `claude_multi_step_AD.py` — 2D (8x8) with working checkerboard masking. 5 trajectories.
- `AD_multi_step.py` — **Multi-step** 4D (4^4) with N sequential gauge-fixing steps, Luscher backward recursion, configurable masking.

### Notebooks (progression of experiments)
- `AD_gfft_HMC_normal.ipynb` — Baseline: standard 2D HMC without gauge transformation (control)
- `AD_gfft_develop.ipynb` — Exploratory: tests transformation invertibility, gradient verification, plaquette preservation (~1e-16 deviation)
- `AD_gfft_HMC_attempt_realplssave.ipynb` — First attempt at 4D gauge-fixed HMC with log-det Jacobian
- `AD_gfft_HMC_working_2x2_laat.ipynb` — 2D HMC with two sequential gauge transformations
- `AD_gfft_HMC_workinggg.ipynb` — Clean working 4D implementation, beta=10.0, 13 MD steps, stable plaquette ~0.779

### Reference PDFs
- `Gauge_fixing_for_GFFA.pdf` — Main algorithm document
- `Gauge_fixing_for_GFFA_all_left_invariant_basis.pdf` — Companion doc (Jacobian in left-invariant basis)
- `luscher-TM.pdf` — Luscher "Trivializing maps" (arXiv:0907.5491)

---

## Multi-Step Gauge-Fixing HMC

**Key file:** `erik_tests/gauge-fixing-flow/AD_multi_step.py`

Generalizes single-step to N sequential gauge-fixing transformations based on Luscher's trivializing maps (arXiv:0907.5491, eq. 6.5).

### Transformation Chain
W -> V_1 = K_0(W) -> V_2 = K_1(V_1) -> ... -> U = K_{N-1}(V_{N-1})

- W is the dynamical variable (updated by symplectic integrator)
- Gauge action S_gauge evaluated directly on W (gauge-invariant, no chain rule needed)
- Each step k contributes S_logdet_k with its own independent stochastic momentum

### Transformation Factory
```python
def make_ft(grid, nd, eps, checkerboard=None):
    # Precomputes group-typed mask in closure
    # Returns ft(U) that works on both plain fields and AD nodes
    # checkerboard=None disables masking; g.even/g.odd enables it
```

### Force Propagation (Luscher backward recursion, O(N))
```python
F = a_log_det_list[-1].gradient(V[-2] + mom2[-1], V[-2])  # outermost
for k in range(n_gf_steps - 2, -1, -1):
    F = dfm_list[k].jacobian(V[k], V[k+1], F)   # pullback through step k
    F_local = a_log_det_list[k].gradient(V[k] + mom2[k], V[k])
    F = [g(F[mu] + F_local[mu]) for mu in range(nd)]
```

**Key insight:** `dfm.jacobian(V_k, V_{k+1}, F)` computes J_k^T * F (pullback via reverse-mode AD), which is exactly the force propagation in Luscher eq. 6.5.

### Design Choices
- `make_ft(grid, nd, eps, checkerboard)` factory replaces global `num_steps` side effect
- N independent `differentiable_field_transformation` objects, one per step, with alternating even/odd checkerboard masks
- Each step has independent stochastic momentum for log-det estimator
- Inverse chain: `dft_list[k].inverse()` applied in reverse order to recover W from U

### Parameters
Configurable at top of file: `n_gf_steps`, `use_masking`, `eps_gf`, `beta`, `L`, `nd`, `tau`, `n_md_steps`, `n_trajs`.

**Status:** Untested as of Feb 2025, deployed to remote server for first run.

---

## Bug Fixes Applied

### NumPy 2.x Compatibility
NumPy 2.0 deprecated implicit conversion of single-element arrays to scalars via `complex()`/`int()`. Fixed by using `.item()`:

- `lib/gpt/core/tensor.py` — `reduced()` and `trace()`: `complex(self.array)` -> `complex(self.array.item())`
- `lib/gpt/core/expr.py` — singlet multiplication: `complex(res.array)` -> `complex(res.array.item())`
- `lib/gpt/core/util.py` — `value_to_tensor()`: added `isinstance(val, np.ndarray)` guard before `.item()`
- `lib/gpt/core/foundation/lattice/matrix/exp.py` — norm computation: wrapped `g.object_rank_norm2()` with `.item()`
- `lib/gpt/core/basis.py` — `orthogonalize()`: `complex(x)` -> `complex(x.item()) if hasattr(x, 'item') else complex(x)`

### AD Backward Pass Fix
`lib/gpt/ad/reverse/node.py` `__mul__` backward (~line 178): `g.adj(y.value)` returns a lazy `g.expr`, which fails when `z.gradient` is a node (nested forward/reverse mode in `action_log_det_jacobian`). Fixed with `g.eval()`:
```python
x.gradient += z.gradient * g.eval(g.adj(y.value))
y.gradient += g.eval(g.adj(x.value)) * z.gradient
```

---

## Git & Deployment Setup

- **Upstream:** `origin` -> `https://github.com/lehner/gpt`
- **Fork:** `myfork` -> `git@github.com:ErikLL579/gpt.git`
- **Working branch:** `erik-dev`
- **GitHub account:** `ErikLL579`
- **Remote server:** `mrvectorspaces@qcdn0:~/QCD/gpt` (has `myfork` remote configured via SSH)
- **Running long jobs:** `nohup python3 script.py > log.txt 2>&1 &`
