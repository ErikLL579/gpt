# GPT - Grid Python Toolkit

GPT is a Python library for lattice QCD built on top of Grid. This is a local development/research copy.

## Repository Structure

- `lib/gpt/` - Core GPT Python library
- `lib/cgpt/` - C++ backend (Grid interface)
- `applications/` - Example applications (HMC, etc.)
- `tests/` - Test suite
- `scripts/bootstrap/` - Build scripts for different platforms
- `erik_tests/` - Erik's research code and experiments
- `Nambu-mech/` - Nambu HMC implementation (Erik's paper arXiv:2409.18958)

## Erik's Research Projects

### RWGFFA (Random Walk Gauge Fixing with Fourier Acceleration)
Location: `erik_tests/RWGFFA/`

Implements Fourier-accelerated HMC with Landau gauge fixing based on Zhao (2018). See `erik_tests/RWGFFA/CLAUDE.md` for detailed documentation including:
- Fourier kernel implementation and the critical projector normalization fix
- Physics background and formulas
- GPT library usage patterns
- File descriptions and test commands

### Gauge-Fixing Flow (AD-based HMC with Differentiable Field Transformations)
Location: `erik_tests/gauge-fixing-flow/`

HMC in gauge-fixed variables using GPT's automatic differentiation and `differentiable_field_transformation` infrastructure. The gauge-fixing map and its Jacobian are handled automatically via AD.

**Python scripts:**
- `Trying_AD.py` — 2D (4x4) gauge-fixed HMC, single transformation layer, no checkerboard masking (commented out). 1000 trajectories.
- `multi_step_AD.py` — 2D (4x4) gauge-fixed HMC, masking commented out. 1000 trajectories.
- `Gauge-fixing-FT-code-autodiff.py` — 4D (4^4) version with checkerboard masking code (has issues, see below).
- `claude_multi_step_AD.py` — 2D (8x8) with working checkerboard masking via group-typed mask trick (see below). 5 trajectories.
- `AD_multi_step.py` — **Multi-step** 4D (4^4) gauge-fixing HMC with N sequential transformations, Luscher backward recursion for force propagation, configurable checkerboard masking. See below for details.

**Notebooks (progression of experiments):**
- `AD_gfft_HMC_normal.ipynb` — Baseline: standard 2D HMC without gauge transformation (control)
- `AD_gfft_develop.ipynb` — Exploratory: tests transformation invertibility, gradient verification, plaquette preservation (~1e-16 deviation)
- `AD_gfft_HMC_attempt_realplssave.ipynb` — First attempt at 4D gauge-fixed HMC with log-det Jacobian
- `AD_gfft_HMC_working_2x2_laat.ipynb` — 2D HMC with two sequential gauge transformations (Vgf → Vgf2), two separate log-det actions
- `AD_gfft_HMC_workinggg.ipynb` — Clean working 4D implementation, β=10.0, 13 MD steps, stable plaquette ~0.779

**Common pattern across all scripts:**
```python
from gpt.ad import reverse as rad
dft = g.qcd.gauge.smear.differentiable_field_transformation(U, ft0, inverter_force, inverter_action, optimizer)
Vgf = dft.inverse(U)              # Solve for gauge-fixed config
dfm = dft.diffeomorphism()         # Get diffeomorphism object
a_log_det = dft.action_log_det_jacobian()  # Log-det Jacobian as action
# HMC Hamiltonian: H = S_gauge(Vgf) + S_mom(P) + S_logdet(Vgf, mom)
```

### Multi-Step Gauge-Fixing (`AD_multi_step.py`)

Generalizes single-step to N sequential gauge-fixing transformations based on Luscher's trivializing maps (arXiv:0907.5491, eq. 6.5). Reference PDF: `luscher-TM.pdf`.

**Transformation chain:** W → V_1 = K_0(W) → V_2 = K_1(V_1) → ... → U = K_{N-1}(V_{N-1})
- W is the dynamical variable (updated by symplectic integrator)
- Gauge action S_gauge evaluated directly on W (gauge-invariant, no chain rule needed)
- Each step k contributes S_logdet_k with its own independent stochastic momentum

**Transformation factory pattern** (replaces global `num_steps` side effect):
```python
def make_ft(grid, nd, eps, checkerboard=None):
    # Precomputes group-typed mask in closure
    # Returns ft(U) that works on both plain fields and AD nodes
    # checkerboard=None disables masking; g.even/g.odd enables it
```

**Force propagation (Luscher backward recursion, O(N)):**
```python
F = a_log_det_list[-1].gradient(V[-2] + mom2[-1], V[-2])  # outermost
for k in range(n_gf_steps - 2, -1, -1):
    F = dfm_list[k].jacobian(V[k], V[k+1], F)   # pullback through step k
    F_local = a_log_det_list[k].gradient(V[k] + mom2[k], V[k])
    F = [g(F[mu] + F_local[mu]) for mu in range(nd)]
```

**Key insight:** `dfm.jacobian(V_k, V_{k+1}, F)` computes J_k^T · F (pullback via reverse-mode AD), which is exactly the force propagation in Luscher eq. 6.5.

**Parameters** (configurable at top of file): `n_gf_steps`, `use_masking`, `eps_gf`, `beta`, `L`, `nd`, `tau`, `n_md_steps`, `n_trajs`.

### Nambu HMC (Generalized HMC using Nambu Mechanics)
Location: `Nambu-mech/`

Implementation of Erik's paper "Generalized HMC using Nambu mechanics for lattice QCD" (arXiv:2409.18958, PDF in the directory). Each link carries a triplet {U, p, r}; Metropolis accepts on H = Σp²/2 + Σr²/2 + S_β(U) while a second Hamiltonian G drives the dynamics but never enters the measure. G must be even in p (reversibility: flip p only) and differentiable, but need NOT be gauge invariant, local, or physical.

**Equations of motion** for G = c_p Σp²/2 + c_r Σr²/2 + g₃(U), with F_H = ∂S_β, F_G = ∂g₃ (both algebra-valued forces) and a∘b the componentwise product in the adjoint index:
```
U̇ = (c_r − c_p) p∘r
frc_P = (c_r·F_H − F_G)∘R      # update_p convention: dst ← dst − ε·frc
frc_R = (F_G − c_p·F_H)∘P
```
Constraints: c_p ≠ c_r, and neither may equal the coefficient of S_β inside G. Production choice: c_p = 0.75, c_r = 1.5.

**Integrator (PRURP, paper eq. 19)** — nested GPT leapfrogs, no new integrator code:
```python
mdint = sympl.leap_frog(N, ip_P, sympl.leap_frog(1, ip_R, iq_U))
```

**Componentwise adjoint product** — no explicit generator matrices; decompose via `otype.coordinates()` (su_n.py), multiply the 8 coefficient fields, reassemble (`cw_list` in the markov scripts).

**Forces for arbitrary G via reverse-mode AD** — wrap U in AD nodes, build a scalar graph, get a `differentiable_functional` with the same interface as `action.gradient(U, U)`:
```python
from gpt.ad import reverse as rad
aU = [rad.node(g.copy(u)) for u in U]
q_func = g.qcd.gauge.differentiable_topology(aU).functional(*aU)  # G = κ·Q
F_G = q_func.gradient(U, U)   # algebra-typed, auto-projected
```

**5D OBC scaffold** (`nambu_obc5d_*.py`): each 4D link U_μ(x) is coupled through G to a μ5-plaquette rung S5 = Σ Re Tr[U_μ(x) V₅(x+μ) W_μ†(x) V₅†(x)] with vertical links V₅ and an edge layer W — no wrap-around term, so genuinely open in the fictitious 5th direction. V, W carry NO weight in H, so the 4D marginal is exactly e^(−S_β) for any κ₅ (Haar integration factorizes); they are refreshed each trajectory by an exact Haar draw (Gibbs step). All 9 link species (4 U + V + 4 W) run as one big list through the unmodified GPT machinery (`update_p`/`update_q`/`metropolis` all accept field lists). Note: `rng.element` is NOT Haar — use QR of a complex Gaussian with column-phase fix and det^(1/3) projection (`haar_su3` in the scripts). Frozen scaffold would be a Dirichlet wall, and W≡1 collapses to a Landau-gauge functional — W must be dynamical for real OBC.

**Files:**
- `nambu_hmc.py` — first implementation + frozen-r limit check (reproduces standard HMC to 2.5e-16)
- `nambu_markov.py` / `standard_hmc.py` — λS-in-G Markov chain and HMC baseline (4⁴, β=1, τ=2)
- `nambu_topo_scaling.py` / `nambu_topo_markov.py` — G = κ·Q (topological charge, κ=10) scaling + chain
- `nambu_obc5d_scaling.py` / `nambu_obc5d_markov.py` — 5D OBC scaffold scaling + chain (κ₅=1, N=10)
- `compare_plaquette.py` — binned drop-one-bin jackknife comparison with bin-size sweep (`python3 compare_plaquette.py log1 log2 16`)
- `hmc_topo_run.py` / `nambu_obc5d_topo_run.py` — production runs for the topological-tunneling comparison (β=6, 8⁴ defaults; Wilson-flowed Q5LI/Qclover/E each trajectory; NERSC checkpoints + auto-resume; tune `--nsteps` on the target machine to 65–85% acceptance)
- `topo_measure.py` — shared flow+measure+checkpoint helpers
- `tau_int.py` — Madras–Sokal τ_int with automatic windowing, tunneling rate, Q-integerness (`python3 tau_int.py hmc.log nhmc.log`)
- Logs: `hmc_1k.log`, `nambu_1k.log`, `nambu_topo_1k.log`, `nambu_obc5d_1k.log`

**Validation results** (4⁴, β=1.0, τ=2, 1000 trajectories each, bin-16 jackknife):
- Reversibility ~1e-15 and ΔH, ΔG ~ ε² scaling for all three G choices
- ⟨plaq⟩: HMC 0.059907(350); Nambu λS-G 0.060243(318) (0.71σ); Nambu κQ-G 0.059484(319) (0.90σ); Nambu 5D-OBC 0.059912(328) (0.01σ)
- Acceptance tuned to 65–85% window in all chains to expose detailed-balance violations

**Gotchas:**
- Cold-start Metropolis locks up (dH ~ +27 from unit config, rejects forever) — thermalize with `accept_reject=False`
- `g.qcd.gauge.smear.wilson_flow(U, eps)` is ONE RK4 step; loop t/eps times for flow time t; keep flow radius √(8t) ≲ L/4
- `g.qcd.gauge.energy_density(U)` returns complex — take `.real`
- β=1 on 4⁴ shows no topological freezing (τ_int ≈ 1.4); the tunneling comparison needs β≈6

## GPT Quick Reference

### Environment Setup
```bash
source gpt.env  # Set up environment before running
```

### Grid and Fields
```python
import gpt as g
import numpy as np

grid = g.grid([L, L, L, L], g.double)  # 4D lattice
U = g.qcd.gauge.unit(grid)             # Unit gauge config
P = g.group.cartesian(U)               # Lie algebra momenta
rng = g.random("seed")
rng.normal_element(P)                  # Gaussian momenta
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

### Gauge Physics
```python
g.qcd.gauge.plaquette(U)                          # Plaquette
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

### Field Indexing
```python
coords = g.coordinates(field)    # (n_sites, nd) array of coordinates
field[coords] = numpy_array      # Bulk assignment
field[x, y, z, t]                # Single site access
field[x, y, z, t].array          # Get numpy array from site
```

## Automatic Differentiation (AD) System

GPT has a full AD system in `lib/gpt/ad/` with both reverse-mode and forward-mode implementations.

### Reverse-Mode AD (`gpt.ad.reverse`)

Location: `lib/gpt/ad/reverse/`

Core class: `node_base` (aliased as `node`) in `lib/gpt/ad/reverse/node.py`.

**Node construction:**
```python
from gpt.ad import reverse as rad

# Wrap a lattice field as an AD node
x = rad.node(field_value, with_gradient=True)

# Operations on nodes build a computational graph lazily
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
# x.gradient, y.gradient hold ∂z/∂x, ∂z/∂y

# Or create a differentiable functional for use with optimizers/HMC:
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
Tracks types through the graph — `container(complex)`, `container(g.lattice, grid, otype)`, `container(g.tensor, otype)`. Handles automatic trace insertion and sum reduction when types change (e.g., field × field → scalar requires trace + sum).

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

### Differentiable Field Transformation

Location: `lib/gpt/qcd/gauge/smear/differentiable.py`

The main user-facing class is `differentiable_field_transformation` (exported from `g.qcd.gauge.smear`).

```python
dft = g.qcd.gauge.smear.differentiable_field_transformation(
    U,                  # Reference gauge field (sets up AD graph)
    ft,                 # Transformation function: U → U' (must work on AD nodes)
    inverter_force,     # Linear solver for force computation (e.g., FGCR)
    inverter_action,    # Linear solver for action evaluation
    optimizer,          # Non-linear optimizer for inverse (e.g., non_linear_cg)
)
```

**Methods:**
- `dft.inverse(U)` — Solve ft(V) = U for V using non-linear CG (minimizes ||ft(V) - U||²)
- `dft.diffeomorphism()` — Returns `dft_diffeomorphism` object with:
  - `dfm(fields)` — apply transformation
  - `dfm.jacobian(fields, fields_prime, dfields)` — Jacobian-vector product J·v via reverse-mode AD
  - `dfm.adjoint_jacobian(fields, dfields)` — J†·v via forward-mode AD
- `dft.action_log_det_jacobian()` — Returns `dft_action_log_det_jacobian`, a `differentiable_functional` that computes S = -log|det J| stochastically

**How log-det Jacobian works (`dft_action_log_det_jacobian`):**
Constructs a bilinear form ⟨left | J† | right⟩ as an AD graph, then:
1. Evaluation: solves J†J·x = mom for x using iterative solver, returns ⟨mom|x⟩
2. Gradient: differentiates the bilinear form w.r.t. gauge fields using reverse-mode AD, returns anti-Hermitian projection

**Base classes** (in `lib/gpt/core/group/`):
- `diffeomorphism` — abstract: `__call__(fields)`, `jacobian(fields, fields_prime, dfields)`
- `local_diffeomorphism` — adds `log_det_jacobian(fields)` (site-local determinant)
- `differentiable_functional` — abstract: `__call__(fields)`, `gradient(fields, dfields)`, plus `approximate_gradient()` for numerical verification

**Reference test** (`tests/qcd/gauge.py`, lines ~328-390): Canonical example using `differentiable_stout(rho=0.01)` as the transformation. Validates:
- Forward map matches reference stout smearing (~1e-25)
- `dfm.jacobian()` matches reference Jacobian (~1e-25)
- `ald(U + mom2)` evaluates the log-det action (argument = gauge fields + momentum fields concatenated)
- `ald.assert_gradient_error(rng, U + mom2, U, 1e-3, 1e-7)` validates gradients
- `ald.draw(U + mom2, rng)` samples from the action, agrees with direct evaluation to ~1e-8
- `dft.inverse(Uft)` recovers original config to ~1e-25

## Checkerboard Masking with AD

Applying a scalar checkerboard mask `fm` (a `g.complex` field) to a matrix field `B` via `B *= fm` changes the otype from `ot_matrix_su_n_fundamental_group(3)` to `ot_matrix_color(3)`. This causes a container mismatch in the AD framework's `__sub__` (which uses strict `==` instead of `accumulate_compatible`).

**Solution — group-typed mask:** Create the mask as a `su_n_fundamental_group(3)` matrix (diagonal: `fm * Identity`), so `group * group = group` preserves the otype:
```python
fm_group = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
g.identity(fm_group)
fm_group @= fm * fm_group   # @= copies data, preserving fm_group's otype
B *= fm_group                # group * group = group, otype preserved
```

**Required framework fix** (`lib/gpt/ad/reverse/node.py`, `__mul__` backward, ~line 178): `g.adj(y.value)` returns a lazy `g.expr`, which fails when `z.gradient` is a node (as happens in nested forward/reverse mode used by `action_log_det_jacobian`). Fix by evaluating the adjoint:
```python
# Original:
# x.gradient += z.gradient * g.adj(y.value)
# y.gradient += g.adj(x.value) * z.gradient
# Fixed:
x.gradient += z.gradient * g.eval(g.adj(y.value))
y.gradient += g.eval(g.adj(x.value)) * z.gradient
```

See `erik_tests/gauge-fixing-flow/CLAUDE.md` for detailed notes on what was tried.

## Build Notes

Local build scripts in `scripts/bootstrap/`:
- `erikmac` - Erik's Mac build
- `macos-sequoia.clang.no-mpi` - macOS Sequoia without MPI

NumPy 2.x compatibility fix (resolved): NumPy 2.0 deprecated implicit conversion of single-element arrays to scalars via `complex()`/`int()`, raising `TypeError`. Fixed by using `.item()` to extract the scalar first. Affected files:
- `lib/gpt/core/tensor.py` — `reduced()` and `trace()`: `complex(self.array)` → `complex(self.array.item())`
- `lib/gpt/core/expr.py` — singlet multiplication: `complex(res.array)` → `complex(res.array.item())`
- `lib/gpt/core/util.py` — `value_to_tensor()`: added `isinstance(val, np.ndarray)` guard before calling `.item()`
- `lib/gpt/core/foundation/lattice/matrix/exp.py` — norm computation: wrapped result of `g.object_rank_norm2()` with `.item()` so `int(np.log2(...))` works
- `lib/gpt/core/basis.py` — `orthogonalize()` inner product list: `complex(x)` → `complex(x.item()) if hasattr(x, 'item') else complex(x)`

## Running Tests

```bash
source gpt.env
cd tests
python3 <test_file>.py
```
