# Gauge-Fixing Flow with AD

## Goal
Implement checkerboard-masked gauge-fixing field transformation using GPT's automatic differentiation (AD) framework, for use in Fourier-accelerated HMC.

## Working Code
- `claude_multi_step_AD.py` — 2D (8x8) with checkerboard masking working. Runs HMC trajectories (overflow on traj 2, likely numerical tuning needed).
- `Trying_AD.py` / `multi_step_AD.py` — 2D versions with masking commented out, run fine
- `Gauge-fixing-FT-code-autodiff.py` — 4D version with masking, has the same otype issues (fixable with the approach below)
- `AD_multi_step.py` — **Multi-step** 4D (4^4) with N sequential gauge-fixing steps. Uses Luscher backward recursion (eq. 6.5 of arXiv:0907.5491) for O(N) force propagation. Configurable: `n_gf_steps`, `use_masking`, `eps_gf`, etc. Status: untested as of Feb 2025, deployed to remote server for first run.

## The Masking Solution

### Problem
Multiplying `B *= fm` where `fm` is a `g.complex` scalar mask changes the otype from `ot_matrix_su_n_fundamental_group(3)` to `ot_matrix_color(3)`. The AD framework's `__sub__` uses strict `==` on containers, causing a mismatch in `inverse()`.

### Fix 1 — Group-typed mask (user code)
Create the mask as a `su_n_fundamental_group(3)` field (diagonal: `fm * Identity`):
```python
fm_group = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
g.identity(fm_group)
fm_group @= fm * fm_group   # @= copies data, preserving fm_group's otype
B *= fm_group                # group * group = group, otype preserved
```

### Fix 2 — `node.py` backward pass (framework fix)
In `lib/gpt/ad/reverse/node.py`, `__mul__` backward (~line 178): `g.adj(y.value)` returns a lazy `g.expr` which fails when `z.gradient` is a node (nested forward/reverse mode in `action_log_det_jacobian`). Fixed with `g.eval()`:
```python
# x.gradient += z.gradient * g.adj(y.value)
x.gradient += z.gradient * g.eval(g.adj(y.value))
# y.gradient += g.adj(x.value) * z.gradient
y.gradient += g.eval(g.adj(x.value)) * z.gradient
```

### What was tried and didn't work
1. `B *= fm` (plain scalar mask) — container mismatch (`matrix_color` vs `su_n_fundamental_group`)
2. `fm = rad.node(fm); B *= fm` — same container mismatch
3. `fm = rad.node(fm, with_gradient=False); B *= fm` — same container mismatch
4. Group-typed mask WITHOUT `node.py` fix — `g.adj(y.value)` returns lazy expr, crashes in nested node backward pass
5. Wrapping `fm_group = rad.node(fm_group, with_gradient=False)` — same backward pass crash, plus breaks when `ftg` is called with plain lattices (not AD nodes)
6. Various `g.eval()` wrappings on user side — doesn't help because the crash is inside framework backward code

### Remaining numerical issues
- FGCR solver: true residual often much larger than computed residual (e.g., true: 4e-4 vs computed: 4e-25), suggesting ill-conditioning
- Overflow in `matrix.exp` on trajectory 2 during force computation — likely needs tuning (smaller step size, more MD steps, etc.)
- `non_linear_cg` converges in 1 iteration with f(x)=0 — transformation is very weak with mask (eps=1e-2 on active sites, eps=1e-17 on inactive)

## Files
- `fourier_kernel.py`, `fourier_hmc.py` — Fourier acceleration kernel (copied from RWGFFA)
- `Gauge_fixing_for_GFFA.pdf` — Main algorithm document
- `Gauge_fixing_for_GFFA_all_left_invariant_basis.pdf` — Companion doc (Jacobian in left-invariant basis)
- `luscher-TM.pdf` — Luscher "Trivializing maps" (arXiv:0907.5491), basis for multi-step force recursion

## Multi-Step Architecture (`AD_multi_step.py`)

**Chain:** W → V_1 = K_0(W) → V_2 = K_1(V_1) → ... → U = K_{N-1}(V_{N-1})

**Key design choices:**
- `make_ft(grid, nd, eps, checkerboard)` factory replaces global `num_steps` side effect. Precomputes group-typed mask in closure.
- N independent `differentiable_field_transformation` objects, one per step, with alternating even/odd checkerboard masks
- Gauge action evaluated on W directly (gauge-invariant → no chain rule needed)
- Each step has independent stochastic momentum for log-det estimator
- Force propagation via Luscher backward recursion: `F = J_k^T · F + dS_logdet_k/dV_k`, using `dfm.jacobian()` for pullback (reverse-mode AD computes J^T · v)
- Inverse chain: `dft_list[k].inverse()` applied in reverse order to recover W from U

## Bug fixes applied
- `lib/gpt/core/basis.py` line 46: NumPy 2.x fix for `complex(x)` → `complex(x.item()) if hasattr(x, 'item') else complex(x)`
- `lib/gpt/ad/reverse/node.py` ~line 178: `g.adj(y.value)` → `g.eval(g.adj(y.value))` in `__mul__` backward (both x and y branches)
