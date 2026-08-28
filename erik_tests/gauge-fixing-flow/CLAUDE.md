# Gauge-Fixing Flow with AD

## Goal
Implement checkerboard-masked gauge-fixing field transformation using GPT's automatic differentiation (AD) framework, for use in Fourier-accelerated HMC.

## Working Code
- `gf_local_jacobian.py` + `gf_hmc_local.py` — **local-Jacobian version** of the one-step flow (Aug 27 2026). Exact per-site block determinant instead of the J†J estimator; no `mom2`, no FGCR in the log-det path. `dH_scaling.py` runs the step-size scan. See "Local Jacobian" below.
- `gf_fourier_4d.py` — the same 4D local-Jacobian HMC with the **Zhao Fourier-accelerated kinetic term** (Aug 28 2026). sqrt(D) comes from `fourier_kernel.build_sqrt_fourier_kernel` unchanged; it is fed to GPT's built-in `g.qcd.scalar.action.fourier_mass_term`, which derives D = sqrt(D)^2 and D^-1/2 = inv(sqrt(D)) and supplies draw/energy/velocity. Flags: `--tau --nsteps --M --eps_fourier`.
- `claude_multi_step_AD.py` — 2D (8x8) with checkerboard masking working. Runs HMC trajectories (overflow on traj 2, likely numerical tuning needed).
- `Trying_AD.py` — 2D (4x4), masking commented out, 1000 trajectories, runs fine. **The reference the local-Jacobian port was made from.** `multi_step_AD.py` is byte-identical to it (pure duplicate).
- `Gauge-fixing-FT-code-autodiff.py` — 4D version with masking; still contains debug `print`s and the abandoned `rad.node(fm)` attempt.
- `claude_AD_multi_step.py` (renamed from `AD_multi_step.py`) — **Multi-step** 4D (4^4) with N sequential gauge-fixing steps. Uses Luscher backward recursion (eq. 6.5 of arXiv:0907.5491) for O(N) force propagation. Configurable: `n_gf_steps`, `use_masking`, `eps_gf`, etc. Smoke-tested July 10 2026 (log: `smoke_test.log`); ~2.5 min/trajectory at 4^4 with 2 GF steps.

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
- ~~FGCR ill-conditioning~~ **RESOLVED / stale.** The July 10 2026 smoke log shows true residual == computed residual to the last digit across all 445 solves, at 10-16 iterations. The earlier 4e-4 vs 4e-25 observation no longer reproduces.
- Overflow in `matrix.exp` on trajectory 2 during force computation — did not recur in the July 10 4^4 run.
- `non_linear_cg` converges in 1 iteration with f(x)=0 — transformation is very weak with mask (eps=1e-2 on active sites, eps=1e-17 on inactive)

## Efficiency notes for the J†J driver (`claude_AD_multi_step.py`)
- **The integrator nesting is a no-op.** `leap_frog(1, ip_log_det, leap_frog(1, ip_gauge, iq))` called N times in a Python loop prints as `P(0.5) P(0.5) Q(1.0) P(0.5) P(0.5)` — the expensive log-det force is evaluated exactly as often as the free gauge force, 2N times per trajectory. Build `leap_frog(N, ip_log_det, leap_frog(m, ip_gauge, iq))` and call `mdint(tau)` **once**; `simplify()` then merges adjacent half-steps and the log-det count drops to N+1 (14 instead of 26 at N=13, ~1.9x).
- `hamiltonian(draw=True)` pays two extra FGCR solves to evaluate `ald(V + Jη)` when `dft_action_log_det_jacobian.draw()` returns that value analytically as `Σ|η|²`.
- `forward_pass(W)` is recomputed 3-4x per MD step (in `hamiltonian` and again in `log_det_force`).
- `trace_U` divides by `4 * size**4` — that is the 4D normalisation, wrong by 32x in the 2D scripts. Use `nd * V * Nc`.

## Local Jacobian (`gf_local_jacobian.py`, Aug 27 2026)

Detail lives in the root `CLAUDE.md` ("Local Jacobian for the gauge-fixing step"). Summary:
- The masked step is a gauge transformation with V on one checkerboard; J is block diagonal over active sites, blocks of `(2*nd links) x (ng generators)` = 32x32 in 2D, 64x64 in 4D. Masking is **required** — unmasked the stars overlap and no local determinant exists.
- Two traps, both fixed: `M` is not zero on inactive sites (the gather `cshift` reaches neighbouring active stars), and `inv()` of the regularised `M` is the identity there. **Mask both `M` and `Minv`.**
- Validated: block 9.3e-12 vs finite differences, action vs numpy determinant to all printed digits, `assert_gradient_error` 2.2e-10, `dH ~ eps^2`, reversibility 1.7e-30.
- **Fourier-accelerated kinetic term works** (M=1.0, eps_fourier=0.5): tau=0.25 gives dH = +0.109 vs -0.107 for the trivial kinetic term, 614 s vs 608 s. GPT's `fourier_mass_term` is the same construction as `fourier_hmc.py`'s kinetic_energy/velocity/fill_fourier_momenta — `g.adj(fft)` and `g.inv(fft)` are the *same function* in GPT (`lib/gpt/core/coordinates.py`, `adj_mat = inv_mat = mat_backward`) and `scale_unitary**2 == V` — plus a Hermitian symmetrisation of the velocity. Feeding it Erik's sqrt(D) reproduced the closed-form gradient error and drawn KE bit-for-bit, which also confirms `P^L P^L = P^L` numerically.
- **4D runs unchanged** (Aug 28 2026, `dH_4d_single.py`): 4^4, 64x64 block, one tau=0.25 trajectory gives dH = -0.107 on H ~ 7531 (dS_gauge -47.9 vs dS_mom +47.8). ~67 s per log-det force on 12 threads. The log-det itself moves only 0.03 at eps_gf=1e-2, so raise eps_gf to actually stress it. `dH_scaling_4d.py` is the 4D eps^2 scan, written but not run (~1 h).
- **Not validated: the measure.** ⟨plaquette⟩ is provably blind to the log-det here (ft is a gauge transformation, plaq is gauge-invariant), so only a gauge-*variant* observable — the link trace — can test it. Best available check: link-trace comparison against the existing J†J implementation on the same map.

## Files
- `fourier_kernel.py`, `fourier_hmc.py` — Fourier acceleration kernel (copied from RWGFFA)
- `Gauge_fixing_for_GFFA.pdf` — Main algorithm document
- `Gauge_fixing_for_GFFA_all_left_invariant_basis.pdf` — Companion doc (Jacobian in left-invariant basis)
- `luscher-TM.pdf` — Luscher "Trivializing maps" (arXiv:0907.5491), basis for multi-step force recursion

## Multi-Step Architecture (`claude_AD_multi_step.py`)

**Chain:** W → V_1 = K_0(W) → V_2 = K_1(V_1) → ... → U = K_{N-1}(V_{N-1})

**Key design choices:**
- `make_ft(grid, nd, eps, checkerboard)` factory replaces global `num_steps` side effect. Precomputes group-typed mask in closure.
- N independent `differentiable_field_transformation` objects, one per step, with alternating even/odd checkerboard masks
- Gauge action evaluated on W directly (gauge-invariant → no chain rule needed)
- Each step has independent stochastic momentum for log-det estimator
- Force propagation via Luscher backward recursion: `F = J_k^T · F + dS_logdet_k/dV_k`, using `dfm.jacobian()` for pullback (reverse-mode AD computes J^T · v)
- Inverse chain: `dft_list[k].inverse()` applied in reverse order to recover W from U

## Bug fixes applied
- `lib/gpt/core/basis.py` line 46: NumPy 2.x fix for `complex(x)` → `complex(x.item()) if hasattr(x, 'item') else complex(x)` (obsolete once we take the full upstream merge — upstream fixed it independently)
- `lib/gpt/ad/reverse/node.py` ~line 178: `g.adj(y.value)` → `g.eval(g.adj(y.value))` in `__mul__` backward (both x and y branches). Still ours to carry; upstream has not applied it.

## Gotchas
- **Keep every `differentiable_field_transformation` alive.** A GC'd dft can corrupt subsequent ones built over the same `U` (J collapses to the identity). Order-dependent, not root-caused. `claude_AD_multi_step.py` is safe (they live in `dft_list`).
- **`assert_gradient_error` does not catch a NaN gradient** — `if eps > epsilon_assert` is `False` for NaN, so it prints `nan` and passes.
