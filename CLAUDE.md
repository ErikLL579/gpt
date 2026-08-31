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
- `gf_local_jacobian.py` / `gf_hmc_local.py` / `dH_scaling.py` — **local-Jacobian port of `Trying_AD.py`** (Aug 27 2026). Same map/beta/tau/nsteps/eps, but the log-det comes from an exact per-site block determinant instead of the global stochastic J†J estimator, so the auxiliary momentum (`mom2`) and both FGCR solves disappear from the log-det path. See "Local Jacobian for the gauge-fixing step" below.
- `Trying_AD.py` — 2D (4x4) gauge-fixed HMC, single transformation layer, no checkerboard masking (commented out). 1000 trajectories. **This is the working reference the local-Jacobian version was ported from.**
- `multi_step_AD.py` — byte-identical to `Trying_AD.py` (pure duplicate, despite the differing descriptions here).
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
- `nambu_obc5d_linear_scaling.py` / `nambu_obc5d_linear_topo_run.py` — 5D OBC variant with G **linear in r** (paper eq. 30 form): G = γ·Σr_a + κ₅·S5. EOM: U̇ = γp (plain, no p∘r), frc_P = γF_S − κ₅F_G∘r, frc_R = κ₅F_G∘p. γ=1, κ₅=0 reduces exactly to standard HMC ("minimal deformation"). Flags: --gamma replaces --c_p/--c_r. Scaling test passed July 15 2026 (reversibility ~1.7e-15, dH~ε^1.99, dG~ε^2.00)
- `compare_plaquette.py` — binned drop-one-bin jackknife comparison with bin-size sweep (`python3 compare_plaquette.py log1 log2 16`)
- `hmc_topo_run.py` / `nambu_obc5d_topo_run.py` — production runs for the topological-tunneling comparison (β=6, 8⁴ defaults; Wilson-flowed Q5LI/Qclover/E each trajectory; NERSC checkpoints + auto-resume; tune `--nsteps` on the target machine to 65–85% acceptance)
- `topo_measure.py` — shared flow+measure+checkpoint helpers
- `tau_int.py` — Madras–Sokal τ_int with automatic windowing, tunneling rate, Q-integerness (`python3 tau_int.py hmc.log nhmc.log`)
- `nambu_smq_linear_scaling.py` / `nambu_smq_topo_run.py` — G = γ·Σr_a + κ·Q_sm (stout-smeared clover Q via `differentiable_topology(...).functional(*aU).transformed(stout)` chain). Scaling passed July 16 2026 (gradient 1.5e-10, reversibility 1.3e-15, dH~ε^1.98, dG~ε^1.99)
- `nambu_ampl_scaling.py` / `nambu_ampl_topo_run.py` — **smeared-Wilson amplifier** (quadratic scheme, g₃ = λ·S_β[stout^n(U)]): per adjoint component ṗ_a=−A r_a, ṙ_a=−B p_a with A=(c_r F_H−λF_sm)_a, B=(λF_sm−c_p F_H)_a; AB>0 ("armed", iff s=F_sm/F_H ∈ (c_p/λ, c_r/λ)) → exponential (p,r) growth at rate √(AB) with sign-definite p_a r_a = directed U-velocity; AB<0 → bounded rotor. IR/smooth components armed, UV rotors — scale-selective energy concentration. Scaling passed July 16 2026 (gradient 7.2e-11, dH~ε^2.30). Production script logs `armed = X%, rate = Y` vital signs (frozen-force snapshot of the final config each trajectory). **Pure quadratic pays a transport tax** (U̇=(c_r−c_p)p∘r: prefactor, ⟨|pr|⟩/⟨|p|⟩=0.64, rotor sign-flips at 2√|AB|): measured t2E step rms per √(MD time) 0.0030 (cons) / 0.0060 (aggr) / 0.0094 (HMC) — scales linearly with c_r−c_p (ratio 1.99 vs predicted 1.81), so bulk-IR floor unreachable for pure quadratic; ceiling (topology) tested by the 20k aggressive run. **Mixed G via `--gamma`** (both scripts): G += γΣr_a → U̇ = γp + (c_r−c_p)p∘r — γ=1 restores full HMC ballistic transport for all components, armed keep the same ±√(AB) instability ("idle + supercharger"); γ=0 = pure quadratic. Scaling passed July 17 2026 (gradient 7.7e-11, dH~ε^1.94, dG~ε^1.97; test now 6 draws, fit N=20–160)
- `window_histogram.py` — per-component s = F_sm,a/F_H,a histogram on stored checkpoints + **fair-cost window scan** (fixed √(c_p·c_r) = UV-rotor stiffness across candidate windows). Candidates from wscan_b6.log: conservative c_p 0.75/c_r 1.5/λ 1.125 (7.6% armed), middle 0.685/1.643/1.369 (11.9%), aggressive 0.581/1.936/1.453 (17.4%). Live chains confirm: equilibrium `armed` 7.4–7.5% vs predicted 7.6%. s-histogram is UNIMODAL (component = mode-average, central limit) — no UV/IR valley at 8⁴; median s collapses with smearing depth (0.37/0.12/0.045/0.02 for nsm=1–4), so use nsm=1–2
- `nambu_disloc_scaling.py` / `nambu_disloc_topo_run.py` / `disloc_60.sbatch` — **dislocation-gated nucleation amplifier** (July 20 2026): g₃ = λ·D_sm² with D_sm = Q_sm(nsm1) − Q_sm(nsm2), the differentiable proxy of the |Qclover−Q5LI| detector; mixed G with γ=1 default ("idle + supercharger": F_G = 2λD∇D is gated by the scalar D, so equilibrium (D≈small) idles as plain HMC with zero armed components, and a forming dislocation opens the gate and arms lump components — spurious floor arming is rate-suppressed since in-window at the floor forces |F_H,a| ~ floor). Depth pair **(1, 3)** chosen for cost (AD force ∝ total chain depth: ~0.9 s/force set at 8⁴/8 Mac threads vs 2.4 s for (4,10)); tradeoff: shallow charge is in the UV sign-flip regime, equilibrium D floor 0.069 vs 0.0076 for (4,10). λ=1.75 calibrated on ckpoint_lat.0: equilibrium |s| p99.9 = 5.7e-2·λ → 0.1, 7× below window edge c_p=0.75. Forces cached per U (fingerprint = inner products with fixed random probe fields — see Gotchas), halving MD cost. Scaling passed July 20 2026 (gradient 3.9e-10, reversibility 1.6e-15, dH~ε^2.04, dG~ε^2.04; test runs τ=0.5 + 100-traj thermalization — an under-thermalized 4⁴ config has the gate wide open (D≈−0.24) and armed-component Lyapunov growth e^(2√(AB)τ) keeps dH out of the ε² regime). Vital signs per trajectory: `D`, `armed%`, `rate`, `stail` (the λ knob). **THE diagnostic: acceptance conditioned on elevated |D|** — Metropolis selectively rejecting attempt trajectories makes the chain look healthy while vetoing every crossing; shorten τ if so. Q ladder on ckpoint_lat.0 (t ≈ 0.12·nsm): Q_sm noise-dominated with sign flips at nsm≤2, settles nsm≈4-5, integer plateau nsm≳17 (≈ flow t=2, where 89k-traj Qclover has median integer-distance 0.002)
- Logs (4⁴ β=1): `hmc_1k.log`, `nambu_1k.log`, `nambu_topo_1k.log`, `nambu_obc5d_1k.log`; (8⁴ β=6, pulled from Perlmutter): `hmc_60.log`, `obc5d_lin_60.log`, `obc5d_lin_60_k3.log`, `obc5d_quad_60.log`, `whist_b6.log`, `wscan_b6.log`; (local) `disloc_scaling_test.log`

**Validation results** (4⁴, β=1.0, τ=2, 1000 trajectories each, bin-16 jackknife):
- Reversibility ~1e-15 and ΔH, ΔG ~ ε² scaling for all three G choices
- ⟨plaq⟩: HMC 0.059907(350); Nambu λS-G 0.060243(318) (0.71σ); Nambu κQ-G 0.059484(319) (0.90σ); Nambu 5D-OBC 0.059912(328) (0.01σ)
- Acceptance tuned to 65–85% window in all chains to expose detailed-balance violations

**β=6 8⁴ production results (July 2026):**
- HMC baseline (89k+ traj): acceptance 0.77, ⟨|dH|⟩ 0.50 (τ=2, nsteps=50), 0.97 s/traj; equilibrium plaq 0.594231, t2E 0.1366 ± 0.0357 (per-traj step rms 0.0133 → AR(1) τ_int ≈ 14; windowed ≈ 24); tunneling rate 1/67
- 5D OBC linear-r κ₅=1 (14.2k): correctness 0.94σ, efficiency null on all observables
- 5D OBC linear-r κ₅=3 (10.5k): topology null (rate 0.0137 vs 0.0148, τ_int(Q) 33±9 vs 24.9±2.0) but real ~5σ **plaquette** speedup (τ_int 2.92 vs 4.61) with t2E *worse* (43 vs 24) — undirected stirring decorrelates UV only. Quadratic OBC chain strictly dominated (5× lower tunneling). Scaffold-as-tunneling-accelerator: negative
- Theory post-mortem: G is **conserved, not relaxed** — config-only g₃ in linear-r gives an O(κ) random tilt plus an O(κ²) tether to the *trajectory-initial* value (1-dof toy: r(t)=r₀−(κ/γ)[V(q)−V₀] exactly); no sector placement via G is possible. Haar-random scaffold forces spread energy democratically → can't concentrate the ~30 units a barrier crossing needs. The amplifier (energy *concentration* into force-selected components) is the response
- Amplifier expectations: mode-differentiation guarantees at most the FA-like floor (IR/quasi-Gaussian observables, watch τ_int(t2E) as leading indicator); the barrier-crossing ceiling (topology) has no guarantee — read the floor/ceiling split from τ_int(t2E) vs τ_int(Q) jointly
- **Deconfined-box falsification of the arming criterion** (July 17 2026, whist_b65.log: 8⁴ β=6.5, box 0.35 fm ≈ 1.9 T_c, nearly-free modes): s-histogram STILL structureless (medians 0.32/0.088/0.030/0.013 for nsm=1–4, ≈ β=6.0 values; 25% of components have s<0; heavy Cauchy-like tails) despite genuine scale separation — the per-mode prediction s(k)=(1−ck²a²)^(2n) is destroyed by site-component mixing (ratio of correlated Gaussians). **The failure is the basis, not the physics**: position-local force ratios cannot select modes in any regime; window occupancy = tail noise. Mode selection would need k-space kernels in G (= FA-class, ruled out on strategy). Position-local arming remains valid only for position-local objects → D_sm-based nucleation amplifier is the surviving variant
- **Dislocation detector** (July 17 2026, from hmc_60.log): |D| = |Qclover − Q5LI| (both already logged every trajectory, flowed t=2) is 10.6× baseline AT tunneling events and 7–9× elevated 3 measurements BEFORE — predictive barrier-proximity signal; corr with |Q−round(Q)| only 0.47 (independent info). Uses (explored, not built): high-statistics attempt-rate proxy P(|D| > thresh) for chain comparison; D-based g₃ = nucleation amplifier (quadratic scheme only — linear-r tether is anti-nucleation since D₀≈0); soft clutch λ(D(U)). Differentiable proxy: D_sm = Q_sm(nsm=1) − Q_sm(nsm=4) via existing transformed-stout chains (no differentiable 5LI in GPT)

**Production deployment (NERSC Perlmutter, July 2026):**
- CPU nodes (2× EPYC 7763): single process, `export OMP_NUM_THREADS=128` (no MPI ranks needed at 8⁴); build uses `module load PrgEnv-gnu`
- Benchmarks at 8⁴, β=6.0, τ=2, nsteps=50: HMC ~1.12 s/traj+meas (accept 0.80); linear-r OBC ~5.4 s/traj+meas (accept 0.83 — γ=1 makes dH ≈ HMC's, as theory predicts). Flow measurement (~0.55 s) costs more than HMC's MD; the 11× Nambu overhead is the S5 AD gradient ×2 per MD step over 9 leaves
- First production round: 20k trajectories per chain at β=6.0 (HMC ~6.3 h; linear ~30 h → 36 h walltime request; `regular` QOS allows 48 h max)
- sbatch pattern: `cd $SLURM_SUBMIT_DIR; module load PrgEnv-gnu; source ../gpt.env; export OMP_NUM_THREADS=128; mkdir -p $ROOT; python3 <script> --root $ROOT >> chain.log 2>&1`. Always **append** (`>>`) to a stable per-chain log — the log IS the measurement data (tau_int.py parses it); slurm `%j` files would scatter it. Auto-resume makes walltime kills harmless (`--ntraj` is the final index, so a resubmit continues and exits when done; chain legs with `sbatch --dependency=afterany:$j`)
- Checkpoint roots must exist before launch (Grid's NERSC writer does not create directories); use $SCRATCH, never /tmp (node-local RAM disk)
- **Seed new chains from an equilibrated HMC checkpoint** instead of thermalizing: `cp $SCRATCH/nambu/hmc_60/ckpoint_lat.7X000 $ROOT/ckpoint_lat.0` — auto-resume picks it up, skips `--ntherm`, and every measurement is an equilibrium sample (marginal for U is exp(−S_β) in all chains, so HMC configs are valid starts). Use different checkpoints per chain; verify with plaq ≈ 0.594, t2E ≈ 0.10–0.17 on the first measurement. Tune acceptance ON the seeded config — IR-cold configs understate dH
- Amplifier tuning: **τ is the acceptance knob, not nsteps** — armed components have positive Lyapunov exponent √(AB), so dH grows ~exponentially in τ but only polynomially improves with ε; shorten τ before refining ε. At ε=0.03 (τ=1.5/nsteps=50) equilibrium dH ≈ +0.6 → ~70–75% acceptance. Target ~77% to match the HMC baseline. Amplifier ~12 s/traj (S_sm stout-chain gradient ×2 per MD step) → needs ~10× smaller τ_int for wall-clock win; per-trajectory comparison is the mechanism test
- Amplifier vital signs: `armed` collapsing → backreaction switching mechanism off; `armed` well above prediction → chain dragged off the typical set (acceptance will die); dH systematically positive with large spikes (vs symmetric scatter) → amplified-tail signature, pull τ back — spiky rejection selectively cancels the bursty trajectories doing the useful work
- Watch item: early linear-r run showed t²E ≈ 0.085 vs HMC ≈ 0.16 in the first ~100 trajectories — both single autocorrelated samples from cold starts, not yet meaningful; verify ⟨t2E⟩ + ⟨plaq⟩ consistency in tau_int.py once the 20k runs have statistics

**Gotchas:**
- Cold-start Metropolis locks up (dH ~ +27 from unit config, rejects forever) — thermalize with `accept_reject=False`
- **Stout jacobian is 0/0-NaN at exactly-unit links** (Cayley-Hamilton coefficients) — any smeared force (S_sm, Q_sm) NaNs on a cold start; kick off identity first: `U[mu] @= g(g.matrix.exp(g(1e-2 * kick[mu])) * U[mu])` (in the production scripts, fresh starts only)
- Auto-resume grabs whatever `ckpoint_lat.N` is in `--root` — pointing at a stale tuning root silently resumes the wrong (IR-cold) config; check the first measurement's t2E
- E/t2E equilibrate ~10× slower than plaquette (IR vs UV): plaq at 1% of equilibrium ≠ thermalized. Equilibrium σ(E) is measure-fixed and identical across chains — the algorithm comparison is τ_int(E), never the fluctuation size
- `g.qcd.gauge.smear.wilson_flow(U, eps)` is ONE RK4 step; loop t/eps times for flow time t; keep flow radius √(8t) ≲ L/4
- `g.qcd.gauge.energy_density(U)` returns complex — take `.real`
- β=1 on 4⁴ shows no topological freezing (τ_int ≈ 1.4); the tunneling comparison needs β≈6
- **`g.norm2` of a group-valued field is blind to the config** (|U|² = N·V identically for unitary links) — useless as a change-detector/cache fingerprint; it only drifts at roundoff level, causing sporadic false cache hits (frozen forces → O(1) dH floor, broken reversibility). Fingerprint with inner products against fixed random probe fields instead (see `forces()` in the disloc scripts)

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

## Local (per-site) Jacobians — upstream backport, Aug 27 2026

Christoph added machinery to compute Jacobians of general gauge field transformations **locally** (an exact per-site determinant) instead of **globally** (the stochastic J†J estimator). Backported onto our pre-merge base as a Python-only change so the existing `cgpt.so` keeps working — see "Build Notes / upstream state" below.

**New in `g.qcd.gauge.smear`:**
- `parallel_transport(U, description)` — general path-based smearing from a list of `(weight, g.path()...)` per direction; a `dft_diffeomorphism`, so `jacobian()` comes for free. Reproduces `stout` exactly (map 2e-33, Jacobian 3e-29).
- `directional_parallel_transport(U, description_mu, mu, P0, P1)` — one direction, masked, with an **AD-built local Jacobian**: `jacobian_matrix()` assembles the per-site 8×8 adjoint block by reverse-mode AD, `log_det_jacobian()` takes `det`→`log`→`sum`. Matches `local_stout`'s hand-coded analytic log-det to machine precision for all μ and both checkerboards.

**The force.** Upstream shipped only the *value*; `action_log_det_jacobian_gradient` was a stub (debug `print`s, a `sum()` starting on int 0, a stubbed `diagonal_jacobian_gradient`). It is finished here (commit `d8f61c91`) using the same nested reverse-mode AD that `dft_action_log_det_jacobian` already uses for the global case: build `Φ = Σ_μ ⟨left_μ | J right_μ⟩` as a graph in U (`dft_diffeomorphism` over nodes-of-nodes) and differentiate it once more, with `left`/`right` driven by generator basis vectors and `−M⁻¹` instead of random momenta. This gives `∂(−log det J) = −tr[J⁻¹ ∂J]` exactly. Verified against `local_stout`'s analytic force to ~1e-25.

Cost at 8⁴/8 threads: dpt value 0.40 s, force 2.16 s vs `local_stout` 0.024 s / 0.22 s (~10×). Use `local_stout` for the plain stout staple; the point of dpt is that it works for an **arbitrary path set**.

**Deterministic vs stochastic.** The local action takes `(U)` only and has no `.draw()`; repeated evaluations are bit-identical. The `J†J` action takes `(U + mom)`, has `.draw()`, and its value scatters (σ ≈ 65 on a 4⁴ test) because `S = |η|²` depends on the auxiliary field, not on U. Both are **exact** algorithms — the auxiliary field exists only because `det J` of a global non-local operator is not computable; block-diagonality removes the premise, not an approximation.

### Local Jacobian for the gauge-fixing step (`gf_local_jacobian.py`)

The masked gauge-fixing step is `U'_μ(x) = V(x) U_μ(x) V†(x+μ̂)` with `V = exp(−ε P_ah B)` supported on one checkerboard. `V(x)` depends on exactly the `2·nd` links touching x and modifies exactly those same links, and every link has a unique active endpoint ⇒ the links partition into disjoint stars and **J is block diagonal over active sites**, blocks of `(2·nd links) × (ng generators)` = **32×32 in 2D, 64×64 in 4D**. Verified numerically: with the mask, exactly 8 links respond to a one-form probe; unmasked, 15 do.

**Masking is required, not optional** — unmasked, the stars overlap and no local determinant exists.

`directional_parallel_transport` does **not** cover this map (it is one-directional *left* multiplication; this is two-sided and touches all μ with the same V), so `gauge_fixing_step` is a separate class applying the same recipe with the larger block. `g.matrix.det` / `g.matrix.inv` handle `ot_matrix_singlet(32/64)` directly (verified against numpy at n = 4…32) — no new container needed.

**Two traps specific to the multi-site star** (neither appears for `local_stout`/dpt, whose star is a single site):
- `M` is **not** zero on inactive sites — the `cshift` that gathers the star reaches links of *neighbouring* active stars, leaving a scrambled block with `det ≈ −5e-29`. 32 such sites contributed −2124 to an action whose true value is −1.0995. **Mask M before regularising.**
- `inv()` of the regularised `M` is the identity on inactive sites, and those entries leak into `right`, where the star carries them back onto `left` in a neighbouring active block. **Mask `Minv` too.**

**Validated:** block vs finite differences 9.3e-12; action 1.099529914633493 vs an independent numpy block determinant 1.09952991; `assert_gradient_error` 2.2e-10; `dH ~ ε²` (per-halving ratio 6.98 → 4.77 → 4.20, last-pair slope 2.07); reversibility 1.7e-30.

**NOT yet validated — the measure.** Every test above probes the *integrator*, not the sampled distribution. A wrong sign or factor in the log-det would still give `dH ~ ε²` and reversibility 1e-30. Worse, **⟨plaquette⟩ cannot see the log-det here**: since `ft` is a gauge transformation and the plaquette is gauge-invariant, `∫dW |det J| e^{−S(W)} f(W) = ∫dU e^{−S(U)} f(U) = ∫dW e^{−S(W)} f(W)` with both normalisations equal to Z — so including or omitting the term gives identical gauge-invariant expectations. Only a gauge-**variant** observable (the link trace) can test it. The strongest available check is a link-trace comparison between this implementation and the existing J†J one on the same map. Also still open: there are no equilibrium acceptance/⟨plaq⟩ statistics.

**4D (Aug 28 2026).** Runs unchanged — `gf_local_jacobian.py` is already `nd`-generic, so 4D needed no code edits. At 4⁴, β=10, eps_gf=1e-2 the block is **64×64** per active site (2·nd links × ng generators); `ot_matrix_singlet(64)` `det`/`inv`, the 8-link star gather and both inactive-site masks all carry over. One τ=0.25 trajectory (nsteps=8, `dH_4d_single.py`): dH = −0.107 on H ≈ 7531, with dS_gauge = −47.9 cancelled by dS_mom = +47.8 — a real trajectory, not a static one. Caveat: the **log-det piece barely moves** (0.584 → 0.554, |dS| ≈ 0.03 against a gauge action of 3400), so at eps_gf=1e-2 this dH does not meaningfully stress the log-det term; raise eps_gf to test it. Cost **~67 s per log-det force** at 4⁴ on 12 threads (608 s for the trajectory's 9 forces) — the 64-sweep overhead dominates at this volume. `dH_scaling_4d.py` holds the full 4D ε² scan; not run (≈1 h).

**Fourier acceleration (Aug 28 2026).** `gf_fourier_4d.py` runs the 4D local-Jacobian gauge-fixing HMC with the Zhao (2018) FA kinetic term: `sqrt(D)` from `erik_tests/RWGFFA/fourier_kernel.py`'s `build_sqrt_fourier_kernel` (imported unchanged), handed to GPT's built-in `g.qcd.scalar.action.fourier_mass_term`. That built-in is the same construction as Erik's hand-rolled `fourier_hmc.py` trio — **`g.adj(fft)` and `g.inv(fft)` dispatch to the identical function** (`lib/gpt/core/coordinates.py`: `mat=mat_forward, adj_mat=inv_mat=mat_backward`), `scale_unitary**2 == prod(L) == V` matches the `V *` prefactor, and `g.inner_product` is `Σ_x tr(adj(a)·b)` — with two additions: it squares `sqrt(D)` numerically instead of using the closed form (valid only because `P^L P^L = P^L`, i.e. the |k̂|² normalisation fix), and it Hermitian-symmetrises the velocity. τ=0.25, nsteps=8, M=1.0, eps_fourier=0.5: dH = +0.109 (vs −0.107 for the trivial kinetic term), 614 s. Gradient error 1.4e-11, momentum group defects ~8e-15. Same caveat as above: the log-det moves only 0.023, so this does not stress the Jacobian term.

**Cost scaling.** The *sweep count* per force is `n_block = 2·nd·ng` (32 in 2D, 64 in 4D) and is volume-independent — every active site's block is computed in the same field operation. Cost per force ≈ A + B·V with A ≈ 11.6 s of per-sweep Python/graph overhead and B ≈ 2 ms/site of arithmetic, so at 8² the volume-dependent part is ~1% and at 32² still only ~15% (measured: 11.7 → 13.6 s for a 16× volume increase). Crossover around V ≈ 6000 sites, i.e. 4⁴ is overhead-dominated, 8⁴ near the knee, 16⁴ arithmetic-bound (rough — extrapolated from 2D, and in 4D `n_block` doubles while the `Minv` round-trip goes as `n_block²`). Profiled split at 2D 8², 9.87 s/force: 6.41 s for 32 nested-AD gradient calls, 2.03 s rebuilding M, 1.52 s of numpy round-trips. Unoptimised on purpose; the wins all attack A — merge the 32 nested-AD calls into one graph, cache M between action and force, keep `Minv` in lattice form.

### AD / Jacobian gotchas

- **`assert_gradient_error` does not catch a NaN gradient.** It tests `if eps > epsilon_assert`, which is `False` for NaN — so it prints `Assert gradient error: nan` and *passes*. A singular Jacobian block silently produces exactly this. Always read the printed number, don't trust the absence of an exception.
- ~~**A garbage-collected `differentiable_field_transformation` can corrupt later ones built over the same `U`.**~~ **ROOT-CAUSED Aug 31 2026: this was never about the dft or garbage collection.** The symptom (J collapses to the identity; order-dependent; `U` undamaged) came entirely from the discarded `g.identity(fm_group)` return value described under "Checkerboard Masking with AD" -- the mask was uninitialised memory, so it varied with allocation order and sometimes came back all-zero, making the map the identity. With the mask fixed, three identically-constructed steps in one process and repeated runs of the same script all agree to the last digit (`|gf(W)-W|^2 = 1.33371816e+00` every time). Keeping dfts alive is harmless but unnecessary.
- **`g.identity(x)` does not modify `x`** -- it returns a new identity lattice (`lib/gpt/core/foundation/lattice/__init__.py:114`). Calling it for its side effect is a silent no-op that leaves `x` holding whatever the allocator handed back.
- **Forward-mode AD has no `cshift_plan_add`** (upstream too), so `differentiable_field_transformation`'s `J†J` path — which uses `adjoint_jacobian` — cannot run on an `ft` built from `g.parallel_transport`. `ft`s using plain `g.cshift` (all of Erik's) are unaffected.
- `g.matrix.det`/`inv` **do** work on decomposed `ot_matrix_singlet(n)` for n = 8, 16, 32, 64 (verified against numpy) even though the fundamental sizes are only 4, 10, 30.

## Checkerboard Masking with AD

Applying a scalar checkerboard mask `fm` (a `g.complex` field) to a matrix field `B` via `B *= fm` changes the otype from `ot_matrix_su_n_fundamental_group(3)` to `ot_matrix_color(3)`. This causes a container mismatch in the AD framework's `__sub__` (which uses strict `==` instead of `accumulate_compatible`).

**Solution — group-typed mask:** Create the mask as a `su_n_fundamental_group(3)` matrix (diagonal: `fm * Identity`), so `group * group = group` preserves the otype:
```python
# g.identity(x) RETURNS a new identity lattice -- it does NOT fill x in place.
eye = g.identity(g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3)))
fm_group = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
fm_group @= fm * eye        # @= copies data, preserving fm_group's otype
B *= fm_group               # group * group = group, otype preserved
```

**The `g.identity` trap (fixed Aug 31 2026).** The earlier form of this recipe wrote
`fm_group = g.lattice(...)` / `g.identity(fm_group)` / `fm_group @= fm * fm_group` and
**discarded `g.identity`'s return value**. `identity()` in
`lib/gpt/core/foundation/lattice/__init__.py:114` allocates `eye = gpt.lattice(src)`,
fills *that*, and returns it; `src` is never touched. So `fm_group` was uninitialised
memory. Measured `|fm_group|^2` (must be exactly `n_active * Nc` = 384 at 4^4):
`384.00`, `383.88`, `383.83`, `0.00` -- varying between successive constructions in one
process and between processes with identical seeds. Downstream, three *identical*
`gauge_fixing_step(U, 1e-2, g.even)` objects gave maps of strength `1.124`, `0.932`,
`0.932`, and across runs `1.30` / `0.93` / `5.3e-24` -- the last being the identity map,
i.e. a log-det of exactly 0. Affected `gf_local_jacobian.py`, `claude_multi_step_AD.py`,
`claude_AD_multi_step.py`; all fixed. **Every masked result predating the fix is void**
(the machinery validations -- block vs finite differences, log-det vs numpy, gradient
error, `dH ~ eps^2`, reversibility -- survive, since they check the Jacobian against
whatever map was actually in force).

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

### Upstream state (Aug 27 2026)

- **`master`** — pre-merge base + the Python-only local-Jacobian backport (`20da0d73`, `d8f61c91`). Runs on the **existing** `lib/cgpt/build/cgpt.so`; no rebuild needed. Pushed as `myfork/erik-dev`.
- **`upstream-full-merge`** (`5e3fe68d`) — the complete merge with `origin/master` (`851072d7`), conflicts already resolved. **Does not run**: the merged Python calls `cgpt.lattice_rank_inner_product(a, b, n_block, use_accelerator)` (4 args) against the 3-arg symbol in the built `.so`.
- **`backup-pre-upstream-merge`** (`d75de651`) — untouched pre-merge snapshot.

**Why the full merge needs a Grid rebuild:** upstream `cgpt` uses `PlannedFFT` and `GridBLAS::inverseBatched`, absent from the Feb-2025 Grid. `dependencies/Grid` has been updated to `lehner/Grid feature/gpt` HEAD (`b11c58b5`) and reconfigured, but **its `build/` was deleted and not rebuilt** — `make -j12` in `dependencies/Grid/build/Grid`, then `./make ../../dependencies/Grid/build 10` from `lib/cgpt`. The existing `cgpt.so` is statically linked and unaffected.

Homebrew python moved 3.13 → 3.14, so the old Grid Makefile's `-F/opt/homebrew/opt/python@3.13/Frameworks -framework Python` link flags are dead — drop them (cgpt links with `-undefined dynamic_lookup`); the reconfigure already did.

Deviations we carry from upstream, both to keep the existing global-Jacobian path (`dft_action_log_det_jacobian`, used by `erik_tests/gauge-fixing-flow`) working: `dft_diffeomorphism.jacobian` keeps the old `2j*dU*U` form for AD-node inputs (upstream's `cartesian_to_infinitesimal` dereferences a not-yet-computed node value and breaks `tests/qcd/gauge.py`), and otype assignment is skipped for node inputs. Upstream independently fixed the NumPy 2.x issues below (using `[0]` / `gpt.util.to_complex`), so our `.item()` patches become obsolete on the full merge; the AD `__mul__` `g.eval(g.adj(...))` fix stays ours.

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
