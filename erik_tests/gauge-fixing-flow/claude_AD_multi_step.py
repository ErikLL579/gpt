import gpt as g
import numpy as np
import time
from gpt.ad import reverse as rad

##############################################################################
# Parameters
##############################################################################
n_gf_steps = 2          # number of sequential gauge-fixing steps
use_masking = True       # toggle checkerboard masking (set False to disable)
eps_gf = 1e-2            # gauge-fixing step size
beta = 10.0              # Wilson action coupling
L = 4                    # lattice extent
nd = 4                   # number of dimensions
tau = 1.0                # MD trajectory length
n_md_steps = 13          # leapfrog steps per trajectory
n_trajs = 5              # number of HMC trajectories

##############################################################################
# Grid and initial configuration
##############################################################################
grid = g.grid([L] * nd, g.double)
rng = g.random("seed")

U = g.qcd.gauge.unit(grid)
rng.normal_element(U)

g.message(f"Initial plaquette: {g.qcd.gauge.plaquette(U)}")

##############################################################################
# Transformation factory
##############################################################################
def make_ft(grid, nd, eps, checkerboard=None):
    """Create a gauge-fixing transformation with optional checkerboard mask.

    Args:
        grid: lattice grid
        nd: number of dimensions
        eps: step size for the gauge transformation
        checkerboard: None (no masking), g.even, or g.odd
    Returns:
        ft: callable that maps U -> U' (works on both plain fields and AD nodes)
    """
    # Precompute the group-typed mask outside the closure
    fm_group = None
    if checkerboard is not None:
        grid_cb = grid.checkerboarded(g.redblack)
        one_cb = g.complex(grid_cb)
        one_cb[:] = 1

        fm = g.complex(grid)
        fm[:] = 0
        one_cb.checkerboard(checkerboard)
        g.set_checkerboard(fm, one_cb)

        fm_inv = g.complex(grid)
        fm_inv[:] = 0
        one_cb.checkerboard(checkerboard.inv())
        g.set_checkerboard(fm_inv, one_cb)

        fm = g(fm + 1e-15 * fm_inv)

        # NOTE: g.identity(x) RETURNS a new identity lattice; it does not fill x in place
        eye = g.identity(g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3)))
        fm_group = g.lattice(grid, g.ot_matrix_su_n_fundamental_group(3))
        fm_group @= fm * eye

    def ft(U):
        # Covariant divergence: B = sum_mu (U_mu - U_mu shifted backward)
        B = U[0] - g.cshift(U[0], 0, -1)
        for mu in range(1, nd):
            B += U[mu] - g.cshift(U[mu], mu, -1)

        # Apply checkerboard mask if enabled
        if fm_group is not None:
            B *= fm_group

        # Gauge transformation: U'_mu = exp(-eps*K) * U_mu * exp(+eps*K_shifted)
        U_prime = []
        for mu in range(nd):
            U_mu_prime = g(
                g.matrix.exp(-eps * g.qcd.gauge.project.traceless_anti_hermitian(B))
                * U[mu]
                * g.matrix.exp(
                    +eps
                    * g.qcd.gauge.project.traceless_anti_hermitian(
                        g.cshift(B, mu, +1)
                    )
                )
            )
            U_prime.append(U_mu_prime)

        return U_prime

    return ft


##############################################################################
# Solvers and optimizer
##############################################################################
fr = g.algorithms.optimize.fletcher_reeves
ls2 = g.algorithms.optimize.line_search_quadratic

inverter_force = g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10)
inverter_action = g.algorithms.inverter.fgcr(eps=1e-13, maxiter=100, restartlen=10)
optimizer = g.algorithms.optimize.non_linear_cg(
    maxiter=100, eps=1e-15, step=1e-1, line_search=ls2, beta=fr
)

##############################################################################
# Create N differentiable_field_transformation objects
##############################################################################
dft_list = []
dfm_list = []
a_log_det_list = []

for k in range(n_gf_steps):
    if use_masking:
        cb = [g.even, g.odd][k % 2]
    else:
        cb = None

    ft_k = make_ft(grid, nd, eps_gf, checkerboard=cb)

    dft_k = g.qcd.gauge.smear.differentiable_field_transformation(
        U, ft_k, inverter_force, inverter_action, optimizer
    )

    dft_list.append(dft_k)
    dfm_list.append(dft_k.diffeomorphism())
    a_log_det_list.append(dft_k.action_log_det_jacobian())

g.message(f"Created {n_gf_steps} gauge-fixing steps (masking={use_masking})")

##############################################################################
# Inverse chain: find dynamical variables W from initial config U
# U -> V_{N-1} -> ... -> V_1 -> W
##############################################################################
g.message("Inverting transformation chain...")
W = g.copy(U)
for k in range(n_gf_steps - 1, -1, -1):
    g.message(f"  Inverting step {k}...")
    W = dft_list[k].inverse(W)

g.message(f"Plaquette after inversion: {g.qcd.gauge.plaquette(W)}")

##############################################################################
# Forward pass helper
##############################################################################
def forward_pass(W):
    """Compute all intermediate fields: W = V_0 -> V_1 -> ... -> V_N."""
    intermediates = [W]
    for k in range(n_gf_steps):
        V_next = dfm_list[k](intermediates[-1])
        intermediates.append(V_next)
    return intermediates


##############################################################################
# Actions and momenta
##############################################################################
U_mom = g.group.cartesian(W)

action_gauge_mom = g.qcd.scalar.action.mass_term()
action_gauge = g.qcd.gauge.action.wilson(beta)

metro = g.algorithms.markov.metropolis(rng)
sympl = g.algorithms.integrator.symplectic
log = sympl.log()

# Stochastic momenta for each log-det action (one per step)
mom2_list = [[g.group.cartesian(w) for w in W] for _ in range(n_gf_steps)]

##############################################################################
# Initial test: evaluate log-det actions
##############################################################################
intermediates = forward_pass(W)
for k in range(n_gf_steps):
    mom_k = [g.group.cartesian(w) for w in W]
    rng.normal_element(mom_k)
    mom2_list[k] = dfm_list[k].jacobian(
        intermediates[k], intermediates[k + 1], mom_k
    )
    ald_k = a_log_det_list[k](intermediates[k] + mom2_list[k])
    g.message(f"Initial log-det action (step {k}): {ald_k}")


##############################################################################
# Hamiltonian
##############################################################################
def hamiltonian(draw):
    global mom2_list

    intermediates = forward_pass(W)

    if draw:
        # Refresh HMC momenta
        rng.normal_element(U_mom)

        # Draw independent stochastic momenta for each log-det action
        for k in range(n_gf_steps):
            eta_k = [g.group.cartesian(w) for w in W]
            rng.normal_element(eta_k)
            mom2_list[k] = dfm_list[k].jacobian(
                intermediates[k], intermediates[k + 1], eta_k
            )

    # Evaluate actions
    s_gauge = action_gauge(W)
    h = s_gauge + action_gauge_mom(U_mom)

    s_logdet_total = 0.0
    for k in range(n_gf_steps):
        s_logdet_k = a_log_det_list[k](intermediates[k] + mom2_list[k])
        s_logdet_total += s_logdet_k
        h += s_logdet_k

    if draw:
        g.message(f"Gluonic action: {s_gauge}")
        g.message(f"Log-det action (total): {s_logdet_total}")

    return h, s_gauge


##############################################################################
# Force functions
##############################################################################
def log_det_force():
    """Compute total log-det force on W using Luscher backward recursion."""
    intermediates = forward_pass(W)

    # Start from outermost step
    F = a_log_det_list[n_gf_steps - 1].gradient(
        intermediates[n_gf_steps - 1] + mom2_list[n_gf_steps - 1],
        intermediates[n_gf_steps - 1],
    )

    # Backward recursion: propagate force and accumulate local contributions
    for k in range(n_gf_steps - 2, -1, -1):
        # Pull F back through transformation k
        F = dfm_list[k].jacobian(intermediates[k], intermediates[k + 1], F)
        # Add local log-det gradient at step k
        F_local = a_log_det_list[k].gradient(
            intermediates[k] + mom2_list[k], intermediates[k]
        )
        F = [g(F[mu] + F_local[mu]) for mu in range(nd)]

    return F


def gauge_force():
    return action_gauge.gradient(W, W)


##############################################################################
# Integrator
##############################################################################
iq = sympl.update_q(
    W, log(lambda: action_gauge_mom.gradient(U_mom, U_mom), "gauge_mom")
)

ip_gauge = sympl.update_p(
    U_mom, lambda: log(gauge_force, "gauge")()
)

ip_log_det = sympl.update_p(
    U_mom, lambda: log(log_det_force, "log_det")()
)

mdint = sympl.leap_frog(1, ip_log_det, sympl.leap_frog(1, ip_gauge, iq))

g.message(f"Integration scheme:\n{mdint}")


##############################################################################
# HMC
##############################################################################
def hmc(tau, no_accept_reject=False):
    accrej = metro(W)

    h0, s0 = hamiltonian(True)

    for i in range(n_md_steps):
        mdint(tau / n_md_steps)

    h1, s1 = hamiltonian(False)

    if no_accept_reject:
        return True, s1 - s0, h1 - h0
    else:
        return accrej(h1, h0), s1 - s0, h1 - h0


##############################################################################
# Main loop
##############################################################################
accept, total = 0, 0
plaq_measurements = []

start = time.time()

for it in range(n_trajs):
    no_ar = it < 3  # disable accept/reject for first 3 trajectories
    g.message(f"\n--- Trajectory {it} (no_accept_reject={no_ar}) ---")

    a, dS, dH = hmc(tau, no_accept_reject=no_ar)
    accept += a
    total += 1

    plaq = g.qcd.gauge.plaquette(W)
    plaq_measurements.append(plaq)

    g.message(
        f"plaq={plaq:.6f}, dS={dS:.6e}, dH={dH:.6e}, "
        f"acceptance={accept/total:.2f}"
    )
    g.message(f"Timing:\n{log.time}")

end = time.time()
g.message(f"\nTotal time: {end - start:.1f} seconds")

p = np.array(plaq_measurements)
discard = min(1, len(p) - 1)
g.message(
    f"Avg plaq = {np.mean(p[discard:]):.6f}, "
    f"std = {np.std(p[discard:]):.6f}"
)
