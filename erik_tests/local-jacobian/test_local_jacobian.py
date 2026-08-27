#!/usr/bin/env python3
#
# Tests for Christoph's local (per-site) Jacobian machinery for general
# gauge field transformations, backported from lehner/gpt master:
#
#   g.qcd.gauge.smear.parallel_transport             general path-based smearing
#   g.qcd.gauge.smear.directional_parallel_transport same, one direction + mask,
#                                                    with an AD-built LOCAL Jacobian
#
# The point: for a masked single-direction transformation the Jacobian is block
# diagonal in position, so log|det J| = sum_x log det J(x) with J(x) an 8x8
# adjoint matrix built by reverse-mode AD -- no global stochastic estimator.
#
# Test 1: general parallel_transport reproduces stout smearing (map + jacobian)
# Test 2: directional_parallel_transport reproduces local_stout's hand-coded
#         analytic local log-det Jacobian, for all mu and both checkerboards
#
import gpt as g

L = g.default.get_int("--L", 8)
grid = g.grid([L, L, L, L], g.double)
rng = g.random("local-jacobian")
U = g.qcd.gauge.random(grid, rng)
rho = 0.1


def even_odd_projectors(grid):
    even, odd = [g.complex(grid) for _ in range(2)]
    one_half = g.pick_checkerboard(g.even, even)
    one_half[:] = 1
    even[:] = 0
    odd[:] = 0
    one_half.checkerboard(g.even)
    g.set_checkerboard(even, one_half)
    one_half.checkerboard(g.odd)
    g.set_checkerboard(odd, one_half)
    return even, odd


def staple_paths(mu, rho):
    return [(rho, g.path().f(nu).f(mu).b(nu).b(mu)) for nu in range(4) if nu != mu] + [
        (rho, g.path().b(nu).f(mu).f(nu).b(mu)) for nu in range(4) if nu != mu
    ]


g.message("=" * 70)
g.message("Test 1: general parallel_transport versus stout")

pt = g.qcd.gauge.smear.parallel_transport(U, [staple_paths(mu, rho) for mu in range(4)])
st = g.qcd.gauge.smear.stout(rho=rho)

st0, st1 = pt(U), st(U)
A = [g.group.cartesian(U[0]) for _ in range(4)]
rng.element(A)
Aprime0 = pt.jacobian(U, st0, A)
Aprime1 = st.jacobian(U, st1, A)

eps = sum(g.norm2(x - y) for x, y in zip(st0, st1)) / sum(g.norm2(x) for x in st0)
g.message(f"  smearing: {eps}")
assert eps < 1e-25

eps = sum(g.norm2(x - y) for x, y in zip(Aprime0, Aprime1)) / sum(g.norm2(x) for x in Aprime0)
g.message(f"  jacobian: {eps}")
assert eps < 1e-25

g.message("=" * 70)
g.message("Test 2: directional_parallel_transport local log-det versus local_stout")

even, odd = even_odd_projectors(grid)

for mu in range(4):
    for cb, P1 in [(g.even, even), (g.odd, odd)]:
        dpt = g.qcd.gauge.smear.directional_parallel_transport(
            U, staple_paths(mu, rho), mu, P0=None, P1=P1
        )
        ls = g.qcd.gauge.smear.local_stout(rho=rho, dimension=mu, checkerboard=cb)

        eps = sum(g.norm2(a - b) for a, b in zip(dpt(U), ls(U))) / sum(g.norm2(a) for a in ls(U))
        g.message(f"  [mu={mu},{cb.__name__}] forward map: {eps}")
        assert eps < 1e-24

        ldj_dpt = dpt.log_det_jacobian(U)
        ldj_ls = g.sum(ls.log_det_jacobian(U))
        rel = abs(ldj_dpt - ldj_ls) / abs(ldj_ls)
        g.message(
            f"  [mu={mu},{cb.__name__}] log|det J| = {ldj_dpt.real:.12f}"
            f" vs local_stout {ldj_ls.real:.12f} : rel = {rel:.3e}"
        )
        assert rel < 1e-12

g.message("=" * 70)
g.message("All local-Jacobian tests passed")

# NOTE (as of upstream 851072d7): directional_parallel_transport gives the VALUE
# of the local log-det Jacobian only.  Its force,
#
#   dpt.action_log_det_jacobian().gradient(U, U)
#
# is unfinished upstream -- action_log_det_jacobian_gradient() still contains
# debug print()s, a `sum()` that starts on int 0, and a stubbed
# diagonal_jacobian_gradient() whose forward pass is replaced by `aaUft = aaU`.
# For HMC in gauge-fixed / smeared variables today, use local_stout, which has
# both the analytic local log-det and its gradient.
