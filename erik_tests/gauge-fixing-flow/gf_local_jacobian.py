#!/usr/bin/env python3
#
# One-step gauge-fixing flow HMC with an EXACT LOCAL Jacobian.
#
# Port of Trying_AD.py: same map, same HMC, but the log-det Jacobian comes from
# a per-site block determinant instead of the global stochastic J^dag J
# estimator -- so there is no auxiliary momentum field.
#
# The masked step is a gauge transformation U'_mu(x) = V(x) U_mu(x) V^dag(x+mu)
# with V supported on one checkerboard.  V(x) depends on exactly the 8 links
# touching x, and modifies exactly those same 8 links, so the Jacobian is block
# diagonal over active sites with blocks of size (2*nd links) x (ng generators).
#
import gpt as g
import numpy as np
from gpt.ad import reverse as rad
from gpt.core.group import differentiable_functional
from gpt.qcd.gauge.smear.differentiable import dft_diffeomorphism


class gauge_fixing_step:
    def __init__(self, U, eps, checkerboard=g.even):
        grid = U[0].grid
        nd = len(U)
        self.nd = nd
        self.eps = eps
        self.grid = grid
        self.otype = U[0].otype
        self.cartesian = self.otype.cartesian()
        self.generators = self.cartesian.generators(grid.precision.complex_dtype)
        self.ng = len(self.generators)
        self.trace_norm = (self.generators[0].array @ self.generators[0].array).trace()
        self.idx = (slice(None),) * nd

        # active-site mask
        grid_cb = grid.checkerboarded(g.redblack)
        one_cb = g.complex(grid_cb)
        one_cb[:] = 1
        P = g.complex(grid)
        P[:] = 0
        one_cb.checkerboard(checkerboard)
        g.set_checkerboard(P, one_cb)
        self.P = P
        self.iP = g.complex(grid)
        self.iP[:] = 1
        self.iP -= P

        # group-typed mask, so that group * group keeps the otype through AD
        # NOTE: g.identity(x) RETURNS a new identity lattice, it does not fill x
        # in place -- discarding the return value leaves fm_group uninitialised.
        eye = g.identity(g.lattice(grid, self.otype))
        fm_group = g.lattice(grid, self.otype)
        fm_group @= P * eye

        def ft(xU):
            B = xU[0] - g.cshift(xU[0], 0, -1)
            for mu in range(1, nd):
                B += xU[mu] - g.cshift(xU[mu], mu, -1)
            B *= fm_group
            out = []
            for mu in range(nd):
                out.append(
                    g(
                        g.matrix.exp(-eps * g.qcd.gauge.project.traceless_anti_hermitian(B))
                        * xU[mu]
                        * g.matrix.exp(
                            +eps
                            * g.qcd.gauge.project.traceless_anti_hermitian(g.cshift(B, mu, +1))
                        )
                    )
                )
            return out

        self.ft = ft
        self.dfm = dft_diffeomorphism(U, ft)

        # the 2*nd links of one active-site star:
        #   side 0 -> U_nu(x)      (left-multiplied by V(x))
        #   side 1 -> U_nu(x-nu)   (right-multiplied by V^dag(x))
        self.slots = [(nu, side) for side in range(2) for nu in range(nd)]
        self.slot_mask = [
            P if side == 0 else g.eval(g.cshift(P, nu, +1)) for nu, side in self.slots
        ]
        self.n_block = len(self.slots) * self.ng
        self._ljr = None

    # ---------------------------------------------------------------- forward
    def __call__(self, fields):
        return self.dfm(fields)

    # ------------------------------------------------------- local Jacobian
    def _row(self, fields, fields_prime, nu, v):
        """One row of J: contract the output one-form v (on component nu) back
        onto every input link.  Reverse-mode AD gives J^T v directly."""
        seed = g(2j * v * fields_prime[nu])
        n = len(self.dfm.aU)
        for k in range(n):
            self.dfm.aU[k].value = fields[k]
            self.dfm.aU[k].zero_gradient()
        self.dfm.aUft[nu](initial_gradient=seed)
        out = []
        for k in range(n):
            gr = self.dfm.aU[k].gradient
            gr.otype = self.cartesian
            out.append(gr)
        return out

    def jacobian_matrix(self, fields):
        """Per-active-site block of the Jacobian, gathered onto the active site."""
        fields_prime = self(fields)
        ng, idx = self.ng, self.idx
        M = g.lattice(self.grid, g.ot_matrix_singlet(self.n_block))
        M[:] = 0
        src = g.group.cartesian(fields[0])

        for I, (nu_i, _) in enumerate(self.slots):
            for a in range(ng):
                src @= self.slot_mask[I] * self.generators[a]
                row = self._row(fields, fields_prime, nu_i, src)
                for J, (nu_j, side_j) in enumerate(self.slots):
                    r = row[nu_j]
                    if side_j == 1:
                        # link sits at x - nu_j; bring it to the active site x
                        r = g.eval(g.cshift(r, nu_j, -1))
                        r.otype = self.cartesian
                    coor = self.cartesian.coordinates(r)
                    for b in range(ng):
                        M[idx + (I * ng + a, J * ng + b)] = coor[b][:]
        return M

    def _regular_M(self, fields):
        """Only active sites carry a meaningful block.  Inactive sites hold a
        scrambled gather (the cshift readout reaches links of neighbouring
        active stars), so mask them away and put the identity there instead --
        det = 1, log det = 0, and they drop out of every contraction."""
        M = self.jacobian_matrix(fields)
        return g(self.P * M + self.iP * g.identity(M))

    def log_det_jacobian(self, fields):
        M = self._regular_M(fields)
        return g(g.component.real(g.component.log(g.matrix.det(M))))

    def action_log_det_jacobian(self):
        return gf_action_log_det_jacobian(self)

    # ------------------------------------------------------------- the force
    def _left_J_right_functional(self):
        """<left | J right> as an AD graph in U; differentiating it once more
        gives tr[J^-1 dJ].  Same construction as dft_action_log_det_jacobian,
        driven by generator basis vectors rather than random momenta."""
        U0 = [u.value for u in self.dfm.aU]
        n = len(U0)
        _U = [rad.node(g.copy(u)) for u in U0]
        dfm_node = dft_diffeomorphism(_U, self.ft)
        mom = [g.group.cartesian(u) for u in U0]
        _left = [rad.node(g.copy(m), with_gradient=False) for m in mom]
        _right = [rad.node(g.copy(m), with_gradient=False) for m in mom]
        _Up = dfm_node(_U)
        J_right = dfm_node.jacobian(_U, _Up, _right)
        act = None
        for nu in range(n):
            term = g.inner_product(_left[nu], J_right[nu])
            act = term if act is None else g(act + term)
        return act.functional(*(_U + _left + _right))

    def action_log_det_jacobian_gradient(self, fields, dfields):
        if self._ljr is None:
            self._ljr = self._left_J_right_functional()

        ng, idx, nd = self.ng, self.idx, self.nd
        # inv() of the regularised M is the identity on inactive sites; those
        # entries must not reach `right`, because the 8-link star would carry
        # them back onto `left` in a neighbouring active block
        Minv = g(self.P * g.matrix.inv(self._regular_M(fields)))

        gradient = None
        cf = g.complex(self.grid)
        for I, (nu_i, _) in enumerate(self.slots):
            for a in range(ng):
                left = [g.group.cartesian(fields[mu]) for mu in range(nd)]
                right = [g.group.cartesian(fields[mu]) for mu in range(nd)]
                for mu in range(nd):
                    left[mu][:] = 0
                    right[mu][:] = 0
                left[nu_i] @= self.slot_mask[I] * self.generators[a]

                for J, (nu_j, side_j) in enumerate(self.slots):
                    for b in range(ng):
                        cf[idx] = Minv[idx + (I * ng + a, J * ng + b)]
                        c = g.eval(g.cshift(cf, nu_j, +1)) if side_j == 1 else cf
                        right[nu_j] += (-1.0 / self.trace_norm) * c * self.generators[b]

                gr = self._ljr.gradient(list(fields) + left + right, fields)
                gradient = gr if gradient is None else [g(x + y) for x, y in zip(gradient, gr)]

        return [gradient[list(fields).index(d)] for d in dfields]


class gf_action_log_det_jacobian(differentiable_functional):
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, fields):
        return -g.sum(self.parent.log_det_jacobian(fields)).real

    def gradient(self, fields, dfields):
        return self.parent.action_log_det_jacobian_gradient(fields, dfields)
