#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g
from gpt.qcd.gauge.smear.differentiable import dft_diffeomorphism
from .differentiable import assert_compatible
from gpt.core.group import differentiable_functional


class directional_parallel_transport(dft_diffeomorphism):
    def __init__(self, U, description_mu, mu, P0=None, P1=None, parameters=[]):
        self.description_mu = description_mu
        self.mu = mu
        self.P0 = P0
        self.P1 = P1
        self.left_J_right = None

        parameter_indices = [
            parameters.index(weight) if weight in parameters else None
            for weight, path in description_mu
        ]
        nd = len(U)
        np = len(parameters)
        ntot = nd + np
        fields = U + parameters

        cache = {}

        def ft(xU):
            assert len(xU) == ntot

            cache_key = f"{type(xU[0])}"
            if cache_key not in cache:
                paths = [y[1] for y in description_mu]
                cache[cache_key] = g.parallel_transport(xU[0:nd], paths)

            pt = cache[cache_key]
            if P0 is not None:
                xU_P0 = [g(xU[i] * P0) if i == mu else xU[i] for i in range(nd)]
            else:
                xU_P0 = xU[0:nd]

            xparams = xU[nd:]
            assert len(xparams) == np

            sU = list(pt(xU_P0))
            sm = None
            idx = 0
            for weight, path in description_mu:
                if weight in parameters:
                    assert not isinstance(weight, g.ad.reverse.node_base)
                    weight = xparams[parameter_indices[idx]]
                xp = g(weight * sU[idx])
                if sm is None:
                    sm = xp
                else:
                    sm += xp
                idx += 1

            assert sm is not None

            if P1 is not None:
                sm *= P1

            sm = g(g.matrix.exp(g.qcd.gauge.project.traceless_anti_hermitian(sm)) * xU[mu])
            return [sm if i == mu else xU[i] for i in range(nd)] + xparams

        super().__init__(fields, ft)

    def diagonal_jacobian(self, fields, fields_prime, dfields_mu):
        mu = self.mu
        N = len(fields_prime)
        assert len(fields) == N
        aU_prime_mu = g.cartesian_to_infinitesimal(fields_prime[mu], dfields_mu)
        for nu in range(len(fields)):
            assert_compatible(self.aU[nu].value, fields[nu])
            self.aU[nu].value = fields[nu]
        self.aUft[mu](initial_gradient=aU_prime_mu)
        self.aU[mu].gradient.otype = dfields_mu.otype
        return g(self.aU[mu].gradient * self.P1)

    def jacobian_matrix(self, fields):
        fields_prime = self(fields)
        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(dt)
        src = g.group.cartesian(fields[0])
        M = g.lattice(grid, g.ot_matrix_su_n_adjoint_algebra(otype.Nc))

        for a in range(len(generators)):
            src @= self.P1 * generators[a]
            dst = self.diagonal_jacobian(fields, fields_prime, src)
            coor = otype_cartesian.coordinates(dst)
            for b in range(len(generators)):
                M[:, :, :, :, a, b] = coor[b][:]
        return M

    def log_det_jacobian(self, fields):
        M = self.jacobian_matrix(fields)
        M_det = g.matrix.det(M)
        M_log_det = g.component.log(M_det)
        zero = g.lattice(M_log_det)
        zero[:] = 0
        M_log_det = g.where(self.P1, M_log_det, zero)
        return g.sum(M_log_det)

    def action_log_det_jacobian(self):
        return dpt_action_log_det_jacobian(self)

    def _left_J_right_functional(self):
        # Nested reverse-mode AD: build the scalar  sum_mu <left_mu | J right_mu>
        # as a graph in U, so that differentiating it once more gives
        # tr[J^-1 dJ] -- the same trick dft_action_log_det_jacobian uses for the
        # global stochastic log-det, here driven by generator basis vectors
        # instead of random momenta.
        rad = g.ad.reverse

        U0 = [u.value for u in self.aU]
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
        # det(J_{ab} + drho_c \partial_{rho_c} J_{ab}) = det(J) (1 + J^-1_{ba} drho_c \partial_{rho_c} J_{ab})
        # -> \partial_{rho_c} det(J) = det(J) J^-1_{ba} \partial_{rho_c} J_{ab}
        # \partial -\log \det(J) = -1/det(J) \partial det(J)
        # Compute tr[\partial_rho (\partial_U f) M]
        assert list(dfields) == list(fields)

        if self.left_J_right is None:
            self.left_J_right = self._left_J_right_functional()

        mu = self.mu
        n = len(fields)

        grid = fields[0].grid
        dt = grid.precision.complex_dtype
        otype = fields[0].otype
        otype_cartesian = otype.cartesian()
        generators = otype_cartesian.generators(dt)
        ng = len(generators)

        # <T_a|X> = trace_norm * coordinates_a(X), and jacobian_matrix is built
        # from coordinates(), so undo that normalization here
        trace_norm = (generators[0].array @ generators[0].array).trace()

        # off-mask sites have J = 0 (the map is the identity there), which would
        # make the inverse singular; put the identity there instead -- those sites
        # drop out anyway because left = P1 * T_a vanishes on them
        M = self.jacobian_matrix(fields)
        imask = g.complex(grid)
        imask[:] = 1
        imask -= self.P1
        M = g(M + imask * g.identity(M))
        Minv = g.separate_color(g.matrix.inv(M))

        zero = [g.group.cartesian(fields[i]) for i in range(n)]
        for z in zero:
            z[:] = 0

        left = g.group.cartesian(fields[mu])
        right = g.group.cartesian(fields[mu])

        gradient = None
        for a in range(ng):
            left @= self.P1 * generators[a]

            right[:] = 0
            for b in range(ng):
                right += (-1.0 / trace_norm) * Minv[a, b] * generators[b]

            L = [left if i == mu else zero[i] for i in range(n)]
            R = [right if i == mu else zero[i] for i in range(n)]

            gr = self.left_J_right.gradient(list(fields) + L + R, fields)
            if gradient is None:
                gradient = gr
            else:
                gradient = [g(x + y) for x, y in zip(gradient, gr)]

        return [gradient[list(fields).index(d)] for d in dfields]


class dpt_action_log_det_jacobian(differentiable_functional):
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, fields):
        return -self.parent.log_det_jacobian(fields).real

    def gradient(self, fields, dfields):
        return self.parent.action_log_det_jacobian_gradient(fields, dfields)
