def div_A(U):
    a = g.qcd.gauge.fix.landau(U)
    V = g.mcolor(grid) # V is a SU(3) matrix on every lattice site
    rng.normal_element(V, scale=0.00)
    return a.gradient(V, 0.) * 1j

def exp_div_A(fields, delta_t):
    a = g.matrix.exp(  - delta_t * div_A(fields))
    V = g.mcolor(grid)
    a.otype = V.otype
    return a