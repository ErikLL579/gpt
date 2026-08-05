#
# Shared measurement + checkpoint helpers for the topological tunneling
# comparison runs (hmc_topo_run.py, nambu_obc5d_topo_run.py).
#
# Wilson flow (RK4 gradient flow) to flow time t, then measure:
#   Q5LI    - O(a^4)-improved topological charge (5-loop-improved)
#   Qclover - clover-leaf topological charge
#   E       - flowed energy density (clover field strength), and t^2 E
#
# Flow-radius guidance: smoothing radius = sqrt(8 t). Keep < L/4.
#
import os
import gpt as g


def flow_and_measure(U, t_flow, eps_flow):
    Uf = g.copy(U)
    n = int(round(t_flow / eps_flow))
    for _ in range(n):
        Uf = g.qcd.gauge.smear.wilson_flow(Uf, eps_flow)
    Q5 = g.qcd.gauge.topological_charge_5LI(Uf)
    Qc = g.qcd.gauge.differentiable_topology(Uf).real
    E = complex(g.qcd.gauge.energy_density(Uf)).real
    return {
        "Q5LI": Q5,
        "Qclover": Qc,
        "E": E,
        "t2E": t_flow * t_flow * E,
    }


def measurement_line(i, m):
    return (
        f"Measurement {i}: Q5LI = {m['Q5LI']:+.6f}, Qclover = {m['Qclover']:+.6f}, "
        f"E = {m['E']:.8e}, t2E = {m['t2E']:.6f}"
    )


def latest_checkpoint(root, n_max):
    latest = None
    for it in range(n_max):
        if os.path.exists(f"{root}/ckpoint_lat.{it}"):
            latest = it
    return latest


def save_checkpoint(root, it, U):
    g.save(f"{root}/ckpoint_lat.{it}", U, g.format.nersc())


def load_tau_state(root):
    path = f"{root}/tau_state.txt"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tau, tuning_done = f.read().split()
    return float(tau), bool(int(tuning_done))


def save_tau_state(root, tau, tuning_done):
    with open(f"{root}/tau_state.txt", "w") as f:
        f.write(f"{tau} {int(tuning_done)}\n")
