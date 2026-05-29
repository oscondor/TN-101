import numpy as np
import mps_library as mps_lib
import mps_class
import mpo_class
import DMRG
from fermions_1d import cdag_fermions
from fermions_1d import random_mps
from fermions_1d import TB_hamiltonian_interaction
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from plot_analytic import compute_nj
import time_evolution as tevo

def print_run_progress(step, total_steps):
    pct = 100.0 * step / total_steps
    print(f"Progression du run: {step}/{total_steps} ({pct:.1f}%)", end="\r", flush=True)

def max_bond_dimension(mps):
    if len(mps.cores) < 2:
        return 1
    return max(core.shape[2] for core in mps.cores[:-1])

def bond_dimensions(mps):
    if len(mps.cores) < 2:
        return np.array([], dtype=int)
    return np.array([core.shape[2] for core in mps.cores[:-1]], dtype=int)

def run_simulation(n_sites, total_time=40.0, dt=0.01, chi_max=16):
    n_steps = int(round(total_time / dt))

    # Hamiltonien tight-binding complet : saut i <-> i+1 pour tout i
    h = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        h[i, i + 1] = 1.0
        h[i + 1, i] = 1.0

    mpo = TB_hamiltonian_interaction(h, n_sites, tol=1e-8)
    print(f"MPO constructed for n_sites={n_sites}.")

    # Calculer la fermi sea avec DMRG
    mps_init = random_mps(n_sites, chi_max, d=2)
    mps_init = DMRG.dmrg(mpo, mps_init, num_sweeps=10, tol=1e-8)
    print("chi_max after DMRG: ", max_bond_dimension(mps_init))
    # Ajouter c0dag à la fermi sea
    mps_init.shift_center(0, None)
    c0dag_mpo = cdag_fermions(0, n_sites)
    mps_init = c0dag_mpo.apply_to_mps(mps_init, tol=1e-8)
    # c†₀ n'est pas unitaire : ||c†₀|ψ⟩||² = 1 - ⟨n₀⟩ < 1, il faut renormaliser
    mps_init.cores[mps_init.center] /= mps_init.norm()

    E0 = mpo.expectation_value(mps_init)
    print(f"Energy with hamiltonian (n_sites={n_sites}): {E0:.10f}")

    # Hamiltonien local à 2 sites pour la porte TEBD
    h_local = np.zeros((4, 4), dtype=complex)
    h_local[1, 2] = 1.0
    h_local[2, 1] = 1.0
    gate = tevo.make_gate(h_local, dt, imaginary_time=False)

    gate_info_even = [(gate, i) for i in range(n_sites - 1) if i % 2 == 0]
    gate_info_odd = [(gate, i) for i in range(n_sites - 1) if i % 2 == 1]
    gates = [gate_info_even, gate_info_odd]

    bond_dims = np.zeros(n_steps + 1, dtype=int)
    error_sq_sum = None
    bond_dims[0] = max_bond_dimension(mps_init)
    print(f"Initial max bond dimension: {bond_dims[0]}")

    if n_sites == 20:
        analytic_nrt = np.zeros((n_steps + 1, n_sites), dtype=float)
        for t in range(n_steps + 1):
            analytic_nrt[t] = np.real(compute_nj(n_sites, t * dt))
        error_sq_sum = np.zeros(n_steps + 1, dtype=float)
        error_sq_sum[0] = np.sum((np.real([n_r_mpo.expectation_value(mps_init) for n_r_mpo in []])) ** 2)

    for t in range(1, n_steps + 1):
        print_run_progress(t, n_steps)
        tevo.time_evolution(gates, mps_init, dt=dt, N=1, tol=1e-6)
        bond_dims[t] = max_bond_dimension(mps_init)

        if n_sites == 20:
            nrt_t = np.zeros(n_sites, dtype=float)
            for r in range(n_sites):
                hr = np.zeros((n_sites, n_sites))
                hr[r, r] = 1.0
                n_r_mpo = TB_hamiltonian_interaction(hr, n_sites, tol=1e-8)
                nrt_t[r] = np.real(n_r_mpo.expectation_value(mps_init))
            error_sq_sum[t] = np.sum((nrt_t - analytic_nrt[t]) ** 2)

    print_run_progress(n_steps, n_steps)
    print()

    time_array = np.arange(n_steps + 1) * dt
    return time_array, bond_dims, error_sq_sum


dt = 0.1
total_time = 10.0
sizes = [20]

plt.figure(figsize=(8, 5))
error_20 = None
for n_sites in sizes:
    time_array, bond_dims, error_sq_sum = run_simulation(n_sites, total_time=total_time, dt=dt)
    plt.plot(time_array, bond_dims, label=f"N = {n_sites}")
    if n_sites == 20:
        error_20 = (time_array, error_sq_sum)

plt.xlabel("Temps")
plt.ylabel("Dimension de liaison max")
plt.title("chi_max(t)")
plt.xlim(0, total_time)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

if error_20 is not None:
    time_array, error_sq_sum = error_20
    plt.figure(figsize=(8, 5))
    plt.plot(time_array, error_sq_sum, color="C4")
    plt.xlabel("Temps")
    plt.ylabel(r'$\sum_i (n_i^{num} - n_i^{ana})^2$')
    plt.title("Erreur quadratique totale pour L = 20")
    plt.xlim(0, total_time)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

