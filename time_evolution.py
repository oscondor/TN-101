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

#gate est une matrice 4x4 représentant une porte quantique à 2 sites, on veut l'appliquer à un MPS en contractant les bons indices

from scipy.linalg import expm

def make_gate(h_local, dt, imaginary_time=False):
    """
    h_local : matrice (4,4) du hamiltonien local h_{i,i+1}
    Retourne la gate e^{-i h dt} (ou e^{-h dt} si imaginaire)
    """
    sign = 1.0 if imaginary_time else 1j
    return expm(-sign * h_local * dt)  # (4,4)

def apply_gate(gate, i, mps, tol):
    mps.shift_center(i, None)
    tensor_gate = gate.reshape(2, 2, 2, 2)  # (i_in, i+1_in, i_out, i+1_out)
    mps_core_1 = mps.cores[i]      # (bl, d, br)
    mps_core_2 = mps.cores[i+1]    # (bl, d, br)

    mps_fused = np.tensordot(mps_core_1, mps_core_2, axes=(2, 0))  # (bl, d, d, br)
    bloc = np.tensordot(tensor_gate, mps_fused, axes=([0, 1], [1, 2]))  # (i_out, i+1_out, bl, br)
    bloc = bloc.transpose(2, 0, 1, 3)  # (bl, i_out, i+1_out, br)

    u, s, vt = mps_lib.svd_for_tensor_compression(bloc, 2, tol)
    chi = u.shape[1]

    mps.cores[i]   = (u * s[np.newaxis, :]).reshape(mps_core_1.shape[0], 2, chi)
    mps.cores[i+1] = vt.reshape(chi, 2, mps_core_2.shape[2])

def apply_multiple_gates(gates, mps, tol):
    for gate_info in gates:
        gate, i = gate_info
        apply_gate(gate, i, mps, tol)

def time_evolution(gates, mps_init, dt, N,tol):
    #gates : list of dictionaries with keys 'gate' (4x4 matrix) and 'site' (int) and list bcs we need to apply separately the gates that do not commute
    for _ in range(N):
        for gate_info in gates:
            apply_multiple_gates(gate_info, mps_init, tol)

 
        



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

    mpo = TB_hamiltonian_interaction(h, n_sites, tol=20)
    print(f"MPO constructed for n_sites={n_sites}.")

    # Calculer la fermi sea avec DMRG
    mps_init = random_mps(n_sites, d=2, chi_max=chi_max)
    mps_init = DMRG.dmrg(mpo, mps_init, num_sweeps=10, tol=1e-8)

    # Ajouter c0dag à la fermi sea
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
    gate = make_gate(h_local, dt, imaginary_time=False)

    gate_info_even = [(gate, i) for i in range(n_sites - 1) if i % 2 == 0]
    gate_info_odd = [(gate, i) for i in range(n_sites - 1) if i % 2 == 1]
    gates = [gate_info_even, gate_info_odd]

    bond_dims = np.zeros(n_steps + 1, dtype=int)
    bond_dims[0] = max_bond_dimension(mps_init)

    for t in range(1, n_steps + 1):
        print_run_progress(t, n_steps)
        time_evolution(gates, mps_init, dt=dt, N=1, tol=1e-8)
        bond_dims[t] = max_bond_dimension(mps_init)

    print_run_progress(n_steps, n_steps)
    print()

    time_array = np.arange(n_steps + 1) * dt
    return time_array, bond_dims


dt = 0.01
total_time = 40.0
sizes = [10, 14, 20]

plt.figure(figsize=(8, 5))
for n_sites in sizes:
    time_array, bond_dims = run_simulation(n_sites, total_time=total_time, dt=dt, chi_max=16)
    plt.plot(time_array, bond_dims, label=f"N = {n_sites}")

plt.xlabel("Temps")
plt.ylabel("Dimension de liaison max")
plt.title("Dimension de liaison maximale en fonction du temps")
plt.xlim(0, total_time)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()