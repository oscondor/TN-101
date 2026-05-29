import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from scipy.linalg import expm
import quimb.tensor as qtn
qtn.set_contract_backend('numpy')

# --- imports de ton propre code ---
import mps_class, DMRG
from fermions_1d import random_mps, TB_hamiltonian_interaction, cdag_fermions

def run_simulation_quimb(n_sites, total_time=10.0, dt=0.1, chi_max=32):
    n_steps = int(round(total_time / dt))

    # GS via ton propre DMRG
    h = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        h[i, i+1] = h[i+1, i] = 1.0
    mpo = TB_hamiltonian_interaction(h, n_sites, tol=1e-8)
    mps = random_mps(n_sites, chi_max, d=2)
    mps = DMRG.dmrg(mpo, mps, num_sweeps=10, tol=1e-10)

    # c†_0
    mps.shift_center(0, None)
    c0dag_mpo = cdag_fermions(0, n_sites)
    mps = c0dag_mpo.apply_to_mps(mps, tol=1e-8)
    mps.cores[mps.center] /= mps.norm()

    print("Shapes originaux:", [c.shape for c in mps.cores])
    arrays = []
    for idx, core in enumerate(mps.cores):
        t = core.transpose(1, 0, 2)  # (phys, bl, br)
        if idx == 0:
            t = t.squeeze(1)          # (phys, br)
        elif idx == len(mps.cores) - 1:
            t = t.squeeze(2)          # (phys, bl)
        arrays.append(t)

    print("Shapes après squeeze:", [a.shape for a in arrays])
    psi = qtn.MatrixProductState(arrays)

    # Gates TEBD
    h_local = np.zeros((4, 4), dtype=complex)
    h_local[1, 2] = h_local[2, 1] = 1.0
    gate      = expm(-1j * h_local * dt    ).reshape(2, 2, 2, 2)
    gate_half = expm(-1j * h_local * dt / 2).reshape(2, 2, 2, 2)

    even_bonds = list(range(0, n_sites - 1, 2))
    odd_bonds  = list(range(1, n_sites - 1, 2))

    bond_dims = np.zeros(n_steps + 1, dtype=int)
    bond_dims[0] = psi.max_bond()

    for step in range(1, n_steps + 1):
        for i in even_bonds:
            psi.gate_split_(gate_half, (i, i+1), max_bond=chi_max, cutoff=1e-10)
        for i in odd_bonds:
            psi.gate_split_(gate, (i, i+1), max_bond=chi_max, cutoff=1e-10)
        for i in even_bonds:
            psi.gate_split_(gate_half, (i, i+1), max_bond=chi_max, cutoff=1e-10)

        bond_dims[step] = psi.max_bond()
        print(f"  step {step}/{n_steps} | chi_max = {bond_dims[step]}", end="\r")

    print()
    return np.arange(n_steps + 1) * dt, bond_dims


dt = 0.1
total_time = 10.0
sizes = [10, 12, 14, 20]

plt.figure(figsize=(8, 5))
for n_sites in sizes:
    t, chi = run_simulation_quimb(n_sites, total_time=total_time, dt=dt)
    plt.plot(t, chi, label=f"N = {n_sites}")

plt.xlabel("Temps")
plt.ylabel(r"$\chi_\mathrm{max}$")
plt.title(r"Croissance de $\chi$ — TEBD quimb")
plt.xlim(0, total_time)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()