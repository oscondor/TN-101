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

 
        



n_sites = 10

# Hamiltonien tight-binding complet : saut i <-> i+1 pour tout i
h = np.zeros((n_sites, n_sites))
for i in range(n_sites - 1):
    h[i, i + 1] = 1.0
    h[i + 1, i] = 1.0

mpo = TB_hamiltonian_interaction(h, n_sites, tol=20)
print("MPO constructed.")
# Calculer la fermi sea avec DMRG
mps_init = random_mps(n_sites, d=2, chi_max=16)
mps_init = DMRG.dmrg(mpo, mps_init, num_sweeps=10, tol=1e-8)

# Ajouter c0dag à la fermi sea
c0dag_mpo = cdag_fermions(0, n_sites)
mps_init = c0dag_mpo.apply_to_mps(mps_init, tol=1e-8)
# c†₀ n'est pas unitaire : ||c†₀|ψ⟩||² = 1 - ⟨n₀⟩ < 1, il faut renormaliser
mps_init.cores[mps_init.center] /= mps_init.norm()

E0 = mpo.expectation_value(mps_init)
print(f"Energy with hamiltonian: {E0:.10f}")

from plot_analytic import compute_nj
dt = 0.01
n_steps = 3000


# Hamiltonien local à 2 sites pour la porte TEBD
h_local = np.zeros((4, 4), dtype=complex)
h_local[1, 2] = 1.0
h_local[2, 1] = 1.0
gate = make_gate(h_local, dt, imaginary_time=False)

gate_info_even = [(gate, i) for i in range(n_sites - 1) if i % 2 == 0]
gate_info_odd = [(gate, i) for i in range(n_sites - 1) if i % 2 == 1]
gates = [gate_info_even, gate_info_odd]

n0 = TB_hamiltonian_interaction(np.diag([1.0] + [0.0] * (n_sites - 1)), n_sites, tol=20)
print(f"Occupation initiale du site 0 : {n0.expectation_value(mps_init):.10f}")
# MPOs des opérateurs n_r
n_r_mpos = []
for r in range(n_sites):
    hr = np.zeros((n_sites, n_sites))
    hr[r, r] = 1.0
    n_r_mpos.append(TB_hamiltonian_interaction(hr, n_sites, tol=4))

# Stockage de n_r(t)
nrt = np.zeros((n_steps + 1, n_sites), dtype=float)

# Stockage de tous les chi (dimensions de liaison) pour chaque bond
bond_hist = np.zeros((n_steps + 1, n_sites - 1), dtype=int)

# Stockage du nombre total de particules en fonction du temps
total_n = np.zeros(n_steps + 1, dtype=float)

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

bond_dims = np.zeros(n_steps + 1, dtype=int)
for r in range(n_sites):
    nrt[0, r] = np.real(n_r_mpos[r].expectation_value(mps_init))

# calcul analytique via compute_nj(t) importé de plot_analytic
analytic_nrt = np.zeros_like(nrt)
for t in range(n_steps + 1):
    analytic_nrt[t] = np.real(compute_nj(n_sites, t * dt))

bond_dims[0] = max_bond_dimension(mps_init)
bond_hist[0, :] = bond_dimensions(mps_init)
total_n[0] = nrt[0].sum()

for t in range(1, n_steps + 1):
    print_run_progress(t, n_steps)
    
    time_evolution(gates, mps_init, dt=dt, N=1, tol=1e-8)
    for r in range(n_sites):
        nrt[t, r] = np.real(n_r_mpos[r].expectation_value(mps_init))
    total_n[t] = nrt[t].sum()
    bond_dims[t] = max_bond_dimension(mps_init)
    bond_hist[t, :] = bond_dimensions(mps_init)

print_run_progress(n_steps, n_steps)
print()


""""
# Animation : n_r en fonction de r et des chi en fonction du bond
fig, (ax, bond_ax, total_ax) = plt.subplots(
    3, 1, figsize=(8, 9), sharex=False,
    gridspec_kw={'height_ratios': [2, 1, 1]}
)
x = np.arange(n_sites)
line, = ax.plot(x, nrt[0], lw=2)
analytic_line, = ax.plot(x, analytic_nrt[0], lw=2, ls='--', color='C1')
ax.set_xlim(0, n_sites - 1)
ymin = float(np.min(nrt))
ymax = float(np.max(nrt))
pad = 0.05 * max(1e-12, ymax - ymin)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_xlabel('r')
ax.set_ylabel(r'$\langle n_r(t) \rangle$')
title = ax.set_title('t = 0.0')

# second plot: tous les chi sur les bonds du MPS
bond_x = np.arange(n_sites - 1)
bond_line, = bond_ax.plot(bond_x, bond_hist[0], lw=2, color='C2', marker='o')
bond_ax.set_xlim(0, max(0, n_sites - 2))
bond_ax.set_ylim(0, max(1, int(bond_hist.max()) + 1))
bond_ax.set_xlabel('Bond')
bond_ax.set_ylabel(r'$\chi$')
bond_ax.grid(True, alpha=0.3)

# texte affichant le temps réel écoulé depuis le début de l'animation
time_text = ax.text(0.02, 0.95, 'Temps réel: 0.00s', transform=ax.transAxes,
                    ha='left', va='top')

# variable pour suivre le temps réel écoulé
elapsed = 0.0

def update(frame):
    line.set_ydata(nrt[frame])
    analytic_line.set_ydata(analytic_nrt[frame])
    bond_line.set_ydata(bond_hist[frame])
    title.set_text(f't = {frame * dt:.3f}')
    time_marker.set_xdata([frame * dt, frame * dt])
    return line, analytic_line, bond_line, title, time_marker

# third plot: nombre total de particules en fonction du temps
time_array = np.arange(n_steps + 1) * dt
total_ax.plot(time_array, total_n, lw=2, color='C3')  # courbe fixe
time_marker = total_ax.axvline(0.0, color='red', lw=1.5, linestyle='--')
total_ax.set_xlim(0, time_array.max())
pad_tot = 0.05 * max(1e-12, total_n.max() - total_n.min())
total_ax.set_ylim(total_n.min() - pad_tot, total_n.max() + pad_tot)
total_ax.set_xlabel('Temps')
total_ax.set_ylabel('N_tot')

ani = FuncAnimation(fig, update, frames=n_steps + 1, interval=1, blit=True)
plt.tight_layout()
plt.show()
"""