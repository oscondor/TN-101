import numpy as np
import mps_library as mps_lib
import mps_class
import mpo_class
import DMRG
from fermions_1d import random_mps
from fermions_1d import TB_hamiltonian_interaction

#gate est une matrice 4x4 représentant une porte quantique à 2 sites, on veut l'appliquer à un MPS en contractant les bons indices

def apply_gate(gate, i, mps, tol):
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



n_sites = 5
mps_init = random_mps(n_sites, d=2, chi_max=16)
print('cores shapes :', [core.shape for core in mps_init.cores])
mps_init.shift_center(1,None)
h = np.zeros((n_sites, n_sites))
h[1, 2] = h[2, 1] = 1
mpo = TB_hamiltonian_interaction(h, n_sites, tol=20)
E0 = mpo.expectation_value(mps_init)
print(f"Energy with hamiltonian: {E0:.10f}")

gate = np.zeros((4, 4))
gate[1,2] = gate[2,1] = 1 
mps_copy = mps_class.MPS(mps_init.cores.copy()) 
apply_gate(gate, 1, mps_init, tol=None)
mps_class.MPS(mps_init.cores)
#print('shapes :', len(mps_init.cores),len(mps_copy.cores))
#print('cores shapes :', [core.shape for core in mps_init.cores], [core.shape for core in mps_copy.cores])
E1 = mps_init.overlap(mps_copy)
print(f"Energy with gate: {E1:.10f}")