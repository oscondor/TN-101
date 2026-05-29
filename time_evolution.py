import numpy as np
import mps_library as mps_lib
import mps_class
import mpo_class
import DMRG
from fermions_1d import cdag_fermions
from fermions_1d import random_mps
from fermions_1d import TB_hamiltonian_interaction



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

 
        



