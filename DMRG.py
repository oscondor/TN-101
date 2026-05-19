import mps_class 
import numpy as np
import mps_library as mps_lib 
import mpo_class
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def dmrg(mpo, mps_init, num_sweeps, tol):
    """
    Perform DMRG sweeps (1 sweep = right->left + left->right).
    The MPS is first put in canonical form with center on the rightmost site.
    """

    n_sites = len(mps_init.cores)

    # S'assurer qu'on commence bien avec le centre canonique tout à droite
    while mps_init.center < n_sites - 1:
        mps_init.shift_center_right()

    def optimize_current_site():
        H_eff = mpo.H_eff(mps_init)
        
        shape_eff = H_eff.shape
        n = shape_eff[0] * shape_eff[1] * shape_eff[2]
        H_eff_matrix = H_eff.reshape(n, shape_eff[3] * shape_eff[4] * shape_eff[5])
        current_core = mps_init.cores[mps_init.center]
        v0 = current_core.ravel()

        # Use dense solver for small matrices where ARPACK is unreliable (n < ncv_min)
        if n < 20:
            _, eigvecs = np.linalg.eigh(H_eff_matrix)
            ground_state_vec = eigvecs[:, 0]
        else:
            H_eff_sparse = csr_matrix(H_eff_matrix)
            ncv = min(max(2 + 1, 20), n)
            _, eigvecs = eigsh(H_eff_sparse, k=1, which='SA', ncv=ncv, v0=v0)
            ground_state_vec = eigvecs[:, 0]

        mps_init.cores[mps_init.center] = ground_state_vec.reshape(current_core.shape)

    for _ in range(num_sweeps):
        # Aller : droite -> gauche
        for _ in range(n_sites - 1):
            optimize_current_site()
            mps_init.shift_center_left(tol)
        optimize_current_site()  # site gauche

        # Retour : gauche -> droite
        for _ in range(n_sites - 1):
            optimize_current_site()
            mps_init.shift_center_right(tol)
        optimize_current_site()  # site droit

    return mps_init

