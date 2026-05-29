import numpy as np
import mps_class
import mpo_class
import DMRG

def cdag_fermions(i, n_sites):
    I     = np.eye(2, dtype=float)
    c_dag = np.array([[0., 0.], [1., 0.]])
    sign  = np.array([[1., 0.], [0., -1.]])

    cores = []
    for k in range(n_sites):
        if k < i:
            op = sign
        elif k == i:
            op = c_dag
        else:
            op = I
        # MPO bond dim 1 partout : shape (1, 2, 2, 1)
        W = op.reshape(1, 2, 2, 1)
        cores.append(W)
    return mpo_class.MPO(cores)

def c_fermions(i, n_sites):
    I     = np.eye(2, dtype=float)
    c     = np.array([[0., 1.], [0., 0.]])
    sign  = np.array([[1., 0.], [0., -1.]])

    cores = []
    for k in range(n_sites):
        if k < i:
            op = sign
        elif k == i:
            op = c
        else:
            op = I
        # MPO bond dim 1 partout : shape (1, 2, 2, 1)
        W = op.reshape(1, 2, 2, 1)
        cores.append(W)
    return mpo_class.MPO(cores)

def n_fermions(i, n_sites):
    I = np.eye(2, dtype=float)
    n = np.array([[0., 0.], [0., 1.]])
    cores = []
    for k in range(n_sites):
        op = n if k == i else I
        cores.append(op.reshape(1, 2, 2, 1))
    return mpo_class.MPO(cores)

def TB_hamiltonian_interaction(h, n_sites,tol):
    H = None
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            if h[i, j] == 0:
                continue          # ← skip les termes nuls !
            term1 = cdag_fermions(i, n_sites).multiply(c_fermions(j, n_sites),tol)
            term2 = cdag_fermions(j, n_sites).multiply(c_fermions(i, n_sites),tol)
            hopping = term1.addition(term2).multiply_by_scalar(h[i, j])
            #print('bound dim:' )
            if H is None:
                H = hopping
            else:
                H = H.addition(hopping)
        if h[i, i] != 0:
            onsite = n_fermions(i, n_sites).multiply_by_scalar(h[i, i])
            if H is None:
                H = onsite
            else:
                H = H.addition(onsite)
    return H


def random_mps(n_sites,chi_max, d=2):
    """MPS left-canonical aléatoire : isométries QR pour les n-1 premiers sites, dernier core normalisé."""
    def bond_dims(i):
        chi_l = max(min(chi_max, d ** i,       d ** (n_sites - i)),     1) #pour assurer que les dimensions de lien sont au moins 1 et ne dépassent pas d^i ou d^(n-i) ou chi_max
        chi_r = max(min(chi_max, d ** (i + 1), d ** (n_sites - i - 1)), 1)
        return chi_l, chi_r

    cores = []
    for i in range(n_sites - 1):
        chi_l, chi_r = bond_dims(i)
        # QR d'une matrice aléatoire (chi_l*d, chi_r) → Q est une isométrie gauche
        Q, _ = np.linalg.qr(np.random.randn(chi_l * d, chi_r))
        cores.append(Q.reshape(chi_l, d, chi_r))

    chi_l, chi_r = bond_dims(n_sites - 1)
    last = np.random.randn(chi_l, d, chi_r)
    last /= np.linalg.norm(last)
    cores.append(last)

    return mps_class.MPS_Canonical(cores)

def correlator(mps, i, j):
    if i<j :
        mps.shift_center(i, None)
        c_i_dag = cdag_fermions(i, mps.n_sites)
        c_j     = c_fermions(j, mps.n_sites)
        op = c_i_dag.multiply(c_j)
        return op.expectation_value(mps)
    else:
        return np.conjugate(correlator(mps, j, i))


def correlation_matrix(mps):
    n_sites = mps.n_sites
    C = np.zeros((n_sites, n_sites), dtype=complex)
    for i in range(n_sites):
        for j in range(i, n_sites):
            C[i, j] = correlator(mps, i, j)
            if i != j:
                C[j, i] = np.conjugate(C[i, j])
    return C

if __name__ == "__main__":

    n_sites = 10
    h = np.random.rand(n_sites, n_sites)  # Matrice de hopping aléatoire pour 10 sites
    hsym = (h + h.T) / 2  # Rendre la matrice hermitienne
    print(hsym.shape)
    hsym = np.zeros((n_sites, n_sites))  # Matrice de hopping nulle pour tester le cas trivial
    for i in range(n_sites - 1):
        hsym[i, i + 1] = hsym[i + 1, i] = -1.0  # Hopping entre sites voisins
    vp = np.linalg.eigvalsh(hsym)
    somme_vp_negatives = np.sum(vp[vp < 0])
    print(f"Somme des valeurs propres négatives de h : {somme_vp_negatives:.10f}")
    n_sweeps = 10
    chi_max = 32
    print(f"TB modèle XXX — {n_sites} sites")
    print("=" * 50)

    print("Construction du MPO...")
    mpo = TB_hamiltonian_interaction(hsym,n_sites,tol=20)
   
    np.random.seed(42)
    mps_init = random_mps(n_sites, chi_max=chi_max, d=2)   
    print("\nDMRG (sweep par sweep) :")
    E0 = mpo.expectation_value(mps_init)
    print(f"  Sweep  0 : E = {E0:.10f}")

    for sweep in range(1, n_sweeps + 1):
        mps_init = DMRG.dmrg(mpo, mps_init, 1, 1e-3)
        E = mpo.expectation_value(mps_init)
        print(f"  Sweep {sweep:2d} : E = {E:.10f}")

