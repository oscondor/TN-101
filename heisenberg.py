import numpy as np
import mps_class
import mpo_class
import DMRG


def heisenberg_mpo(n_sites, Jz,Jxy):
    """
    Construit le MPO de Heisenberg XXX avec dimension de lien 5.

    Automate fini sur les états auxiliaires :
      0 : identité propagée de gauche à droite
      1 : Sz en attente (côté gauche)
      2 : S+ en attente
      3 : S- en attente
      4 : identité propagée de droite à gauche (résultat final)

    Cœurs de forme (chi_l, d_up, d_down, chi_r).
    """
    I  = np.eye(2, dtype=float)
    Sz = 0.5 * np.array([[1., 0.], [0., -1.]])
    Sp = np.array([[0., 1.], [0.,  0.]])
    Sm = np.array([[0., 0.], [1.,  0.]])

    # Cœur générique (5, 2, 2, 5)
    W = np.zeros((5, 2, 2, 5))
    W[0, :, :, 0] = I           # passe-identité gauche→droite
    W[0, :, :, 1] = Sz          # démarre terme Sz
    W[0, :, :, 2] = Sp          # démarre terme S+
    W[0, :, :, 3] = Sm          # démarre terme S-
    W[1, :, :, 4] = Jz * Sz      # complète Sz·Sz
    W[2, :, :, 4] = Jxy / 2 * Sm  # complète S+·S-
    W[3, :, :, 4] = Jxy / 2 * Sp  # complète S-·S+
    W[4, :, :, 4] = I           # passe-identité droite→gauche

    # Bord gauche : fixe alpha=0 → shape (1, 2, 2, 5)
    W0 = W[0:1, :, :, :]
    # Bord droit : fixe beta=4 → shape (5, 2, 2, 1)
    WN = W[:, :, :, 4:5]

    cores = [W0] + [W.copy() for _ in range(n_sites - 2)] + [WN]
    return mpo_class.MPO(cores)


def random_mps(n_sites, d=2, chi_max=32):
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


if __name__ == "__main__":
    n_sites = 400
    Jz = 1.0
    Jxy = 2
    n_sweeps = 10
    chi_max = 32

    print(f"Modèle de Heisenberg XXX — {n_sites} sites, Jz = {Jz}, Jxy = {Jxy}")
    print("=" * 50)

    print("Construction du MPO...")
    mpo = heisenberg_mpo(n_sites, Jz, Jxy)

    np.random.seed(42)
    mps_init = random_mps(n_sites, d=2, chi_max=chi_max)

    print("\nDMRG (sweep par sweep) :")
    E0 = mpo.expectation_value(mps_init)
    print(f"  Sweep  0 : E = {E0:.10f}")

    for sweep in range(1, n_sweeps + 1):
        mps_init = DMRG.dmrg(mpo, mps_init, 1, 1e-2)
        E = mpo.expectation_value(mps_init)
        print(f"  Sweep {sweep:2d} : E = {E:.10f}")

    # Valeur de référence analytique (Bethe ansatz, thermodynamique) : E/N ≈ -0.4431...
    E_bethe = -0.4431471805599453 * n_sites
    print("\n" + "=" * 50)
    print(f"Énergie DMRG finale  : {E:.10f}")
    print(f"Référence Bethe (TL) : {E_bethe:.10f}")
    print(f"Écart relatif        : {abs(E - E_bethe) / abs(E_bethe):.2e}")
