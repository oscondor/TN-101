import numpy as np
import mps_library as mps_lib
from mps_class import MPS, MPS_Canonical


class MPO:
    """
    Matrix Product Operator with cores of shape (chi_l, d_up, d_down, chi_r).

    Physical convention:
      d_up  = "bra" index (output / row of the operator)
      d_down = "ket" index (input / col of the operator)
    """

    def __init__(self, cores_or_operator, local_dim=None):
        """
        Args:
            cores_or_operator:
                - list of 4-index arrays (chi_l, d_up, d_down, chi_r)
                - ndarray of shape (d1,d1,d2,d2,...,dN,dN) with alternating (up,down) indices
                - square matrix (d^N, d^N) — local_dim must then be provided
        """
        if isinstance(cores_or_operator, list):
            self.cores = cores_or_operator
        elif isinstance(cores_or_operator, np.ndarray):
            if cores_or_operator.dtype == object:
                self.cores = list(cores_or_operator)
            else:
                self.cores = MPO.from_operator_to_mpo(cores_or_operator, local_dim)
        else:
            raise ValueError("Input must be a list of cores or a numpy array.")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def from_operator_to_mpo(operator, local_dim=None):
        """
        Decompose a full operator into MPO form via sequential SVD.

        Args:
            operator: ndarray with shape (d1,d1,d2,d2,...,dN,dN) — indices alternate
                      (up_i, down_i) — OR square matrix (d^N, d^N) with local_dim given.
            local_dim: local Hilbert-space dimension (required only for matrix input).

        Returns:
            List of MPO cores of shape (chi_l, d_up, d_down, chi_r).
        """
        if operator.ndim == 2:
            if local_dim is None:
                raise ValueError("local_dim must be provided for matrix input.")
            d = local_dim
            N = round(np.log(operator.shape[0]) / np.log(d))
            # Reshape to (d_up_1,...,d_up_N, d_down_1,...,d_down_N) then interleave
            op = operator.reshape([d] * (2 * N))
            perm = [idx for i in range(N) for idx in (i, N + i)]
            operator = op.transpose(perm)

        # operator shape: (d1_up, d1_down, d2_up, d2_down, ..., dN_up, dN_down)
        N = operator.ndim // 2

        # Pad with trivial virtual bonds → (1, d1_up, d1_down, ..., dN_up, dN_down, 1)
        tensor = operator.reshape((1,) + operator.shape + (1,))
        tensor_shape = tensor.shape

        mpo = []
        for _ in range(N - 1):
            chi_l, d_up, d_down = tensor_shape[0], tensor_shape[1], tensor_shape[2]
            u, s, vt = mps_lib.svd_for_tensor(tensor, 3)
            mpo.append(u.reshape(chi_l, d_up, d_down, s.shape[0]))
            tensor = np.tensordot(np.diag(s), vt, axes=(1, 0))
            tensor = tensor.reshape([s.shape[0]] + list(tensor_shape[3:]))
            tensor_shape = tensor.shape

        # Last core: shape (chi_l, d_up, d_down, 1)
        mpo.append(tensor)
        return mpo

    # ------------------------------------------------------------------
    # Apply MPO to MPS  →  new MPS_Canonical
    # ------------------------------------------------------------------
    def mpo_to_mps(self):
        """
        Merging the two physical legs in a reshape to convert an MPO into an MPS.
        """
        new_cores = []
        for W in self.cores:
            chi_l, d_up, d_down, chi_r = W.shape
            new_core = W.reshape(chi_l, d_up * d_down, chi_r)
            new_cores.append(new_core)
        return MPS_Canonical(new_cores)
    
    @staticmethod
    def mps_to_mpo(mps_canonical):
        """
        Splitting the physical leg of an MPS core into two legs to convert an MPS into an MPO.
        """
        new_cores = []
        for A in mps_canonical.cores:
            chi_l, d, chi_r = A.shape
            d_up = d_down = int(np.sqrt(d))
            new_core = A.reshape(chi_l, d_up, d_down, chi_r)
            new_cores.append(new_core)
        return MPO(new_cores)
    
    def evaluate(self, vector):
        cores = self.cores
        ev_vector = vector
        if len(cores) != len(ev_vector) // 2:
            raise ValueError("...")
        res = cores[0][:, ev_vector[0], ev_vector[1], :]
        for i in range(1, len(cores)):
            mat = cores[i][:, ev_vector[2*i], ev_vector[2*i+1], :]
            res = np.tensordot(res, mat, axes=([-1], [0]))
        return res.squeeze()


    def apply_to_mps(self, mps_canonical, tol=None):
        """
        Compute O|ψ⟩.

        Args:
            mps_canonical: MPS_Canonical instance representing |ψ⟩.
            tol: optional truncation tolerance/bond-dim cap passed to sweep_left
                 for compression after application.

        Returns:
            MPS_Canonical of the resulting state.
            Bond dimensions are chi_W * chi_A before any compression.
        """
        if len(self.cores) != len(mps_canonical.cores):
            raise ValueError("MPO and MPS must have the same number of sites.")

        new_cores = []
        for W, A in zip(self.cores, mps_canonical.cores):
            # W: (chi_wl, d_up, d_down, chi_wr)
            # A: (chi_l,  d,    chi_r)
            # Contract over d_down (W axis 2) with d (A axis 1)
            C = np.tensordot(W, A, axes=([2], [1]))
            # C: (chi_wl, d_up, chi_wr, chi_l, chi_r)
            C = C.transpose(0, 3, 1, 2, 4)
            # C: (chi_wl, chi_l, d_up, chi_wr, chi_r)
            chi_wl, chi_l, d_up, chi_wr, chi_r = C.shape
            new_cores.append(C.reshape(chi_wl * chi_l, d_up, chi_wr * chi_r))

        # MPS_Canonical canonicalizes at rightmost site by default
        result = MPS_Canonical(new_cores)
        if tol is not None:
            result.sweep_left(tol)   # compress right → left
        return result

    # ------------------------------------------------------------------
    # Expectation value  <ψ|O|ψ>
    # ------------------------------------------------------------------

    def expectation_value(self, mps_canonical):
        """
        Compute ⟨ψ|O|ψ⟩ via sequential left-to-right environment contraction.

        Complexity: O(N · chi_A² · chi_W · d²)

        Args:
            mps_canonical: MPS_Canonical instance representing |ψ⟩.

        Returns:
            Scalar ⟨ψ|O|ψ⟩.
        """
        if len(self.cores) != len(mps_canonical.cores):
            raise ValueError("MPO and MPS must have the same number of sites.")

        # Left environment L[alpha, beta, gamma] with shape (chi_bra, chi_mpo, chi_ket)
        L = np.ones((1, 1, 1))

        for W, A in zip(self.cores, mps_canonical.cores):
            # W:      (chi_wl, d_up, d_down, chi_wr)
            # A:      (chi_l,  d,    chi_r)
            # A_conj: (chi_l,  d,    chi_r)
            A_conj = A.conj()

            # Step 1: contract L(alpha, beta, gamma) with A_conj(alpha, sigma, alpha')
            #         over alpha  →  (beta, gamma, sigma, alpha')
            tmp = np.tensordot(L, A_conj, axes=([0], [0]))

            # Step 2: contract (beta, gamma, sigma, alpha') with W(beta, sigma, tau, beta')
            #         over (beta, sigma) = (axis 0, axis 2)  →  (gamma, alpha', tau, beta')
            tmp = np.tensordot(tmp, W, axes=([0, 2], [0, 1]))

            # Step 3: contract (gamma, alpha', tau, beta') with A(gamma, tau, gamma')
            #         over (gamma, tau) = (axis 0, axis 2)  →  (alpha', beta', gamma')
            L = np.tensordot(tmp, A, axes=([0, 2], [0, 1]))

        return L.squeeze()
    
    def compress(self, tol):
        """Two-sweep SVD compression: QR right-sweep then truncated SVD left-sweep."""
        N = len(self.cores)

        # Right sweep: left-canonicalize via QR (no truncation)
        for i in range(N - 1):
            W = self.cores[i]
            chi_l, d_up, d_down, chi_r = W.shape
            Q, R = mps_lib.qr_for_tensor(W, 3)
            chi_new = Q.shape[1]
            self.cores[i] = Q.reshape(chi_l, d_up, d_down, chi_new)
            W_next = self.cores[i + 1]
            chi_l_n, d_up_n, d_down_n, chi_r_n = W_next.shape
            self.cores[i + 1] = np.tensordot(R, W_next.reshape(chi_l_n, d_up_n * d_down_n * chi_r_n),
                                              axes=(1, 0)).reshape(chi_new, d_up_n, d_down_n, chi_r_n)

        # Left sweep: right-canonicalize with truncated SVD
        for i in range(N - 1, 0, -1):
            W = self.cores[i]
            chi_l, d_up, d_down, chi_r = W.shape
            U, S, Vt = mps_lib.svd_for_tensor_compression(W, 1, tol)
            chi_new = S.shape[0]
            self.cores[i] = Vt.reshape(chi_new, d_up, d_down, chi_r)
            US = np.dot(U, np.diag(S))  # (chi_l, chi_new)
            W_prev = self.cores[i - 1]
            chi_l_p, d_up_p, d_down_p, chi_r_p = W_prev.shape
            self.cores[i - 1] = np.tensordot(
                W_prev.reshape(chi_l_p * d_up_p * d_down_p, chi_r_p), US, axes=(1, 0)
            ).reshape(chi_l_p, d_up_p, d_down_p, chi_new)

    def addition(self, other):
        """
        Add another MPO to this one, returning a new MPO representing the sum.
        
        Uses block-diagonal construction on internal virtual bonds while
        preserving open boundary conditions on the first and last cores.
        """
        if len(self.cores) != len(other.cores):
            raise ValueError("MPOs must have the same number of sites for addition.")

        N = len(self.cores)
        new_cores = []
        for i, (W1, W2) in enumerate(zip(self.cores, other.cores)):
            chi_wl1, d_up, d_down, chi_wr1 = W1.shape
            chi_wl2, _, _, chi_wr2 = W2.shape

            dtype = np.result_type(W1, W2)

            if N == 1:
                # Single-site MPO: keep boundary dimensions at 1.
                new_core = (W1 + W2).astype(dtype, copy=False)
            elif i == 0:
                # Left boundary: keep left bond dimension equal to 1.
                new_core = np.zeros((1, d_up, d_down, chi_wr1 + chi_wr2), dtype=dtype)
                new_core[0, :, :, :chi_wr1] = W1[0]
                new_core[0, :, :, chi_wr1:] = W2[0]
            elif i == N - 1:
                # Right boundary: keep right bond dimension equal to 1.
                new_core = np.zeros((chi_wl1 + chi_wl2, d_up, d_down, 1), dtype=dtype)
                new_core[:chi_wl1, :, :, 0] = W1[:, :, :, 0]
                new_core[chi_wl1:, :, :, 0] = W2[:, :, :, 0]
            else:
                # Bulk sites: block-diagonal on both virtual bonds.
                new_core = np.zeros((chi_wl1 + chi_wl2, d_up, d_down, chi_wr1 + chi_wr2), dtype=dtype)
                new_core[:chi_wl1, :, :, :chi_wr1] = W1
                new_core[chi_wl1:, :, :, chi_wr1:] = W2
            new_cores.append(new_core)

        return MPO(new_cores)
    
    def multiply(self, other,tol):
        """
        Multiply this MPO by another MPO, returning a new MPO representing the product.

        The resulting MPO has cores obtained by contracting the physical legs of the two MPOs,
        leading to a bond dimension that is the product of the original bond dimensions.
        """
        if len(self.cores) != len(other.cores):
            raise ValueError("MPOs must have the same number of sites for multiplication.")

        new_cores = []
        for W1, W2 in zip(self.cores, other.cores):
            chi_wl1, d_up1, d_down1, chi_wr1 = W1.shape
            chi_wl2, d_up2, d_down2, chi_wr2 = W2.shape

            if d_down1 != d_up2:
                raise ValueError("Physical dimensions do not match for multiplication.")

            # (self @ other): contract self's d_down (ket) with other's d_up (bra)
            new_core = np.tensordot(W1, W2, axes=([2], [1]))
            # shape: (chi_wl1, d_up1, chi_wr1, chi_wl2, d_down2, chi_wr2)

            new_core = new_core.transpose(0, 3, 1, 4, 2, 5)
            # shape: (chi_wl1, chi_wl2, d_up1, d_down2, chi_wr1, chi_wr2)

            new_core = new_core.reshape(chi_wl1 * chi_wl2, d_up1, d_down2, chi_wr1 * chi_wr2)
            u,s,v = mps_lib.svd_for_tensor_compression(new_core, 3, tol)  # Compression après multiplication
            tens = np.tensordot(u, np.diag(s) @ v, axes=(1, 0))
            new_core = tens.reshape(chi_wl1 * chi_wl2, d_up1, d_down2,-1)
            new_cores.append(new_core)
        return MPO(new_cores)
    
    def multiply_by_scalar(self, scalar):
        """Multiply the MPO by a scalar (scale only the first core)."""
        new_cores = [scalar * self.cores[0]] + [W.copy() for W in self.cores[1:]]
        return MPO(new_cores)

    def H_eff(self, mps_canonical):
        """
        Compute the effective Hamiltonian H_eff for the current center site of the MPS.

        H_eff is the operator that acts on the local tensor at the center site when
        applying the MPO to the MPS. It is obtained by contracting all MPO cores
        and MPS cores except for the center site, resulting in a 4-index tensor
        that can be reshaped into a matrix.

        Args:
            mps_canonical: MPS_Canonical instance representing |ψ⟩.

        Returns:
            H_eff: ndarray of shape (d * chi_left, d * chi_right) representing the effective Hamiltonian.
        """
        if len(self.cores) != len(mps_canonical.cores):
            raise ValueError("MPO and MPS must have the same number of sites.")

        N = len(self.cores)
        center = mps_canonical.get_center()

        # Left environment L[alpha, beta, gamma] with shape (chi_bra, chi_mpo, chi_ket)
        L = np.ones((1, 1, 1))
        for i in range(center):
            W = self.cores[i]
            A = mps_canonical.cores[i]
            A_conj = A.conj()
            tmp = np.tensordot(L, A_conj, axes=([0], [0]))
            tmp = np.tensordot(tmp, W, axes=([0, 2], [0, 1]))
            L = np.tensordot(tmp, A, axes=([0, 2], [0, 1]))

        # Right environment R[alpha', beta', gamma'] with shape (chi_bra', chi_mpo', chi_ket')
        R = np.ones((1, 1, 1))
        for i in range(N - 1, center, -1):
            W = self.cores[i]
            A = mps_canonical.cores[i]
            A_conj = A.conj()
            tmp = np.tensordot(R, A_conj, axes=([0], [2]))
            tmp = np.tensordot(tmp, W, axes=([0, 3], [3, 1])) # chi_mpo_r (ax0) ↔ chi_wr (ax3), d_bra (ax3) ↔ d_up (ax1)
            R = np.tensordot(tmp, A, axes=([3, 0], [1, 2]))
        
        # Now contract L, W_center, R to get H_eff
        W_center = self.cores[center]
        H_eff = np.tensordot(L, W_center, axes=([1], [0]))
        H_eff = np.tensordot(H_eff, R, axes=([4], [1])) #ici les legs sont dans le bon ordre (leg physique au milieu)
        H_eff = H_eff.transpose(0,2,4,1,3,5) # (bl, br, phys_out, phys_in) -> (bl, phys_out, br, phys_in)

        return H_eff
    


    
    