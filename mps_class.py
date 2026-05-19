import numpy as np
import mps_library as mps_lib

class MPS :
    def __init__(self, mps_or_tensor):
        """
        Initialize a canonical MPS with center at the rightmost position.

        Args:
            mps_or_tensor: Either a list of MPS cores or a full tensor.
                          If tensor, it's converted to MPS first.
        """
        if isinstance(mps_or_tensor, (list, np.ndarray)) and not isinstance(mps_or_tensor, np.ndarray):
            # It's already an MPS (list of cores)
            self.cores = list(mps_or_tensor)
        elif isinstance(mps_or_tensor, np.ndarray):
            # Check if it looks like MPS (list of arrays) or a full tensor
            if mps_or_tensor.dtype == object:
                # It's an object array, treat as MPS cores
                self.cores = [mps_or_tensor[i] for i in range(len(mps_or_tensor))]
            else:
                # It's a full tensor, convert to MPS
                self.cores = MPS.from_tensor_to_mps(mps_or_tensor)
        else:
            # Assume it's a list of cores
            self.cores = list(mps_or_tensor)


    def from_tensor_to_mps(tensor):
        mps = []
        tensor_shape = tensor.shape
        tensor = tensor.reshape((1,) + tensor_shape + (1,))
        tensor_shape = tensor.shape
        L= len(tensor_shape)
        for i in range(L-3) :
            u, s, vt = mps_lib.svd_for_tensor(tensor, 2)
            mps.append(u.reshape(list(tensor_shape[:2])+[s.shape[0]]))
            tensor = np.tensordot(np.diag(s), vt, axes=(1,0))
            tensor = tensor.reshape([s.shape[0]] + list(tensor_shape[2:]))
            tensor_shape = tensor.shape
        mps.append(tensor)
        return mps
    
    def tensor_to_canonical_mps(tensor, i):
        N = len(tensor.shape)
        mps = np.empty(N, dtype=object)
        tensor = tensor.reshape((1,) + tensor.shape + (1,))

        # QR decomposition from left to i (left-canonical tensors)
        for j in range(i):
            q, r = mps_lib.qr_for_tensor(tensor, 2)
            mps[j] = q.reshape(list(tensor.shape[:2]) + [-1])
            tensor = r.reshape([-1] + list(tensor.shape[2:]))

        # RQ decomposition from right to i (right-canonical tensors)
        for j in range(N - 1, i, -1):
            q, r = mps_lib.rq_for_tensor(tensor, 2)
            mps[j] = q.reshape([-1] + list(tensor.shape[-2:]))
            tensor = r.reshape(list(tensor.shape[:-2]) + [-1])

        # Store the remaining tensor at the orthogonality center
        mps[i] = tensor

        return mps
    
    def canonicalize_mps(mps, i):
        N = len(mps)
        # QR decomposition from left to i (left-canonical tensors)
        for j in range(i):
            q, r = mps_lib.qr_for_tensor(mps[j], 2)
            mps[j] = q.reshape(list(mps[j].shape[:2]) + [-1])
            mps[j + 1] = np.tensordot(r, mps[j + 1], axes=(1, 0))

        # RQ decomposition from right to i (right-canonical tensors)
        for j in range(N - 1, i, -1):
            q, r = mps_lib.rq_for_tensor(mps[j], 2)
            mps[j] = q.reshape([-1] + list(mps[j].shape[-2:]))
            mps[j - 1] = np.tensordot(mps[j - 1], r, axes=([-1], [0]))

        return mps
    
    def from_mps_to_tensor( mps) :
        tensor = mps[0]
        for i in range(1, len(mps)) :
            tensor = np.tensordot(tensor, mps[i], axes=([-1], [0]))
        tensor_shape = tensor.shape
        tensor = tensor.reshape(tensor_shape[1:-1])
        return tensor
    
    def mps_evaluation(mps,ev_vector) :
        if len(mps) != len(ev_vector) :
            raise ValueError("The length of the evaluation vector must match the number of MPS tensors.")
        res = mps[0][:,ev_vector[0],:]
        for i in range(1,len(mps)) :
            res = np.tensordot(res,mps[i][:,ev_vector[i],:],axes=(1,0))
        return res.squeeze()
    

    def mps_multiplication(mps1, mps2) :
        if len(mps1) != len(mps2) :
            raise ValueError("The two MPS must have the same number of tensors.")
        res = np.tensordot(mps1[0], mps2[0], axes=([1], [1]))
        for i in range(1, len(mps1)) :
            inter_i = np.tensordot(mps1[i], mps2[i], axes=([1], [1]))
            res = np.tensordot(res, inter_i, axes=([1, 3], [0, 2]))
            res = res.transpose(0, 2, 1, 3)
        return res.squeeze()
    
    def mps_norm(mps): # just for real tensors for now
        return np.sqrt(MPS.mps_multiplication(mps, mps))
    
    def mps_can_norm(mps, i) :
        return np.linalg.norm(mps[i])


class MPS_Canonical:
    """
    Represents an MPS with a well-defined orthogonality center.

    - Tensors to the left of center are left-canonical (orthonormal on right bonds)
    - Tensors to the right of center are right-canonical (orthonormal on left bonds)
    - The center tensor absorbs all the non-orthogonal weight
    - The center position can be moved efficiently with shift_center_right/left
    """

    def __init__(self, mps_or_tensor):
        """
        Initialize a canonical MPS with center at the rightmost position.

        Args:
            mps_or_tensor: Either a list of MPS cores or a full tensor.
                          If tensor, it's converted to MPS first.
        """
        if isinstance(mps_or_tensor, (list, np.ndarray)) and not isinstance(mps_or_tensor, np.ndarray):
            # It's already an MPS (list of cores)
            self.cores = list(mps_or_tensor)
        elif isinstance(mps_or_tensor, np.ndarray):
            # Check if it looks like MPS (list of arrays) or a full tensor
            if mps_or_tensor.dtype == object:
                # It's an object array, treat as MPS cores
                self.cores = [mps_or_tensor[i] for i in range(len(mps_or_tensor))]
            else:
                # It's a full tensor, convert to MPS
                self.cores = MPS.from_tensor_to_mps(mps_or_tensor)
        else:
            # Assume it's a list of cores
            self.cores = list(mps_or_tensor)

        # Canonicalize with center at rightmost position
        N = len(self.cores)
        self.cores = MPS.canonicalize_mps(self.cores, N - 1)
        self.center = N - 1

    def shift_center_right(self, tol):
        """
        Shift orthogonality center one position to the right (i → i+1).
        Uses SVD to decompose the current center and absorb factors into next tensor.
        """
        if tol is None :
            if self.center >= len(self.cores) - 1:
                raise ValueError("Cannot shift center right beyond the rightmost position")

            # QR on current center
            q,r = mps_lib.qr_for_tensor(self.cores[self.center], 2)

            # Left-canonical part at current position
            self.cores[self.center] = q.reshape(list(self.cores[self.center].shape[:2]) + [-1])

            # Absorb singular values and right singular vectors into next tensor
            self.cores[self.center + 1] = np.tensordot(r, self.cores[self.center + 1], axes=(1, 0))

            self.center += 1
        else : 
            if self.center >= len(self.cores) - 1:
                raise ValueError("Cannot shift center right beyond the rightmost position")

            # QR on current center
            u,s,vt = mps_lib.svd_for_tensor_compression(self.cores[self.center], 2, tol)

            # Left-canonical part at current position
            self.cores[self.center] = u.reshape(list(self.cores[self.center].shape[:2]) + [-1])
            r = np.tensordot(np.diag(s), vt, axes=(1, 0))
            # Absorb singular values and right singular vectors into next tensor
            self.cores[self.center + 1] = np.tensordot(r, self.cores[self.center + 1], axes=(1, 0))

            self.center += 1

    def shift_center_left(self,tol):
        """
        Shift orthogonality center one position to the left (i → i-1).
        Uses SVD to decompose the current center and absorb factors into previous tensor.
        """
        if tol is None :
            if self.center <= 0:
                raise ValueError("Cannot shift center left beyond the leftmost position")

            # RQ
            q,r = mps_lib.rq_for_tensor(self.cores[self.center], 2)

            # Right-canonical part at current position
            self.cores[self.center] = q.reshape([-1] + list(self.cores[self.center].shape[-2:]))

            # Absorb into previous tensor
            self.cores[self.center - 1] = np.tensordot( self.cores[self.center - 1],r, axes=(2, 0))

            self.center -= 1

        else : 
            if self.center <= 0:
                raise ValueError("Cannot shift center left beyond the leftmost position")

            u,s,vt = mps_lib.svd_for_tensor_compression(self.cores[self.center], 1, tol)

            # Left-canonical part at current position
            self.cores[self.center] = vt.reshape(list([-1] + list(self.cores[self.center].shape[1:])) )
            r = np.tensordot( u,np.diag(s), axes=(1, 0))

            # Absorb singular values and right singular vectors into next tensor
            self.cores[self.center - 1] = np.tensordot( self.cores[self.center - 1],r, axes=(2, 0))

            self.center -= 1

    def shift_center(self, target,tol):
        """
        Shift orthogonality center to a target position.

        Args:
            target: Target center position (0 to N-1)
        """
        if target < 0 or target >= len(self.cores):
            raise ValueError(f"Target center position {target} out of range [0, {len(self.cores)-1}]")

        # Shift right if needed
        while self.center < target:
            self.shift_center_right(tol)

        # Shift left if needed
        while self.center > target:
            self.shift_center_left(tol)

    def overlap(self, other):
        # L shape: (β_bra, β_ket)
        L = np.array([[1.0]])
        
        for A, B in zip(self.cores, other.cores):
            # A : (α_bra, σ, β_bra), B : (α_ket, σ, β_ket)
            # Contraction L(β_bra, β_ket) avec A(α_bra, σ, β_bra) sur axe β_bra
            L = np.tensordot(L, A, axes=([0], [0]))   # (β_ket, σ, β_bra)
            # Contraction avec B(α_ket, σ, β_ket) sur β_ket (axe 0 de L) et σ (axe 1)
            L = np.tensordot(L, B, axes=([0, 1], [0, 1]))  # (β_bra, β_ket)
        
        return L.squeeze()
    
    def to_tensor(self):
        """Reconstruct the full tensor from the MPS cores."""
        return MPS.from_mps_to_tensor(self.cores)

    def get_cores(self):
        """Return the list of MPS cores."""
        return self.cores

    def get_center(self):
        """Return the current center position."""
        return self.center

    def get_core(self, i):
        """Return the i-th MPS core."""
        return self.cores[i]

    def norm(self):
        """Compute the Frobenius norm of the MPS (i.e., norm of reconstructed tensor)."""
        # Norm is norm of tensor at center times product of orthonormal factors
        # Since left/right parts are orthonormal, the result is just sqrt of ||center||^2
        return np.linalg.norm(self.cores[self.center])

    def sweep_right(self, tol):
        """Perform a right sweep through the MPS, shifting the center from left to right."""
        self.shift_center(len(self.cores) - 1, tol)
    
    def sweep_left(self, tol):
        """Perform a left sweep through the MPS, shifting the center from right to left."""
        self.shift_center(0, tol)   
    




