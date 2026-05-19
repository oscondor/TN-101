import numpy as np

def svd(A):
    """ Compute the Singular Value Decomposition (SVD) of a matrix A."""
    u,s,vt = np.linalg.svd(A, full_matrices=False)
    return u, s, vt

def svd_for_tensor(tensor, i):
    """ Compute SVD and contracts the ith first legs in the first inex and the rest in the second index."""
    m = tensor.reshape(int(np.prod(tensor.shape[:i])), -1)
    u, s, vt = svd(m)
    return u, s, vt

def qr_for_tensor(tensor, i):
    """ Compute QR decomposition and contracts the ith first legs in the first inex and the rest in the second index."""
    m = tensor.reshape(int(np.prod(tensor.shape[:i])), -1)
    q, r = np.linalg.qr(m)
    return q, r

def rq_for_tensor(tensor, i):
    """ Compute RQ decomposition and contracts the right ith first legs in the first inex and the rest in the second index."""
    L = len(tensor.shape)
    m = tensor.reshape(-1, int(np.prod(tensor.shape[L-i:])))
    q, r = np.linalg.qr(m.T)
    return q.T, r.T

def back_to_matrix(U, S, Vt, k):
    """ Reconstruct a matrix from its SVD components, using only the top k singular values."""
    if k is None:
        k = len(S)
    k = min(k, len(S), U.shape[1], Vt.shape[0])
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))

def controle_tronq_tol(U,S,Vt,eps):
    """ Return a truncated version of S where the truncation is determined by the tolerance eps."""
    S = np.asarray(S, dtype=float)
    norm_S = np.linalg.norm(S)
    i = 0
    while i < len(S) and np.linalg.norm(S[i:]) / norm_S > eps:
        i += 1
    if i == 0:
        return U, S, Vt
    return U[:,:i],S[:i],Vt[:i,:]

def controle_tronq_ind(u,s,vt, i):
    """Truncate S to the first i singular values, setting the rest to zero."""
    i=  max(0, min(int(i), len(s)))
    return u[:,:i],s[:i],vt[:i,:]

def svd_for_tensor_compression(tensor, i, tol):
    if tol is None:
        return svd_for_tensor(tensor, i)
    u, s, vt = svd_for_tensor(tensor, i)
    if isinstance(tol, (int, np.integer)):
        # tol interpreted as max bond dimension
        k = min(int(tol), len(s))
        u_trunc,s_trunc,vt_trunc = controle_tronq_ind(u,s,vt, k)   # zeros s[k:]
        return u_trunc, s_trunc, vt_trunc
    else:
        return controle_tronq_tol(u, s, vt, tol)

