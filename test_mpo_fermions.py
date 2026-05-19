import numpy as np
import pytest
import mpo_class
import fermions_1d as f1d

# ─── helpers ────────────────────────────────────────────────────────────────

def mpo_to_dense(mpo):
    """Contract all MPO cores into a full (2^N, 2^N) matrix."""
    d = mpo.cores[0].shape[1]
    W0 = mpo.cores[0]           # (1, d, d, chi_r)
    result = W0[0]              # (d, d, chi_r)
    for i in range(1, len(mpo.cores)):
        W = mpo.cores[i]        # (chi_l, d, d, chi_r)
        # contract chi_l, then interleave physical indices
        new = np.tensordot(result, W, axes=([2], [0]))
        # new: (D_up, D_down, d_up, d_down, chi_r)
        D_up, D_down = result.shape[0], result.shape[1]
        chi_r = W.shape[3]
        new = new.transpose(0, 2, 1, 3, 4)   # (D_up, d_up, D_down, d_down, chi_r)
        result = new.reshape(D_up * d, D_down * d, chi_r)
    return result[:, :, 0]


def kron_op(ops):
    """Kronecker product of a list of 2x2 matrices."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


I     = np.eye(2)
c_dag = np.array([[0., 0.], [1., 0.]])
c_ann = np.array([[0., 1.], [0., 0.]])
sign  = np.array([[1., 0.], [0., -1.]])
n_op  = c_dag @ c_ann          # number operator [[0,0],[0,1]]


def exact_cdag(i, n_sites):
    ops = [sign] * i + [c_dag] + [I] * (n_sites - i - 1)
    return kron_op(ops)


def exact_c(i, n_sites):
    ops = [sign] * i + [c_ann] + [I] * (n_sites - i - 1)
    return kron_op(ops)


# ─── tests cdag / c ─────────────────────────────────────────────────────────

class TestCdagC:

    @pytest.mark.parametrize("n_sites", [1, 2, 3, 4])
    @pytest.mark.parametrize("i", [0, 1, 2, 3])
    def test_cdag_dense(self, i, n_sites):
        if i >= n_sites:
            pytest.skip("site index out of range")
        mpo = f1d.cdag_fermions(i, n_sites)
        np.testing.assert_allclose(
            mpo_to_dense(mpo), exact_cdag(i, n_sites), atol=1e-12,
            err_msg=f"cdag_fermions({i},{n_sites}) differs from exact"
        )

    @pytest.mark.parametrize("n_sites", [1, 2, 3, 4])
    @pytest.mark.parametrize("i", [0, 1, 2, 3])
    def test_c_dense(self, i, n_sites):
        if i >= n_sites:
            pytest.skip("site index out of range")
        mpo = f1d.c_fermions(i, n_sites)
        np.testing.assert_allclose(
            mpo_to_dense(mpo), exact_c(i, n_sites), atol=1e-12,
            err_msg=f"c_fermions({i},{n_sites}) differs from exact"
        )

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    @pytest.mark.parametrize("i", [0, 1, 2, 3])
    def test_anticommutation_same_site(self, i, n_sites):
        """{ c_i, cdag_i } = I"""
        if i >= n_sites:
            pytest.skip()
        Ci  = mpo_to_dense(f1d.c_fermions(i, n_sites))
        CDi = mpo_to_dense(f1d.cdag_fermions(i, n_sites))
        anticomm = Ci @ CDi + CDi @ Ci
        np.testing.assert_allclose(
            anticomm, np.eye(2**n_sites), atol=1e-12,
            err_msg=f"anticommutator {{c_{i}, cdag_{i}}} != I for n={n_sites}"
        )

    @pytest.mark.parametrize("n_sites", [3, 4])
    def test_anticommutation_different_sites(self, n_sites):
        """{ c_i, cdag_j } = 0 for i != j"""
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    continue
                Ci  = mpo_to_dense(f1d.c_fermions(i, n_sites))
                CDj = mpo_to_dense(f1d.cdag_fermions(j, n_sites))
                anticomm = Ci @ CDj + CDj @ Ci
                np.testing.assert_allclose(
                    anticomm, 0, atol=1e-12,
                    err_msg=f"anticommutator {{c_{i}, cdag_{j}}} != 0 for n={n_sites}"
                )

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    @pytest.mark.parametrize("i", [0, 1, 2, 3])
    def test_number_operator(self, i, n_sites):
        """cdag_i @ c_i = number operator at site i"""
        if i >= n_sites:
            pytest.skip()
        n_mpo = f1d.cdag_fermions(i, n_sites).multiply(f1d.c_fermions(i, n_sites))
        ops = [I] * i + [n_op] + [I] * (n_sites - i - 1)
        np.testing.assert_allclose(
            mpo_to_dense(n_mpo), kron_op(ops), atol=1e-12,
            err_msg=f"number operator at site {i} wrong for n={n_sites}"
        )


# ─── tests MPO.multiply ──────────────────────────────────────────────────────

class TestMPOMultiply:

    def _random_bond1_mpo(self, n_sites, rng):
        """MPO with bond dim 1: each core is a random 2x2 matrix."""
        cores = [rng.standard_normal((1, 2, 2, 1)) for _ in range(n_sites)]
        return mpo_class.MPO(cores)

    @pytest.mark.parametrize("n_sites", [1, 2, 3, 4])
    def test_multiply_matches_matrix_product(self, n_sites):
        rng = np.random.default_rng(0)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        AB_dense = mpo_to_dense(A.multiply(B))
        expected = mpo_to_dense(A) @ mpo_to_dense(B)
        np.testing.assert_allclose(AB_dense, expected, atol=1e-12)

    @pytest.mark.parametrize("n_sites", [2, 3])
    def test_multiply_associative(self, n_sites):
        rng = np.random.default_rng(1)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        C = self._random_bond1_mpo(n_sites, rng)
        AB_C = (A.multiply(B)).multiply(C)
        A_BC = A.multiply(B.multiply(C))
        np.testing.assert_allclose(
            mpo_to_dense(AB_C), mpo_to_dense(A_BC), atol=1e-12
        )

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_multiply_identity(self, n_sites):
        """A * I = A, where I is the identity MPO."""
        rng = np.random.default_rng(2)
        A = self._random_bond1_mpo(n_sites, rng)
        id_cores = [np.eye(2).reshape(1, 2, 2, 1) for _ in range(n_sites)]
        I_mpo = mpo_class.MPO(id_cores)
        np.testing.assert_allclose(
            mpo_to_dense(A.multiply(I_mpo)), mpo_to_dense(A), atol=1e-12
        )

    @pytest.mark.parametrize("n_sites", [2, 3])
    def test_multiply_bond_dimension_grows(self, n_sites):
        """Bond dim of product should be chi_A * chi_B."""
        rng = np.random.default_rng(3)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        AB = A.multiply(B)
        for W in AB.cores:
            chi_l, _, _, chi_r = W.shape
            assert chi_l == 1 and chi_r == 1  # 1*1 = 1 for bond-1 inputs

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_cdag_c_gives_number_op_via_multiply(self, n_sites):
        for i in range(n_sites):
            num = mpo_to_dense(
                f1d.cdag_fermions(i, n_sites).multiply(f1d.c_fermions(i, n_sites))
            )
            ops = [I] * i + [n_op] + [I] * (n_sites - i - 1)
            np.testing.assert_allclose(num, kron_op(ops), atol=1e-12)


# ─── tests MPO.addition ──────────────────────────────────────────────────────

class TestMPOAddition:

    def _random_bond1_mpo(self, n_sites, rng):
        cores = [rng.standard_normal((1, 2, 2, 1)) for _ in range(n_sites)]
        return mpo_class.MPO(cores)

    @pytest.mark.parametrize("n_sites", [1, 2, 3, 4])
    def test_addition_matches_matrix_sum(self, n_sites):
        rng = np.random.default_rng(10)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        np.testing.assert_allclose(
            mpo_to_dense(A.addition(B)),
            mpo_to_dense(A) + mpo_to_dense(B),
            atol=1e-12
        )

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_addition_commutative(self, n_sites):
        rng = np.random.default_rng(11)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        np.testing.assert_allclose(
            mpo_to_dense(A.addition(B)),
            mpo_to_dense(B.addition(A)),
            atol=1e-12
        )

    @pytest.mark.parametrize("n_sites", [2, 3])
    def test_addition_associative(self, n_sites):
        rng = np.random.default_rng(12)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        C = self._random_bond1_mpo(n_sites, rng)
        np.testing.assert_allclose(
            mpo_to_dense(A.addition(B).addition(C)),
            mpo_to_dense(A.addition(B.addition(C))),
            atol=1e-12
        )

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_addition_bond_dim_grows(self, n_sites):
        """Bond dim of sum should be chi_A + chi_B on interior bonds."""
        rng = np.random.default_rng(13)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        S = A.addition(B)
        for k, W in enumerate(S.cores):
            chi_l, _, _, chi_r = W.shape
            if n_sites == 1:
                assert chi_l == 1 and chi_r == 1
            elif k == 0:
                assert chi_l == 1 and chi_r == 2
            elif k == n_sites - 1:
                assert chi_l == 2 and chi_r == 1
            else:
                assert chi_l == 2 and chi_r == 2

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_addition_scalar_linearity(self, n_sites):
        """alpha*A + beta*B (via MPO ops) == (alpha+beta)*A when A==B."""
        rng = np.random.default_rng(14)
        A = self._random_bond1_mpo(n_sites, rng)
        alpha, beta = 3.0, -1.5
        lhs = mpo_to_dense(A.multiply_by_scalar(alpha).addition(A.multiply_by_scalar(beta)))
        rhs = (alpha + beta) * mpo_to_dense(A)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    @pytest.mark.parametrize("n_sites", [2, 3, 4])
    def test_addition_then_multiply_distributive(self, n_sites):
        """(A + B) * C = A*C + B*C"""
        rng = np.random.default_rng(15)
        A = self._random_bond1_mpo(n_sites, rng)
        B = self._random_bond1_mpo(n_sites, rng)
        C = self._random_bond1_mpo(n_sites, rng)
        lhs = mpo_to_dense(A.addition(B).multiply(C))
        rhs = mpo_to_dense(A.multiply(C)) + mpo_to_dense(B.multiply(C))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)
