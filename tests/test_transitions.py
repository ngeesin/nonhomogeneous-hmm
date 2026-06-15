import numpy as np
import pytest

from nhmm.transitions import SoftmaxTransitions


def test_rows_are_normalised():
    rng = np.random.default_rng(0)
    tr = SoftmaxTransitions(3, 2).init_params(rng)
    X = rng.standard_normal((10, 2))
    A = tr.transition_matrices(X)
    assert A.shape == (10, 3, 3)
    assert np.allclose(A.sum(axis=2), 1.0)


def test_intercept_only_is_homogeneous():
    rng = np.random.default_rng(1)
    tr = SoftmaxTransitions(3, 1).init_params(rng)
    X = np.ones((5, 1))
    A = tr.transition_matrices(X)
    # All time steps share the same matrix when the only feature is constant.
    assert np.allclose(A, A[0])


def test_mstep_recovers_known_softmax():
    # Generate xi proportional to a known softmax and check the M-step recovers
    # transition probabilities that match.
    rng = np.random.default_rng(2)
    n, n_states, n_features = 4000, 2, 2
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])

    true = SoftmaxTransitions(n_states, n_features)
    true.weights = np.array(
        [
            [[0.0, 1.5], [0.0, -2.0]],  # from state 0
            [[0.0, -1.0], [0.0, 0.5]],  # from state 1
        ]
    )
    A_true = true.transition_matrices(X)

    # Expected counts: one unit of mass from each (uniform over origin states).
    xi = np.zeros((n, n_states, n_states))
    for i in range(n_states):
        xi[:, i, :] = 0.5 * A_true[:, i, :]

    est = SoftmaxTransitions(n_states, n_features, reg=1e-6)
    est.m_step(X, xi)
    A_est = est.transition_matrices(X)

    assert np.allclose(A_est, A_true, atol=0.05)
