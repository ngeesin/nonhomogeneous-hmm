import numpy as np
import pytest

from nhmm import _core


def _homogeneous_logtrans(n_obs, A):
    log_A = np.log(A)
    return np.repeat(log_A[None], n_obs - 1, axis=0)


def test_forward_matches_brute_force_homogeneous():
    # 2-state, 3-step chain; compare forward log-likelihood to the explicit
    # sum over all 2**3 state paths.
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.2, 0.8]])
    B = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])  # P(obs_t | state)

    log_alpha, ll = _core.forward(
        np.log(pi), _homogeneous_logtrans(3, A), np.log(B)
    )

    total = 0.0
    for s0 in range(2):
        for s1 in range(2):
            for s2 in range(2):
                p = pi[s0] * B[0, s0]
                p *= A[s0, s1] * B[1, s1]
                p *= A[s1, s2] * B[2, s2]
                total += p
    assert ll == pytest.approx(np.log(total))


def test_gamma_sums_to_one():
    rng = np.random.default_rng(1)
    n_obs, n_states = 7, 3
    log_start = np.log(np.full(n_states, 1.0 / n_states))
    log_trans = np.log(
        np.array(
            [rng.dirichlet(np.ones(n_states), size=n_states) for _ in range(n_obs - 1)]
        )
    )
    frameprob = np.log(rng.random((n_obs, n_states)))

    log_alpha, ll = _core.forward(log_start, log_trans, frameprob)
    log_beta = _core.backward(log_start, log_trans, frameprob)
    gamma, xi_sum, xi = _core.posteriors(log_alpha, log_beta, log_trans, frameprob, ll)

    assert np.allclose(gamma.sum(axis=1), 1.0)
    # xi over (i, j) sums to the marginal of the source state.
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)
    assert xi_sum.shape == (n_states, n_states)


def test_viterbi_recovers_obvious_path():
    # Emissions strongly favour state 0 then 1 then 0.
    pi = np.array([0.5, 0.5])
    A = np.array([[0.5, 0.5], [0.5, 0.5]])
    B = np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]])
    states, _ = _core.viterbi(np.log(pi), _homogeneous_logtrans(3, A), np.log(B))
    assert list(states) == [0, 1, 0]


def test_single_observation():
    log_start = np.log([0.3, 0.7])
    frameprob = np.log([[0.5, 0.5]])
    log_trans = np.empty((0, 2, 2))
    log_alpha, ll = _core.forward(log_start, log_trans, frameprob)
    log_beta = _core.backward(log_start, log_trans, frameprob)
    gamma, xi_sum, xi = _core.posteriors(log_alpha, log_beta, log_trans, frameprob, ll)
    assert gamma.shape == (1, 2)
    assert xi is None and xi_sum is None
