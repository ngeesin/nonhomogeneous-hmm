import numpy as np
import pytest

from nhmm import GaussianEmissions, NonHomogeneousHMM, _core
from nhmm.transitions import SoftmaxTransitions


def _sample_homogeneous(rng, n, A_true, means=(-3.0, 3.0), sd=0.5):
    """Generate a plain homogeneous 2-state HMM with a known matrix."""
    A_true = np.asarray(A_true)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=A_true[states[t - 1]])
    Y = rng.normal(np.asarray(means)[states], sd)
    return Y[:, None], states


def _sample_nhmm(rng, n, w_intercept, w_cov):
    """Generate a 2-state NHMM whose transitions depend on one covariate."""
    cov = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), cov])
    tr = SoftmaxTransitions(2, 2)
    tr.weights = np.array([w_intercept, w_cov])  # (from_state, feature, to_state)
    A = tr.transition_matrices(X)

    means = np.array([-3.0, 3.0])
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=A[t, states[t - 1]])
    Y = rng.normal(means[states], 0.6)
    return cov, Y[:, None], states


def test_loglikelihood_increases_monotonically():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 1))
    Y = np.vstack([rng.normal(0, 1, (150, 1)), rng.normal(4, 1, (150, 1))])
    model = NonHomogeneousHMM(2, random_state=0, n_iter=30).fit(Y, X)
    hist = np.array(model.monitor_.history)
    # EM must never decrease the log-likelihood (allow tiny numerical slack).
    assert np.all(np.diff(hist) > -1e-6)


def test_recovers_states_and_covariate_effect():
    rng = np.random.default_rng(42)
    # Strong covariate effect on the transition out of each state.
    w_intercept = [[0.0, 0.0], [0.0, 0.0]]
    w_cov = [[0.0, 3.0], [0.0, -3.0]]
    cov, Y, states = _sample_nhmm(rng, 1500, w_intercept, w_cov)

    model = NonHomogeneousHMM(2, random_state=1, n_iter=100).fit(Y, cov)
    pred = model.predict(Y, cov)

    # Account for arbitrary label permutation.
    acc = max((pred == states).mean(), (pred == 1 - states).mean())
    assert acc > 0.9

    # The fitted transition probabilities should vary with the covariate.
    Xgrid = model._design_matrix(np.array([[-2.0], [2.0]]), 2)
    A = model.transitions_.transition_matrices(Xgrid)
    # Self-transition probability of at least one state changes substantially.
    spread = np.abs(A[0].diagonal() - A[1].diagonal()).max()
    assert spread > 0.2


def test_multiple_sequences_via_lengths():
    rng = np.random.default_rng(3)
    seqs_Y, seqs_X, lengths = [], [], []
    for _ in range(4):
        n = rng.integers(40, 80)
        X = rng.standard_normal((n, 1))
        Y = np.vstack([rng.normal(0, 1, (n // 2, 1)), rng.normal(5, 1, (n - n // 2, 1))])
        seqs_Y.append(Y)
        seqs_X.append(X)
        lengths.append(n)
    Y = np.vstack(seqs_Y)
    X = np.vstack(seqs_X)

    model = NonHomogeneousHMM(2, random_state=0, n_iter=20).fit(Y, X, lengths=lengths)
    proba = model.predict_proba(Y, X, lengths=lengths)
    assert proba.shape == (len(Y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Score equals the converged EM log-likelihood.
    assert model.score(Y, X, lengths=lengths) == pytest.approx(
        model.monitor_.history[-1], rel=1e-6
    )


def test_homogeneous_special_case_runs_without_covariates():
    rng = np.random.default_rng(7)
    Y = np.vstack([rng.normal(0, 1, (100, 1)), rng.normal(6, 1, (100, 1))])
    model = NonHomogeneousHMM(2, random_state=0, n_iter=30).fit(Y)  # intercept only
    states = model.predict(Y)
    acc = max((states == np.r_[np.zeros(100), np.ones(100)]).mean(),
              (states == np.r_[np.ones(100), np.zeros(100)]).mean())
    assert acc > 0.9


def test_sample_then_refit_roundtrip():
    rng = np.random.default_rng(11)
    model = NonHomogeneousHMM(2, random_state=0)
    # Build a fitted-looking model by fitting on quick synthetic data first.
    Y0 = np.vstack([rng.normal(-2, 0.5, (80, 1)), rng.normal(2, 0.5, (80, 1))])
    X0 = rng.standard_normal((160, 1))
    model.fit(Y0, X0)

    X = rng.standard_normal((400, 1))
    Y, states = model.sample(400, X, random_state=5)
    assert Y.shape == (400, 1)
    assert states.shape == (400,)
    # Log-likelihood of generated data should be finite.
    assert np.isfinite(model.score(Y, X))


def test_categorical_emissions():
    rng = np.random.default_rng(2)
    # Two states with distinct symbol distributions.
    s = np.r_[np.zeros(150, int), np.ones(150, int)]
    Y = np.empty(300, dtype=int)
    Y[s == 0] = rng.choice(3, size=(s == 0).sum(), p=[0.8, 0.1, 0.1])
    Y[s == 1] = rng.choice(3, size=(s == 1).sum(), p=[0.1, 0.1, 0.8])
    X = rng.standard_normal((300, 1))

    model = NonHomogeneousHMM(
        2, emissions="categorical", n_symbols=3, random_state=0, n_iter=50
    ).fit(Y, X)
    pred = model.predict(Y, X)
    acc = max((pred == s).mean(), (pred == 1 - s).mean())
    assert acc > 0.85


def test_custom_emission_instance():
    rng = np.random.default_rng(0)
    em = GaussianEmissions(2, 1, covariance_type="full")
    Y = np.vstack([rng.normal(0, 1, (60, 1)), rng.normal(5, 1, (60, 1))])
    model = NonHomogeneousHMM(2, emissions=em, random_state=0, n_iter=20).fit(Y)
    assert model.emissions_ is em
    assert np.isfinite(model.score(Y))


def test_mean_transition_matrix_rows_normalised():
    rng = np.random.default_rng(0)
    cov, Y, _ = _sample_nhmm(rng, 800, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 2.0], [0.0, -2.0]])
    model = NonHomogeneousHMM(2, random_state=0, n_iter=50).fit(Y, cov)
    A = model.mean_transition_matrix(Y, cov)
    assert A.shape == (2, 2)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_mean_transition_matrix_homogeneous_consistency():
    rng = np.random.default_rng(1)
    A_true = np.array([[0.9, 0.1], [0.2, 0.8]])
    Y, _ = _sample_homogeneous(rng, 6000, A_true)
    model = NonHomogeneousHMM(2, random_state=0, n_iter=100).fit(Y)  # intercept only

    A_est = model.mean_transition_matrix(Y)
    # The model's own constant transition matrix (intercept-only design row).
    A_model = model.transitions_.transition_matrices(model._design_matrix(None, 1))[0]

    # Align label permutation by sorting on the fitted emission means.
    order = np.argsort(model.emissions_.means_.ravel())
    A_est = A_est[np.ix_(order, order)]
    A_model = A_model[np.ix_(order, order)]

    assert np.allclose(A_est, A_true, atol=0.05)
    assert np.allclose(A_est, A_model, atol=0.05)


def test_mean_transition_matrix_matches_normalized_xi():
    rng = np.random.default_rng(2)
    cov, Y, _ = _sample_nhmm(rng, 400, [[0.0, 0.0], [0.0, 0.0]], [[0.0, 2.0], [0.0, -2.0]])
    model = NonHomogeneousHMM(2, random_state=0, n_iter=40).fit(Y, cov)

    # Direct forward-backward on the same data -> normalised summed xi.
    Xd = model._design_matrix(cov, len(Y))
    log_trans = model.transitions_.log_transition_matrices(Xd)[1:]
    frameprob = model.emissions_.log_likelihood(Y)
    log_start = np.log(model.startprob_)
    la, ll = _core.forward(log_start, log_trans, frameprob)
    lb = _core.backward(log_start, log_trans, frameprob)
    _, xi_sum, _ = _core.posteriors(la, lb, log_trans, frameprob, ll)
    expected = xi_sum / xi_sum.sum(axis=1, keepdims=True)

    assert np.allclose(model.mean_transition_matrix(Y, cov), expected)


def test_mean_transition_matrix_multiple_sequences():
    rng = np.random.default_rng(3)
    Y = np.vstack([rng.normal(0, 1, (60, 1)), rng.normal(5, 1, (60, 1)),
                   rng.normal(0, 1, (40, 1))])
    X = rng.standard_normal((160, 1))
    lengths = [60, 60, 40]
    model = NonHomogeneousHMM(2, random_state=0, n_iter=20).fit(Y, X, lengths=lengths)
    A = model.mean_transition_matrix(Y, X, lengths=lengths)
    assert A.shape == (2, 2)
    assert np.allclose(A.sum(axis=1), 1.0)
