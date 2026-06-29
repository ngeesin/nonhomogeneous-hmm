import numpy as np
import pytest

from nhmm import (
    ForwardSimulation,
    GaussianEmissions,
    NonHomogeneousHMM,
    state_durations,
)
from nhmm.transitions import SoftmaxTransitions


def _build_model(n_features, weights, means=(-4.0, 4.0), var=0.25):
    """Hand-build a 2-state Gaussian model with given transition weights.

    Design-matrix columns are ``[intercept, raw_covariates...]``, so a raw
    covariate at index ``c`` corresponds to transition-weight feature ``1 + c``.
    """
    model = NonHomogeneousHMM(2)
    model.startprob_ = np.array([0.5, 0.5])
    em = GaussianEmissions(2, 1)
    em.means_ = np.array([[means[0]], [means[1]]])
    em.covars_ = np.array([[var], [var]])
    model.emissions_ = em
    tr = SoftmaxTransitions(2, n_features)
    tr.weights = np.asarray(weights, dtype=float)
    model.transitions_ = tr
    return model


def _history_in_state0(n=10, n_cov=1):
    """Short history whose emissions clearly place the final state at 0."""
    Y = np.full((n, 1), -4.0)
    X = np.zeros((n, n_cov))
    return Y, X


# A duration-only model (features = [intercept, duration]) in which the hazard
# of leaving the current state grows with how long it has been occupied.
_FORCED_EXIT_WEIGHTS = [
    [[2.0, -2.0], [0.0, 1.0]],   # from state 0: leave-logit = -2 + duration
    [[-2.0, 2.0], [1.0, 0.0]],   # from state 1: leave-logit = -2 + duration
]


def test_dynamic_returns_covariate_paths_and_shapes():
    model = _build_model(2, _FORCED_EXIT_WEIGHTS)
    Y, X = _history_in_state0()
    sim = model.simulate_forward(
        Y, X, X_future=None, horizon=8, duration_col=0,
        initial_duration=1, n_paths=200, random_state=0,
    )
    assert sim.covariate_paths.shape == (200, 8)
    assert sim.state_paths.shape == (200, 8)
    assert sim.y_paths.shape == (200, 8, 1)


def test_exogenous_mode_covariate_paths_is_none():
    model = _build_model(2, _FORCED_EXIT_WEIGHTS)
    Y, X = _history_in_state0()
    sim = model.simulate_forward(Y, X, np.ones((4, 1)), n_paths=10, random_state=0)
    assert sim.covariate_paths is None


def test_initial_duration_increments_when_sticky():
    # Stay-forever model: duration has no effect, so it just counts up.
    sticky = [[[10.0, -10.0], [0.0, 0.0]], [[-10.0, 10.0], [0.0, 0.0]]]
    model = _build_model(2, sticky)
    Y, X = _history_in_state0()
    sim = model.simulate_forward(
        Y, X, X_future=None, horizon=6, duration_col=0,
        initial_duration=3, n_paths=50, random_state=0,
    )
    # Every path stays in state 0; duration counts 3, 4, 5, ...
    expected = 3 + np.arange(6)
    assert np.all(sim.covariate_paths == expected)
    assert np.all(sim.state_paths == 0)


def test_duration_counting_invariant():
    model = _build_model(2, _FORCED_EXIT_WEIGHTS)
    Y, X = _history_in_state0()
    sim = model.simulate_forward(
        Y, X, X_future=None, horizon=20, duration_col=0,
        initial_duration=1, n_paths=500, random_state=1,
    )
    cp, sp = sim.covariate_paths, sim.state_paths
    for k in range(2, 20):
        same = sp[:, k - 1] == sp[:, k - 2]
        # When the origin state persisted, duration grew by 1; else it reset.
        assert np.all(cp[same, k] == cp[same, k - 1] + 1)
        assert np.all(cp[~same, k] == 1)


def test_duration_forces_bounded_sojourns():
    model = _build_model(2, _FORCED_EXIT_WEIGHTS)
    Y, X = _history_in_state0()
    horizon = 20
    sim = model.simulate_forward(
        Y, X, X_future=None, horizon=horizon, duration_col=0,
        initial_duration=1, n_paths=500, random_state=2,
    )
    # Rising hazard => sojourns reset well before the horizon, and states flip.
    assert sim.covariate_paths.max() < horizon
    assert (sim.covariate_paths[:, 1:] == 1).any()        # resets occur
    assert (sim.state_paths[:, 0] != sim.state_paths[:, -1]).any()


def test_combined_exogenous_and_duration():
    # Features = [intercept, exog(col0), duration(col1)]; exog strongly drives
    # the regime, duration column is overwritten online.
    weights = [
        [[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]],   # from 0: +exog -> state 1
        [[0.0, 0.0], [0.0, 3.0], [0.0, 0.0]],   # from 1: +exog -> state 1
    ]
    model = _build_model(3, weights)
    Y, X = _history_in_state0(n_cov=2)
    horizon = 6
    X_future = np.column_stack(
        [np.full(horizon, 4.0), np.full(horizon, 999.0)]  # exog forces 1; dur ignored
    )
    sim = model.simulate_forward(
        Y, X, X_future, duration_col=1, initial_duration=1,
        n_paths=1000, random_state=0,
    )
    # The exogenous column still steers the regime despite the bogus duration
    # placeholder, and the overwritten duration column is sensible (>= 1).
    assert sim.state_probs[-1, 1] > 0.9
    assert sim.covariate_paths.min() >= 1
    assert sim.covariate_paths.max() < 999  # the 999 placeholder was overwritten


def test_dynamic_determinism():
    model = _build_model(2, _FORCED_EXIT_WEIGHTS)
    Y, X = _history_in_state0()
    kw = dict(X_future=None, horizon=10, duration_col=0, n_paths=64, random_state=7)
    a = model.simulate_forward(Y, X, **kw)
    b = model.simulate_forward(Y, X, **kw)
    assert np.array_equal(a.state_paths, b.state_paths)
    assert np.array_equal(a.covariate_paths, b.covariate_paths)


def test_dynamic_validation_errors():
    model = _build_model(2, _FORCED_EXIT_WEIGHTS)
    Y, X = _history_in_state0()
    # duration_col out of range (n_cov == 1).
    with pytest.raises(ValueError):
        model.simulate_forward(Y, X, X_future=None, horizon=4, duration_col=3)
    # No X_future and no horizon.
    with pytest.raises(ValueError):
        model.simulate_forward(Y, X, X_future=None, duration_col=0)

    # Model with an extra exogenous covariate requires X_future.
    weights = [
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]
    model2 = _build_model(3, weights)
    Y2, X2 = _history_in_state0(n_cov=2)
    with pytest.raises(ValueError):
        model2.simulate_forward(Y2, X2, X_future=None, horizon=5, duration_col=1)


def test_state_durations_builds_history_covariate():
    # The helper is the intended way to construct the duration history column.
    states = np.array([0, 0, 0, 1, 1, 0])
    d = state_durations(states)
    assert list(d) == [1, 2, 3, 1, 2, 1]


def _sticky_model():
    """Hand-build a 2-state model with sticky, intercept-driven transitions.

    State 0 emits near -4, state 1 near +4, and each state strongly prefers to
    stay put regardless of the covariate. Lets us isolate the effect of the
    *filtered start* on early simulation steps.
    """
    model = NonHomogeneousHMM(2)
    model.startprob_ = np.array([0.5, 0.5])

    em = GaussianEmissions(2, 1)
    em.means_ = np.array([[-4.0], [4.0]])
    em.covars_ = np.array([[0.25], [0.25]])
    model.emissions_ = em

    tr = SoftmaxTransitions(2, 2)  # features = [intercept, covariate]
    # Intercept logits make the self-transition ~0.98; covariate column is zero.
    tr.weights[0, 0] = [2.0, -2.0]  # from state 0 -> favour state 0
    tr.weights[1, 0] = [-2.0, 2.0]  # from state 1 -> favour state 1
    model.transitions_ = tr
    return model


def _fit_covariate_driven_model(seed=0, n=1500):
    """Fit a 2-state model whose regime is strongly driven by one covariate."""
    rng = np.random.default_rng(seed)
    cov = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), cov])
    tr = SoftmaxTransitions(2, 2)
    # Large positive covariate -> jump to / stay in state 1; negative -> state 0.
    tr.weights = np.array(
        [[[0.0, 0.0], [0.0, 3.0]], [[0.0, 0.0], [0.0, 3.0]]]
    )
    A = tr.transition_matrices(X)
    means = np.array([-4.0, 4.0])
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=A[t, states[t - 1]])
    Y = rng.normal(means[states], 0.5)[:, None]

    model = NonHomogeneousHMM(2, random_state=1, n_iter=100).fit(Y, cov)
    return model, cov, Y


def test_output_shapes():
    model, cov, Y = _fit_covariate_driven_model()
    horizon, n_paths = 8, 200
    X_future = np.zeros((horizon, 1))
    sim = model.simulate_forward(Y, cov, X_future, n_paths=n_paths, random_state=0)

    assert isinstance(sim, ForwardSimulation)
    assert sim.y_paths.shape == (n_paths, horizon, 1)
    assert sim.state_paths.shape == (n_paths, horizon)
    assert sim.mean.shape == (horizon, 1)
    assert sim.quantiles.shape == (3, horizon, 1)
    assert sim.state_probs.shape == (horizon, 2)
    # Regime probabilities are a valid distribution at each step.
    assert np.allclose(sim.state_probs.sum(axis=1), 1.0)


def test_horizon_follows_x_future():
    model, cov, Y = _fit_covariate_driven_model()
    sim = model.simulate_forward(Y, cov, np.zeros((5, 1)), n_paths=10, random_state=0)
    assert sim.y_paths.shape[1] == 5


def test_determinism():
    model, cov, Y = _fit_covariate_driven_model()
    X_future = np.linspace(-1, 1, 6)[:, None]
    a = model.simulate_forward(Y, cov, X_future, n_paths=50, random_state=42)
    b = model.simulate_forward(Y, cov, X_future, n_paths=50, random_state=42)
    assert np.array_equal(a.state_paths, b.state_paths)
    assert np.allclose(a.y_paths, b.y_paths)


def test_covariate_forces_regime():
    model, cov, Y = _fit_covariate_driven_model()
    # Identify which fitted state is the "high" regime.
    high = int(np.argmax(model.emissions_.means_.ravel()))
    high_mean = model.emissions_.means_[high, 0]

    # Strongly positive covariate should force the high regime for all steps.
    X_future = np.full((10, 1), 4.0)
    sim = model.simulate_forward(Y, cov, X_future, n_paths=2000, random_state=0)

    # After a step or two the chain should be overwhelmingly in the high state.
    assert sim.state_probs[-1, high] > 0.9
    # Predictive mean should sit near that regime's emission mean.
    assert sim.mean[-1, 0] == pytest.approx(high_mean, abs=0.6)


def test_filtered_start_reflects_history():
    model = _sticky_model()
    X_future = np.zeros((3, 1))  # neutral covariates; stickiness comes from intercept

    # History ending clearly in the high regime (state 1).
    sim_high = model.simulate_forward(
        np.full((10, 1), 4.0), np.zeros(10), X_future, n_paths=4000, random_state=0
    )
    # History ending clearly in the low regime (state 0).
    sim_low = model.simulate_forward(
        np.full((10, 1), -4.0), np.zeros(10), X_future, n_paths=4000, random_state=0
    )

    # The filtered start carries the ending regime into the first step.
    assert sim_high.state_probs[0, 1] > 0.9
    assert sim_low.state_probs[0, 0] > 0.9


def test_wrong_covariate_width_raises():
    model, cov, Y = _fit_covariate_driven_model()
    # Model was trained with one covariate (+intercept); two columns is invalid.
    with pytest.raises(ValueError):
        model.simulate_forward(Y, cov, np.zeros((4, 2)), n_paths=5, random_state=0)


def test_custom_quantile_levels():
    model, cov, Y = _fit_covariate_driven_model()
    sim = model.simulate_forward(
        Y, cov, np.zeros((4, 1)), n_paths=100, quantiles=(0.1, 0.9), random_state=0
    )
    assert sim.quantile_levels == (0.1, 0.9)
    assert sim.quantiles.shape == (2, 4, 1)
    # Lower band must not exceed the upper band.
    assert np.all(sim.quantiles[0] <= sim.quantiles[1])


def test_intercept_only_forward():
    rng = np.random.default_rng(3)
    Y = np.vstack([rng.normal(0, 1, (100, 1)), rng.normal(6, 1, (100, 1))])
    model = NonHomogeneousHMM(2, random_state=0, n_iter=30).fit(Y)  # no covariates
    # Intercept-only: future covariate matrix is just the per-step driver of
    # length `horizon` with zero extra columns.
    sim = model.simulate_forward(
        Y, None, np.zeros((5, 0)), n_paths=50, random_state=0
    )
    assert sim.y_paths.shape == (50, 5, 1)
    assert np.isfinite(sim.mean).all()
