import numpy as np
import pytest

from nhmm import ForwardSimulation, GaussianEmissions, NonHomogeneousHMM
from nhmm.transitions import SoftmaxTransitions


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
