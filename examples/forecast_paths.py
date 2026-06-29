"""Example: Monte-Carlo forward simulation from a fitted NHMM.

We fit a two-regime model whose switching is driven by a covariate, then use
:meth:`NonHomogeneousHMM.simulate_forward` to draw forward paths conditioned on
the observed history and a *future* covariate scenario.

Because covariates are exogenous, you must supply the future covariate path
yourself (the model cannot generate it). Here we feed a scenario where the
covariate ramps up, pushing the system toward the high regime.

Run with::

    python examples/forecast_paths.py
"""

import numpy as np

from nhmm import NonHomogeneousHMM
from nhmm.transitions import SoftmaxTransitions


def simulate_history(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    cov = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), cov])
    truth = SoftmaxTransitions(2, 2)
    truth.weights = np.array([[[0.0, 0.0], [0.0, 3.0]], [[0.0, 0.0], [0.0, 3.0]]])
    A = truth.transition_matrices(X)
    means = np.array([-4.0, 4.0])
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=A[t, states[t - 1]])
    Y = rng.normal(means[states], 0.5)[:, None]
    return cov, Y


def main():
    cov, Y = simulate_history()
    model = NonHomogeneousHMM(n_states=2, random_state=1, n_iter=100).fit(Y, cov)

    horizon = 12
    # Future covariate scenario: ramp from -1 up to +2 over the horizon.
    X_future = np.linspace(-1.0, 2.0, horizon)[:, None]

    sim = model.simulate_forward(Y, cov, X_future, n_paths=5000, random_state=0)

    high = int(np.argmax(model.emissions_.means_.ravel()))
    print(f"Horizon: {horizon} steps, {sim.y_paths.shape[0]} Monte-Carlo paths\n")
    print(" step  cov   P(high)   pred-mean   [ 5%, 95% ]")
    for k in range(horizon):
        lo, _, hi = sim.quantiles[:, k, 0]
        print(
            f"  {k:3d}  {X_future[k, 0]:+.2f}   {sim.state_probs[k, high]:.3f}    "
            f"{sim.mean[k, 0]:+7.3f}   [{lo:+6.2f}, {hi:+6.2f}]"
        )

    print(
        "\nAs the covariate rises, P(high regime) increases and the predictive "
        "mean shifts toward the high-state emission level."
    )


if __name__ == "__main__":
    main()
