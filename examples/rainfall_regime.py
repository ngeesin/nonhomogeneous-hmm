"""Example: a non-homogeneous HMM whose regime is driven by a covariate.

We simulate a two-regime process (a "dry" and a "wet" state) where the
probability of switching regime depends on a seasonal covariate.  We then fit a
:class:`~nhmm.NonHomogeneousHMM`, recover the hidden states and show that the
estimated transition probabilities vary with the covariate.

Run with::

    python examples/rainfall_regime.py
"""

import numpy as np

from nhmm import NonHomogeneousHMM
from nhmm.transitions import SoftmaxTransitions


def simulate(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # Seasonal covariate in roughly [-1, 1].
    season = np.sin(np.linspace(0, 8 * np.pi, n))
    X = np.column_stack([np.ones(n), season])

    truth = SoftmaxTransitions(2, 2)
    # When the season is high, the wet state (1) becomes "sticky".
    truth.weights = np.array(
        [
            [[0.0, 0.0], [0.0, 2.5]],   # from dry: covariate pushes toward wet
            [[0.0, 0.0], [0.0, 2.5]],   # from wet: covariate keeps it wet
        ]
    )
    A = truth.transition_matrices(X)

    means = np.array([0.0, 8.0])  # mm of rain
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(2, p=A[t, states[t - 1]])
    rain = rng.normal(means[states], 1.0)
    return season, rain[:, None], states


def main():
    season, rain, true_states = simulate()

    model = NonHomogeneousHMM(n_states=2, random_state=0, n_iter=100, verbose=True)
    model.fit(rain, season)

    pred = model.predict(rain, season)
    acc = max((pred == true_states).mean(), (pred == 1 - true_states).mean())
    print(f"\nState recovery accuracy: {acc:.3f}")
    print(f"Final log-likelihood:    {model.score(rain, season):.1f}")

    # How does the wet-state self-transition change with the season?
    grid = np.array([[-1.0], [1.0]])
    A = model.transitions_.transition_matrices(model._design_matrix(grid, 2))
    print("\nP(stay in fitted-wet state):")
    wet = int(np.argmax(model.emissions_.means_.ravel()))
    print(f"  low season  -> {A[0, wet, wet]:.3f}")
    print(f"  high season -> {A[1, wet, wet]:.3f}")


if __name__ == "__main__":
    main()
