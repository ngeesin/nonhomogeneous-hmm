"""Example: forward simulation with an endogenous sojourn-duration covariate.

Here the transition probabilities depend on the *duration* the chain has spent
in its current state (a duration-dependent / semi-Markov-style model). Because
that covariate is a function of the simulated state path, it cannot be supplied
up front -- :meth:`NonHomogeneousHMM.simulate_forward` computes it online, one
step at a time, via ``duration_col``.

We hand-build a two-state model whose hazard of *leaving* a state rises with how
long it has been occupied, then simulate forward and watch the sojourns
self-limit.

Run with::

    python examples/forecast_duration.py
"""

import numpy as np

from nhmm import GaussianEmissions, NonHomogeneousHMM, state_durations
from nhmm.transitions import SoftmaxTransitions


def build_model():
    """2-state model; covariate = [intercept, duration]. Leave-logit = -2 + d."""
    model = NonHomogeneousHMM(2)
    model.startprob_ = np.array([0.5, 0.5])

    em = GaussianEmissions(2, 1)
    em.means_ = np.array([[-4.0], [4.0]])
    em.covars_ = np.array([[0.25], [0.25]])
    model.emissions_ = em

    tr = SoftmaxTransitions(2, 2)  # features = [intercept, duration]
    tr.weights = np.array(
        [
            [[2.0, -2.0], [0.0, 1.0]],   # from state 0: P(leave) grows with duration
            [[-2.0, 2.0], [1.0, 0.0]],   # from state 1: P(leave) grows with duration
        ]
    )
    model.transitions_ = tr
    return model


def main():
    model = build_model()

    # A short history ending in state 0. The duration column is built with the
    # state_durations helper from the (here known) history states.
    hist_states = np.zeros(6, dtype=int)
    Y = np.full((6, 1), -4.0)
    X = state_durations(hist_states)[:, None].astype(float)  # duration history
    initial_duration = int(state_durations(hist_states)[-1])  # = 6

    horizon = 15
    sim = model.simulate_forward(
        Y, X, X_future=None, horizon=horizon, duration_col=0,
        initial_duration=initial_duration, n_paths=5000, random_state=0,
    )

    print(f"Initial duration carried in from history: {initial_duration}")
    print(f"Horizon: {horizon}, paths: {sim.y_paths.shape[0]}\n")
    print(" step  P(state0)  P(state1)  mean-duration-in-state")
    for k in range(horizon):
        print(
            f"  {k:3d}    {sim.state_probs[k, 0]:.3f}      "
            f"{sim.state_probs[k, 1]:.3f}        {sim.covariate_paths[:, k].mean():.2f}"
        )

    print("\nTwo example paths (duration fed each step -> simulated state):")
    for p in range(2):
        pairs = ", ".join(
            f"{int(d)}->{s}"
            for d, s in zip(sim.covariate_paths[p], sim.state_paths[p])
        )
        print(f"  path {p}: {pairs}")

    print(
        "\nThe carried-in duration (6) makes an early switch very likely; "
        "thereafter sojourns reset and self-limit as the duration-driven hazard "
        "rises, so the mean in-state duration stays small."
    )


if __name__ == "__main__":
    main()
