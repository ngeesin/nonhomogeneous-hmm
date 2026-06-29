# nhmm

**Non-homogeneous hidden Markov models with covariate-driven transitions.**

In a classical (homogeneous) hidden Markov model the probability of moving
between hidden states is fixed for all time. In a **non-homogeneous HMM
(NHMM)** the transition probabilities change over time as a function of
external, time-varying covariates. `nhmm` models those transitions with a
per-origin-state softmax (multinomial-logistic) regression:

$$
P(z_t = j \mid z_{t-1} = i, \mathbf{x}_t)
  = \frac{\exp(\mathbf{w}_{ij}^\top \mathbf{x}_t)}
         {\sum_k \exp(\mathbf{w}_{ik}^\top \mathbf{x}_t)} .
$$

Models like this are widely used for weather/rainfall downscaling
(Hughes & Guttorp), econometric regime-switching, ecology and any setting
where regime changes are driven by measurable inputs. With an intercept-only
covariate the model collapses exactly to a standard homogeneous HMM, so the
classical case is a strict special case.

## Features

- Covariate-dependent transitions via a softmax model, fit with EM
  (Baum-Welch); the M-step is a weighted multinomial-logistic regression
  solved with L-BFGS.
- Pluggable emissions: `gaussian` (diagonal or full covariance) and
  `categorical`, or bring your own by implementing the small
  `BaseEmissions` interface.
- Numerically stable log-space forward-backward and Viterbi decoding.
- Multiple sequences via a `lengths` argument (hmmlearn-style API).
- Posterior probabilities, Viterbi decoding, scoring and sampling.
- Pure NumPy/SciPy, no compiled extensions.

## Installation

```bash
pip install nhmm            # from PyPI (once published)
pip install -e .            # from a checkout
```

Requires Python ≥ 3.9, NumPy and SciPy.

## Quick start

```python
import numpy as np
from nhmm import NonHomogeneousHMM

rng = np.random.default_rng(0)

# A covariate that drives the regime, plus toy 1-D observations.
X = rng.standard_normal((1000, 1))
Y = np.where(X[:, 0] > 0, rng.normal(3, 1, 1000), rng.normal(-3, 1, 1000))[:, None]

model = NonHomogeneousHMM(n_states=2, random_state=0).fit(Y, X)

states = model.predict(Y, X)         # Viterbi decoding
gamma = model.predict_proba(Y, X)    # smoothed state posteriors
ll = model.score(Y, X)               # total log-likelihood

# Inspect how the transition matrix depends on the covariate:
A = model.transitions_.transition_matrices(model._design_matrix([[2.0]], 1))
```

### Multiple sequences

Concatenate the sequences and pass their lengths:

```python
Y = np.vstack([seq1, seq2, seq3])
X = np.vstack([cov1, cov2, cov3])
model.fit(Y, X, lengths=[len(seq1), len(seq2), len(seq3)])
```

### Categorical emissions

```python
model = NonHomogeneousHMM(
    n_states=3, emissions="categorical", n_symbols=5, random_state=0
).fit(symbol_codes, X)
```

### Sampling

```python
Y, states = model.sample(500, X)     # generate observations along covariates X
```

### Forward simulation / forecasting

`simulate_forward` draws Monte-Carlo future paths *conditioned on observed
history*. It computes the filtered distribution of the current regime from
`(Y, X)`, then rolls the chain forward over a horizon you define with a
**future covariate path** `X_future`.

> Because covariates are **exogenous**, the model cannot generate `X` for you —
> you must supply `X_future` (e.g. a macro scenario). Its columns, order and
> scaling must match the `X` used in `fit` (apply the same standardisation).

```python
horizon = 12
X_future = np.linspace(-1, 2, horizon)[:, None]   # a covariate scenario

sim = model.simulate_forward(Y, X, X_future, n_paths=5000, random_state=0)

sim.y_paths        # (n_paths, horizon[, n_dim]) raw simulated observations
sim.state_paths    # (n_paths, horizon)          simulated regimes
sim.mean           # (horizon[, n_dim])          predictive mean
sim.quantiles      # (n_levels, horizon[, n_dim]) predictive bands
sim.state_probs    # (horizon, n_states)         per-step regime probabilities
```

See [`examples/forecast_paths.py`](examples/forecast_paths.py) for a runnable
end-to-end forecast.

#### Duration-dependent (endogenous) covariate

Sometimes a covariate is the **sojourn duration** — how many consecutive steps
the chain has spent in its current state (a duration-dependent / semi-Markov
style model). That covariate is *endogenous*: future durations depend on the
simulated state path, so it can't be supplied up front. `simulate_forward`
computes it **online** when you point it at the duration column with
`duration_col`:

```python
from nhmm import state_durations

# Build the duration column for the history (and the starting run-length) from
# the known/decoded history states — the same way you built it for fit():
X_hist = state_durations(history_states)[:, None].astype(float)
init_dur = int(state_durations(history_states)[-1])

# Duration is the only covariate -> no X_future, just a horizon:
sim = model.simulate_forward(
    Y, X_hist, X_future=None, horizon=12,
    duration_col=0, initial_duration=init_dur, n_paths=5000,
)

sim.covariate_paths   # (n_paths, horizon) the duration fed at each step
```

- `duration_col` is the index of the duration covariate among the raw columns.
- If training **also** used other exogenous covariates, pass those via
  `X_future` (full width); only the duration column is overwritten online.
- The running duration is fed as a **raw count**, so fit the model on raw
  counts too. Build the history/training duration column with
  [`state_durations`](src/nhmm/_utils.py).

> Note: at fit time the duration is treated as an ordinary (precomputed)
> covariate; it only becomes endogenous during simulation. See
> [`examples/forecast_duration.py`](examples/forecast_duration.py).

## API overview

| Method | Description |
| --- | --- |
| `fit(Y, X=None, lengths=None)` | Estimate parameters with EM. |
| `predict(Y, X=None, lengths=None)` | Viterbi hidden-state sequence. |
| `predict_proba(Y, X=None, lengths=None)` | Smoothed state posteriors. |
| `decode(Y, X=None, lengths=None)` | `(log_prob, states)` via Viterbi. |
| `score(Y, X=None, lengths=None)` | Total log-likelihood. |
| `sample(n_samples, X=None)` | Generate `(Y, states)` unconditionally. |
| `simulate_forward(Y, X, X_future=None, *, duration_col=None, ...)` | Monte-Carlo forecast conditioned on history (exogenous or online duration covariate); returns a `ForwardSimulation`. |

Module-level helper: `state_durations(states)` — running sojourn length of a
state sequence, for building duration covariates.

Key attributes after fitting: `startprob_`, `transitions_`
(`SoftmaxTransitions`) and `emissions_`.

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
