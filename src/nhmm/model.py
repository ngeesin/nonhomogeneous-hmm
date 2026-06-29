"""The non-homogeneous hidden Markov model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from ._utils import check_random_state, iter_sequences, log_normalize, normalize
from .emissions import (
    BaseEmissions,
    CategoricalEmissions,
    GaussianEmissions,
)
from .transitions import SoftmaxTransitions


@dataclass
class ForwardSimulation:
    """Result of :meth:`NonHomogeneousHMM.simulate_forward`.

    Attributes
    ----------
    y_paths : ndarray of shape (n_paths, horizon) or (n_paths, horizon, n_dim)
        Simulated future observations, one row per Monte-Carlo path.
    state_paths : ndarray of shape (n_paths, horizon)
        Simulated hidden-state paths.
    mean : ndarray of shape (horizon,) or (horizon, n_dim)
        Predictive mean of the observations across paths.
    quantiles : ndarray of shape (n_levels, horizon[, n_dim])
        Predictive quantile bands of the observations across paths.
    quantile_levels : tuple of float
        The probability levels matching the first axis of ``quantiles``.
    state_probs : ndarray of shape (horizon, n_states)
        Per-step probability of occupying each hidden state.
    covariate_paths : ndarray of shape (n_paths, horizon) or None
        In dynamic duration mode, the sojourn-duration value fed to the model
        at each step of each path (the online ``X_future``). ``None`` when a
        fixed exogenous ``X_future`` was used.
    """

    y_paths: np.ndarray
    state_paths: np.ndarray
    mean: np.ndarray
    quantiles: np.ndarray
    quantile_levels: tuple
    state_probs: np.ndarray
    covariate_paths: np.ndarray | None = None


class ConvergenceMonitor:
    """Track the EM log-likelihood and decide when to stop."""

    def __init__(self, tol, n_iter, verbose=False):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history: list[float] = []

    def report(self, log_likelihood):
        if self.verbose:
            delta = (
                log_likelihood - self.history[-1] if self.history else float("nan")
            )
            print(
                f"  iter {len(self.history) + 1:3d}  "
                f"loglik={log_likelihood: .6f}  delta={delta: .3e}"
            )
        self.history.append(log_likelihood)

    @property
    def converged(self):
        return (
            len(self.history) >= 2
            and abs(self.history[-1] - self.history[-2]) < self.tol
        ) or len(self.history) >= self.n_iter


class NonHomogeneousHMM:
    """Hidden Markov model with covariate-dependent transition probabilities.

    The transition matrix at each time step is produced by a softmax model that
    depends on external covariates (:class:`~nhmm.transitions.SoftmaxTransitions`),
    making the chain *non-homogeneous*.  Emissions are handled by a pluggable
    emission model.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    emissions : {"gaussian", "categorical"} or BaseEmissions, default="gaussian"
        Either a ready-made emission model instance or a string selecting one of
        the built-ins (configured at :meth:`fit` time from data and the keyword
        arguments below).
    covariance_type : {"diag", "full"}, default="diag"
        Covariance structure for ``"gaussian"`` emissions.
    n_symbols : int, optional
        Alphabet size for ``"categorical"`` emissions.
    fit_intercept : bool, default=True
        Prepend a constant column of ones to the covariates so that transitions
        always have a bias term.  With ``fit_intercept=True`` and no other
        covariates the model is a standard homogeneous HMM.
    transition_reg : float, default=1e-4
        L2 regularisation for the transition weights.
    n_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence threshold on the change in per-fit log-likelihood.
    random_state : int, Generator or None
        Controls initialisation and sampling.
    verbose : bool, default=False
        Print per-iteration log-likelihood.

    Attributes
    ----------
    startprob_ : ndarray of shape (n_states,)
    transitions_ : SoftmaxTransitions
    emissions_ : BaseEmissions
    monitor_ : ConvergenceMonitor
    """

    def __init__(
        self,
        n_states,
        emissions="gaussian",
        *,
        covariance_type="diag",
        n_symbols=None,
        fit_intercept=True,
        transition_reg=1e-4,
        n_iter=100,
        tol=1e-4,
        random_state=None,
        verbose=False,
    ):
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = int(n_states)
        self.emissions = emissions
        self.covariance_type = covariance_type
        self.n_symbols = n_symbols
        self.fit_intercept = fit_intercept
        self.transition_reg = float(transition_reg)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.verbose = verbose

    # -- covariate handling -----------------------------------------------
    def _design_matrix(self, X, n_obs):
        """Validate covariates and add the intercept column if requested."""
        if X is None:
            if not self.fit_intercept:
                raise ValueError(
                    "X is required when fit_intercept=False (no covariates and "
                    "no intercept leaves transitions undefined)"
                )
            return np.ones((n_obs, 1))
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != n_obs:
            raise ValueError(
                f"X has {X.shape[0]} rows but data has {n_obs} observations"
            )
        if self.fit_intercept:
            X = np.hstack([np.ones((n_obs, 1)), X])
        return X

    # -- model construction ------------------------------------------------
    def _build_emissions(self, Y):
        if isinstance(self.emissions, BaseEmissions):
            return self.emissions
        Y_arr = np.asarray(Y)
        if self.emissions == "gaussian":
            n_dim = 1 if Y_arr.ndim == 1 else Y_arr.shape[1]
            return GaussianEmissions(
                self.n_states, n_dim, covariance_type=self.covariance_type
            )
        if self.emissions == "categorical":
            n_symbols = self.n_symbols
            if n_symbols is None:
                n_symbols = int(Y_arr.astype(int).max()) + 1
            return CategoricalEmissions(self.n_states, n_symbols)
        raise ValueError(f"unknown emissions {self.emissions!r}")

    # -- training ----------------------------------------------------------
    def fit(self, Y, X=None, lengths=None):
        """Fit the model with the EM (Baum-Welch) algorithm.

        Parameters
        ----------
        Y : array-like
            Observations, concatenated across sequences. Shape ``(n_obs,)`` or
            ``(n_obs, n_dim)`` for Gaussian, ``(n_obs,)`` of integer codes for
            categorical emissions.
        X : array-like of shape (n_obs, n_covariates), optional
            Time-varying covariates aligned with ``Y``.
        lengths : array-like of int, optional
            Length of each individual sequence; must sum to ``n_obs``. When
            omitted all observations are treated as a single sequence.

        Returns
        -------
        self
        """
        rng = check_random_state(self.random_state)
        Y = np.asarray(Y)
        n_obs = Y.shape[0]
        Xd = self._design_matrix(X, n_obs)

        self.emissions_ = self._build_emissions(Y).init_params(Y, rng)
        self.transitions_ = SoftmaxTransitions(
            self.n_states, Xd.shape[1], reg=self.transition_reg
        ).init_params(rng)
        self.startprob_ = normalize(rng.random(self.n_states) + 1.0)

        sequences = list(iter_sequences(n_obs, lengths))
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

        for _ in range(self.n_iter):
            stats, total_ll = self._e_step(Y, Xd, sequences)
            self.monitor_.report(total_ll)
            self._m_step(Y, Xd, stats)
            if self.monitor_.converged:
                break
        return self

    def _e_step(self, Y, Xd, sequences):
        log_start = np.log(self.startprob_)
        gamma_full = np.zeros((Y.shape[0], self.n_states))
        start_acc = np.zeros(self.n_states)
        xi_rows_X = []
        xi_rows = []
        total_ll = 0.0

        for start, end in sequences:
            frameprob = self.emissions_.log_likelihood(Y[start:end])
            log_A = self.transitions_.log_transition_matrices(Xd[start:end])
            # Transition into time t (t>=1) is driven by Xd[start+t]; drop t=0.
            log_trans = log_A[1:]
            log_alpha, ll = _core.forward(log_start, log_trans, frameprob)
            log_beta = _core.backward(log_start, log_trans, frameprob)
            gamma, _, xi = _core.posteriors(
                log_alpha, log_beta, log_trans, frameprob, ll
            )
            gamma_full[start:end] = gamma
            start_acc += gamma[0]
            total_ll += ll
            if xi is not None:
                xi_rows.append(xi)
                xi_rows_X.append(Xd[start + 1 : end])

        stats = {
            "gamma": gamma_full,
            "start": start_acc,
            "xi": np.concatenate(xi_rows) if xi_rows else None,
            "xi_X": np.concatenate(xi_rows_X) if xi_rows_X else None,
        }
        return stats, total_ll

    def _m_step(self, Y, Xd, stats):
        self.startprob_ = normalize(stats["start"] + 1e-12)
        self.emissions_.m_step(Y, stats["gamma"])
        if stats["xi"] is not None:
            self.transitions_.m_step(stats["xi_X"], stats["xi"])

    # -- inference ---------------------------------------------------------
    def _per_sequence_logtrans(self, Xd, start, end):
        return self.transitions_.log_transition_matrices(Xd[start:end])[1:]

    def score(self, Y, X=None, lengths=None):
        """Return the total log-likelihood of the data under the model."""
        Y = np.asarray(Y)
        Xd = self._design_matrix(X, Y.shape[0])
        log_start = np.log(self.startprob_)
        total = 0.0
        for start, end in iter_sequences(Y.shape[0], lengths):
            frameprob = self.emissions_.log_likelihood(Y[start:end])
            log_trans = self._per_sequence_logtrans(Xd, start, end)
            _, ll = _core.forward(log_start, log_trans, frameprob)
            total += ll
        return total

    def predict_proba(self, Y, X=None, lengths=None):
        """Posterior state probabilities ``P(z_t | x_{1:T})`` for each time step."""
        Y = np.asarray(Y)
        Xd = self._design_matrix(X, Y.shape[0])
        log_start = np.log(self.startprob_)
        out = np.zeros((Y.shape[0], self.n_states))
        for start, end in iter_sequences(Y.shape[0], lengths):
            frameprob = self.emissions_.log_likelihood(Y[start:end])
            log_trans = self._per_sequence_logtrans(Xd, start, end)
            log_alpha, ll = _core.forward(log_start, log_trans, frameprob)
            log_beta = _core.backward(log_start, log_trans, frameprob)
            gamma, _, _ = _core.posteriors(log_alpha, log_beta, log_trans, frameprob, ll)
            out[start:end] = gamma
        return out

    def mean_transition_matrix(self, Y, X=None, lengths=None):
        """Occupancy-weighted historical mean transition matrix.

        Because the transition matrix is time-varying in a non-homogeneous HMM,
        this summarises it as the single matrix implied by the data:

        .. math::

            A[i, j] = \\frac{\\sum_t P(z_{t-1}=i, z_t=j \\mid \\text{data})}
                           {\\sum_t P(z_{t-1}=i \\mid \\text{data})}

        i.e. expected ``i -> j`` transition counts divided by expected time
        spent in ``i``, from the fitted model's smoothed posteriors over the
        supplied history. This is exactly the single transition matrix a
        *homogeneous* HMM would estimate, and equals the row-normalised sum of
        the per-step transition posteriors.

        Parameters
        ----------
        Y : array-like
            Observed history, as for :meth:`score`.
        X : array-like of shape (n_obs, n_covariates), optional
            Covariates aligned with ``Y``.
        lengths : array-like of int, optional
            Length of each sequence; must sum to ``len(Y)``.

        Returns
        -------
        ndarray of shape (n_states, n_states)
            Row-stochastic mean transition matrix. A state with zero expected
            occupancy in the supplied history yields an all-zero row.
        """
        Y = np.asarray(Y)
        Xd = self._design_matrix(X, Y.shape[0])
        log_start = np.log(self.startprob_)
        xi_total = np.zeros((self.n_states, self.n_states))
        for start, end in iter_sequences(Y.shape[0], lengths):
            frameprob = self.emissions_.log_likelihood(Y[start:end])
            log_trans = self._per_sequence_logtrans(Xd, start, end)
            log_alpha, ll = _core.forward(log_start, log_trans, frameprob)
            log_beta = _core.backward(log_start, log_trans, frameprob)
            _, xi_sum, _ = _core.posteriors(
                log_alpha, log_beta, log_trans, frameprob, ll
            )
            if xi_sum is not None:  # None for single-observation sequences
                xi_total += xi_sum
        return normalize(xi_total, axis=1)

    def predict(self, Y, X=None, lengths=None):
        """Most likely hidden-state sequence (Viterbi decoding)."""
        Y = np.asarray(Y)
        Xd = self._design_matrix(X, Y.shape[0])
        log_start = np.log(self.startprob_)
        states = np.empty(Y.shape[0], dtype=int)
        for start, end in iter_sequences(Y.shape[0], lengths):
            frameprob = self.emissions_.log_likelihood(Y[start:end])
            log_trans = self._per_sequence_logtrans(Xd, start, end)
            seq, _ = _core.viterbi(log_start, log_trans, frameprob)
            states[start:end] = seq
        return states

    def decode(self, Y, X=None, lengths=None):
        """Return ``(log_prob, state_sequence)`` like :mod:`hmmlearn`."""
        Y = np.asarray(Y)
        Xd = self._design_matrix(X, Y.shape[0])
        log_start = np.log(self.startprob_)
        states = np.empty(Y.shape[0], dtype=int)
        total = 0.0
        for start, end in iter_sequences(Y.shape[0], lengths):
            frameprob = self.emissions_.log_likelihood(Y[start:end])
            log_trans = self._per_sequence_logtrans(Xd, start, end)
            seq, logp = _core.viterbi(log_start, log_trans, frameprob)
            states[start:end] = seq
            total += logp
        return total, states

    # -- generation --------------------------------------------------------
    def sample(self, n_samples, X=None, random_state=None):
        """Generate a sequence of observations of length ``n_samples``.

        Parameters
        ----------
        n_samples : int
        X : array-like of shape (n_samples, n_covariates), optional
            Covariate sequence driving the transitions. Required unless the
            model uses an intercept only.
        random_state : int, Generator or None

        Returns
        -------
        Y : ndarray of shape (n_samples, n_dim) or (n_samples,)
        states : ndarray of shape (n_samples,)
        """
        rng = check_random_state(
            random_state if random_state is not None else self.random_state
        )
        Xd = self._design_matrix(X, n_samples)
        log_A = self.transitions_.log_transition_matrices(Xd)
        A = np.exp(log_A)

        states = np.empty(n_samples, dtype=int)
        states[0] = rng.choice(self.n_states, p=self.startprob_)
        for t in range(1, n_samples):
            states[t] = rng.choice(self.n_states, p=A[t, states[t - 1]])

        observations = [self.emissions_.sample(int(s), rng) for s in states]
        Y = np.asarray(observations)
        return Y, states

    def simulate_forward(
        self,
        Y,
        X=None,
        X_future=None,
        n_paths=1000,
        quantiles=(0.05, 0.5, 0.95),
        random_state=None,
        *,
        duration_col=None,
        initial_duration=1,
        horizon=None,
    ):
        """Monte-Carlo forecast of future observation paths given history.

        Conditions on the observed history ``(Y, X)`` to obtain the filtered
        distribution of the current hidden state, then rolls the chain forward,
        drawing ``n_paths`` independent state-and-observation paths.

        Two covariate modes are supported:

        * **Exogenous** (default, ``duration_col=None``): the future covariates
          ``X_future`` are supplied explicitly -- the model cannot generate
          them. They must use the **same columns, order and scaling** as the
          ``X`` passed to :meth:`fit`.
        * **Dynamic sojourn-duration** (``duration_col`` set): one covariate
          column is the *duration* (number of consecutive steps spent in the
          current state). Because that is endogenous -- future durations depend
          on the simulated state path -- it is computed **online** at each step
          and written into column ``duration_col``. Any *other* (genuinely
          exogenous) covariates are still supplied via ``X_future``; if duration
          is the only covariate, omit ``X_future`` and pass ``horizon`` instead.
          The duration is fed as a raw integer count.

        Parameters
        ----------
        Y : array-like
            Observed history (a single sequence), shape ``(T,)`` or ``(T, n_dim)``.
        X : array-like of shape (T, n_covariates) or None
            Covariates aligned with ``Y``. In duration mode this must already
            contain the duration column (see :func:`nhmm.state_durations`).
            ``None`` for an intercept-only model (a plain HMM).
        X_future : array-like of shape (horizon, n_covariates), optional
            Future covariates. ``X_future[k]`` drives the transition into future
            step ``k``. In duration mode the values in column ``duration_col``
            are ignored (overwritten online); omit ``X_future`` entirely when
            duration is the only covariate. For an intercept-only model (a plain
            HMM) omit it too and pass ``horizon`` -- the constant transition
            matrix is used at every step. ``horizon`` is inferred from its
            length when given.
        n_paths : int, default=1000
            Number of Monte-Carlo paths to draw.
        quantiles : sequence of float, default=(0.05, 0.5, 0.95)
            Probability levels for the predictive bands.
        random_state : int, Generator or None
        duration_col : int, optional
            Index (within the raw covariate columns, before the intercept) of
            the sojourn-duration covariate. Enables dynamic mode.
        initial_duration : int or array-like of shape (n_paths,), default=1
            Run-length of the current state at the start of the forecast,
            applied per path. Typically ``state_durations(history_states)[-1]``.
        horizon : int, optional
            Forecast length. Required in duration mode when ``X_future`` is not
            given; otherwise inferred from ``X_future``.

        Returns
        -------
        ForwardSimulation
        """
        rng = check_random_state(
            random_state if random_state is not None else self.random_state
        )
        Y = np.asarray(Y)
        n_hist = Y.shape[0]

        # 1. Filtered distribution of the final observed state P(z_T | history).
        Xd = self._design_matrix(X, n_hist)
        frameprob = self.emissions_.log_likelihood(Y)
        log_start = np.log(self.startprob_)
        if n_hist > 1:
            log_trans = self.transitions_.log_transition_matrices(Xd)[1:]
        else:
            log_trans = np.empty((0, self.n_states, self.n_states))
        log_alpha, _ = _core.forward(log_start, log_trans, frameprob)
        filt = np.exp(log_normalize(log_alpha[-1]))

        dynamic = duration_col is not None
        n_cov = self.transitions_.n_features - (1 if self.fit_intercept else 0)

        # 2. Resolve the horizon and pre-compute transitions where possible.
        if dynamic:
            if not 0 <= duration_col < n_cov:
                raise ValueError(
                    f"duration_col must be in [0, {n_cov}), got {duration_col}"
                )
            if X_future is not None:
                base = np.atleast_2d(np.asarray(X_future, dtype=float))
                if base.shape[1] != n_cov:
                    raise ValueError(
                        f"X_future has {base.shape[1]} columns, expected {n_cov}"
                    )
                horizon = base.shape[0]
            else:
                if horizon is None:
                    raise ValueError(
                        "horizon is required in duration mode when X_future is None"
                    )
                if n_cov != 1:
                    raise ValueError(
                        "X_future is required: the model has exogenous covariates "
                        "besides the duration column"
                    )
                base = np.zeros((horizon, n_cov))
            durations = np.broadcast_to(
                np.asarray(initial_duration, dtype=float), (n_paths,)
            ).copy()
            covariate_paths = np.empty((n_paths, horizon))
        else:
            if X_future is None:
                if n_cov != 0:
                    raise ValueError(
                        "X_future is required when the model has exogenous covariates"
                    )
                if horizon is None:
                    raise ValueError(
                        "horizon is required when simulating without covariates"
                    )
                design = self._design_matrix(None, horizon)  # intercept-only
            else:
                X_future = np.atleast_2d(np.asarray(X_future, dtype=float))
                horizon = X_future.shape[0]
                design = self._design_matrix(X_future, horizon)
            A_future = self.transitions_.transition_matrices(design)
            covariate_paths = None
        if horizon < 1:
            raise ValueError("forecast horizon must be at least one step")

        # 3. Monte-Carlo roll-out, vectorised across paths.
        state_paths = np.empty((n_paths, horizon), dtype=int)
        z_prev = rng.choice(self.n_states, size=n_paths, p=filt)
        rows_idx = np.arange(n_paths)
        per_step_obs = []
        for k in range(horizon):
            if dynamic:
                rows = np.tile(base[k], (n_paths, 1))
                rows[:, duration_col] = durations
                A = self.transitions_.transition_matrices(
                    self._design_matrix(rows, n_paths)
                )
                probs = A[rows_idx, z_prev]  # (n_paths, K)
            else:
                probs = A_future[k, z_prev]  # (n_paths, K)
            u = rng.random(n_paths)[:, None]
            z = (np.cumsum(probs, axis=1) > u).argmax(axis=1)
            state_paths[:, k] = z
            per_step_obs.append(
                np.asarray([self.emissions_.sample(int(s), rng) for s in z])
            )
            if dynamic:
                covariate_paths[:, k] = durations
                durations = np.where(z == z_prev, durations + 1, 1)
            z_prev = z
        y_paths = np.stack(per_step_obs, axis=1)

        # 4. Summaries.
        levels = tuple(quantiles)
        mean = y_paths.mean(axis=0)
        quantile_bands = np.quantile(y_paths, levels, axis=0)
        state_probs = np.empty((horizon, self.n_states))
        for j in range(self.n_states):
            state_probs[:, j] = (state_paths == j).mean(axis=0)

        return ForwardSimulation(
            y_paths=y_paths,
            state_paths=state_paths,
            mean=mean,
            quantiles=quantile_bands,
            quantile_levels=levels,
            state_probs=state_probs,
            covariate_paths=covariate_paths,
        )
