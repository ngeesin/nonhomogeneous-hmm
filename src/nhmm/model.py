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
    """

    y_paths: np.ndarray
    state_paths: np.ndarray
    mean: np.ndarray
    quantiles: np.ndarray
    quantile_levels: tuple
    state_probs: np.ndarray


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
        X,
        X_future,
        n_paths=1000,
        quantiles=(0.05, 0.5, 0.95),
        random_state=None,
    ):
        """Monte-Carlo forecast of future observation paths given history.

        Conditions on the observed history ``(Y, X)`` to obtain the filtered
        distribution of the current hidden state, then rolls the chain forward
        over the horizon defined by ``X_future``, drawing ``n_paths``
        independent state-and-observation paths.

        Because the model treats covariates as *exogenous*, the future
        covariates ``X_future`` must be supplied explicitly -- the model cannot
        generate them. They must use the **same columns, order and scaling** as
        the ``X`` passed to :meth:`fit` (e.g. apply the same standardisation).

        Parameters
        ----------
        Y : array-like
            Observed history (a single sequence), shape ``(T,)`` or ``(T, n_dim)``.
        X : array-like of shape (T, n_covariates) or None
            Covariates aligned with ``Y``. ``None`` for an intercept-only model.
        X_future : array-like of shape (horizon, n_covariates)
            Covariates for each future step. ``X_future[k]`` drives the
            transition into future step ``k`` (matching the training
            convention). ``horizon`` is inferred from its length. Pass an array
            of shape ``(horizon, 0)`` (or ``None`` is *not* accepted here) only
            for intercept-only models; otherwise provide the covariate columns.
        n_paths : int, default=1000
            Number of Monte-Carlo paths to draw.
        quantiles : sequence of float, default=(0.05, 0.5, 0.95)
            Probability levels for the predictive bands.
        random_state : int, Generator or None

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

        # 2. Future transition matrices; X_future[k] drives the move into step k.
        X_future = np.atleast_2d(np.asarray(X_future, dtype=float))
        horizon = X_future.shape[0]
        if horizon < 1:
            raise ValueError("X_future must contain at least one future step")
        Xdf = self._design_matrix(X_future, horizon)
        A_future = self.transitions_.transition_matrices(Xdf)  # (horizon, K, K)

        # 3. Monte-Carlo roll-out, vectorised across paths.
        state_paths = np.empty((n_paths, horizon), dtype=int)
        z_prev = rng.choice(self.n_states, size=n_paths, p=filt)
        per_step_obs = []
        for k in range(horizon):
            probs = A_future[k, z_prev]  # (n_paths, K)
            u = rng.random(n_paths)[:, None]
            z = (np.cumsum(probs, axis=1) > u).argmax(axis=1)
            state_paths[:, k] = z
            per_step_obs.append(
                np.asarray([self.emissions_.sample(int(s), rng) for s in z])
            )
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
        )
