"""Emission (observation) models for the hidden states.

Every emission model implements the same small interface so that
:class:`~nhmm.model.NonHomogeneousHMM` is agnostic to the observation type:

``init_params(Y, random_state)``
    Initialise parameters from data.
``log_likelihood(Y) -> (n_obs, n_states)``
    Log probability of each observation under each state.
``m_step(Y, gamma)``
    Re-estimate parameters from posterior responsibilities.
``sample(state, random_state) -> observation``
    Draw a single observation from ``state``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ._utils import check_random_state, normalize


class BaseEmissions(ABC):
    n_states: int

    @abstractmethod
    def init_params(self, Y, random_state=None): ...

    @abstractmethod
    def log_likelihood(self, Y) -> np.ndarray: ...

    @abstractmethod
    def m_step(self, Y, gamma) -> None: ...

    @abstractmethod
    def sample(self, state: int, random_state=None): ...


class GaussianEmissions(BaseEmissions):
    """(Multivariate) Gaussian emissions, one per state.

    Parameters
    ----------
    n_states : int
    n_dim : int
        Dimensionality of each observation.
    covariance_type : {"diag", "full"}, default="diag"
    min_covar : float, default=1e-3
        Floor added to the covariance diagonal for numerical stability.
    """

    def __init__(self, n_states, n_dim, covariance_type="diag", min_covar=1e-3):
        if covariance_type not in ("diag", "full"):
            raise ValueError("covariance_type must be 'diag' or 'full'")
        self.n_states = int(n_states)
        self.n_dim = int(n_dim)
        self.covariance_type = covariance_type
        self.min_covar = float(min_covar)
        self.means_ = np.zeros((self.n_states, self.n_dim))
        if covariance_type == "diag":
            self.covars_ = np.ones((self.n_states, self.n_dim))
        else:
            self.covars_ = np.array([np.eye(self.n_dim) for _ in range(self.n_states)])

    @staticmethod
    def _as_2d(Y):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        return Y

    def init_params(self, Y, random_state=None):
        rng = check_random_state(random_state)
        Y = self._as_2d(Y)
        n_obs = Y.shape[0]
        # Spread initial means over randomly chosen observations.
        idx = rng.choice(n_obs, size=self.n_states, replace=n_obs < self.n_states)
        self.means_ = Y[idx].copy()
        global_var = Y.var(axis=0) + self.min_covar
        if self.covariance_type == "diag":
            self.covars_ = np.tile(global_var, (self.n_states, 1))
        else:
            base = np.diag(global_var)
            self.covars_ = np.array([base.copy() for _ in range(self.n_states)])
        return self

    def log_likelihood(self, Y):
        Y = self._as_2d(Y)
        n_obs = Y.shape[0]
        ll = np.empty((n_obs, self.n_states))
        for k in range(self.n_states):
            ll[:, k] = self._logpdf_state(Y, k)
        return ll

    def _logpdf_state(self, Y, k):
        diff = Y - self.means_[k]
        d = self.n_dim
        if self.covariance_type == "diag":
            var = self.covars_[k]
            log_det = np.sum(np.log(var))
            maha = np.sum((diff * diff) / var, axis=1)
        else:
            cov = self.covars_[k]
            sign, log_det = np.linalg.slogdet(cov)
            solved = np.linalg.solve(cov, diff.T).T
            maha = np.einsum("ij,ij->i", diff, solved)
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)

    def m_step(self, Y, gamma):
        Y = self._as_2d(Y)
        weight = gamma.sum(axis=0) + 1e-12  # (n_states,)
        self.means_ = (gamma.T @ Y) / weight[:, None]
        if self.covariance_type == "diag":
            covars = np.empty((self.n_states, self.n_dim))
            for k in range(self.n_states):
                diff = Y - self.means_[k]
                covars[k] = (gamma[:, k] @ (diff * diff)) / weight[k]
            self.covars_ = covars + self.min_covar
        else:
            covars = np.empty((self.n_states, self.n_dim, self.n_dim))
            eye = np.eye(self.n_dim) * self.min_covar
            for k in range(self.n_states):
                diff = Y - self.means_[k]
                weighted = diff * gamma[:, k][:, None]
                covars[k] = (weighted.T @ diff) / weight[k] + eye
            self.covars_ = covars

    def sample(self, state, random_state=None):
        rng = check_random_state(random_state)
        if self.covariance_type == "diag":
            return rng.normal(self.means_[state], np.sqrt(self.covars_[state]))
        return rng.multivariate_normal(self.means_[state], self.covars_[state])


class CategoricalEmissions(BaseEmissions):
    """Categorical (multinoulli) emissions over a finite alphabet.

    Observations are integer symbol codes in ``[0, n_symbols)``.
    """

    def __init__(self, n_states, n_symbols):
        self.n_states = int(n_states)
        self.n_symbols = int(n_symbols)
        self.emissionprob_ = np.full((self.n_states, self.n_symbols), 1.0 / self.n_symbols)

    @staticmethod
    def _as_codes(Y):
        return np.asarray(Y).astype(int).ravel()

    def init_params(self, Y, random_state=None):
        rng = check_random_state(random_state)
        probs = rng.random((self.n_states, self.n_symbols)) + 1.0
        self.emissionprob_ = normalize(probs, axis=1)
        return self

    def log_likelihood(self, Y):
        codes = self._as_codes(Y)
        if codes.min() < 0 or codes.max() >= self.n_symbols:
            raise ValueError("categorical observations out of range")
        with np.errstate(divide="ignore"):
            log_e = np.log(self.emissionprob_)
        return log_e[:, codes].T

    def m_step(self, Y, gamma):
        codes = self._as_codes(Y)
        counts = np.zeros((self.n_states, self.n_symbols))
        np.add.at(counts.T, codes, gamma)
        self.emissionprob_ = normalize(counts + 1e-12, axis=1)

    def sample(self, state, random_state=None):
        rng = check_random_state(random_state)
        return int(rng.choice(self.n_symbols, p=self.emissionprob_[state]))
