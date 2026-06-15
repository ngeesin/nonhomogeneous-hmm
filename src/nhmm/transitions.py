"""Covariate-dependent (non-homogeneous) transition model.

The probability of moving from state ``i`` to state ``j`` at time ``t`` is

.. math::

    P(z_t = j \\mid z_{t-1} = i, \\mathbf{x}_t)
        = \\frac{\\exp(\\mathbf{w}_{ij}^\\top \\mathbf{x}_t)}
               {\\sum_k \\exp(\\mathbf{w}_{ik}^\\top \\mathbf{x}_t)},

i.e. one multinomial-logistic (softmax) model per origin state ``i``.  With an
intercept-only covariate this reduces exactly to a classical homogeneous HMM,
so the homogeneous case is a strict special case of this class.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._utils import check_random_state, log_softmax, softmax


class SoftmaxTransitions:
    """Multinomial-logistic transition model.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_features : int
        Dimensionality of the covariate vector (including the intercept column
        if one is used by the caller).
    reg : float, default=1e-4
        L2 regularisation strength applied to the weights during the M-step.
        Keeps the over-parameterised softmax identifiable and well behaved.

    Attributes
    ----------
    weights : ndarray of shape (n_states, n_features, n_states)
        ``weights[i, :, j]`` is :math:`\\mathbf{w}_{ij}`.
    """

    def __init__(self, n_states: int, n_features: int, reg: float = 1e-4):
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.reg = float(reg)
        self.weights = np.zeros((self.n_states, self.n_features, self.n_states))

    # -- initialisation ----------------------------------------------------
    def init_params(self, random_state=None, scale: float = 0.01) -> "SoftmaxTransitions":
        """Randomly initialise the weights with small values."""
        rng = check_random_state(random_state)
        self.weights = scale * rng.standard_normal(
            (self.n_states, self.n_features, self.n_states)
        )
        return self

    # -- evaluation --------------------------------------------------------
    def log_transition_matrices(self, X: np.ndarray) -> np.ndarray:
        """Return per-time log transition matrices.

        Parameters
        ----------
        X : ndarray of shape (n_obs, n_features)
            Covariates. ``X[t]`` drives the transition *into* time ``t``.

        Returns
        -------
        log_A : ndarray of shape (n_obs, n_states, n_states)
            ``log_A[t, i, j] = log P(z_t=j | z_{t-1}=i, X[t])``. Row ``t=0`` is
            produced for convenience but is unused by the HMM (there is no
            transition into the first observation).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"X must have shape (n_obs, {self.n_features}), got {X.shape}"
            )
        # scores[t, i, j] = X[t] . weights[i, :, j]
        scores = np.einsum("tf,ifj->tij", X, self.weights)
        return log_softmax(scores, axis=2)

    def transition_matrices(self, X: np.ndarray) -> np.ndarray:
        """Probability version of :meth:`log_transition_matrices`."""
        return np.exp(self.log_transition_matrices(X))

    # -- M-step ------------------------------------------------------------
    def m_step(self, X: np.ndarray, xi: np.ndarray, max_iter: int = 100) -> None:
        """Update the weights from expected transition counts.

        Parameters
        ----------
        X : ndarray of shape (n_drivers, n_features)
            Stacked covariates driving each modelled transition, i.e. ``X[t+1]``
            for every transition ``t -> t+1`` across all sequences.
        xi : ndarray of shape (n_drivers, n_states, n_states)
            Matching expected transition posteriors.
        max_iter : int
            Maximum number of L-BFGS iterations per origin state.
        """
        X = np.asarray(X, dtype=float)
        xi = np.asarray(xi, dtype=float)
        if X.shape[0] != xi.shape[0]:
            raise ValueError("X and xi must have the same number of rows")

        for i in range(self.n_states):
            self.weights[i] = self._fit_one_state(X, xi[:, i, :], self.weights[i], max_iter)

    def _fit_one_state(self, X, target, w0, max_iter):
        n_features, n_states = w0.shape
        # Weight of each observation = expected number of transitions out of i.
        row_weight = target.sum(axis=1)  # (n_drivers,)
        reg = self.reg

        def neg_ll_and_grad(flat):
            W = flat.reshape(n_features, n_states)
            scores = X @ W  # (n, K)
            logp = log_softmax(scores, axis=1)
            nll = -np.sum(target * logp) + reg * np.sum(W * W)
            probs = softmax(scores, axis=1)
            # grad of -ll wrt scores: row_weight * probs - target
            grad_scores = probs * row_weight[:, None] - target
            grad = X.T @ grad_scores + 2.0 * reg * W
            return nll, grad.ravel()

        res = minimize(
            neg_ll_and_grad,
            w0.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        return res.x.reshape(n_features, n_states)
