"""Low-level inference routines for (non-homogeneous) hidden Markov models.

All routines operate in log space and accept *time-varying* transition
matrices, which is what makes the model non-homogeneous: ``log_A`` has one
``(n_states, n_states)`` matrix per transition rather than a single shared one.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _check_shapes(log_startprob, log_transmat, log_frameprob):
    n_obs, n_states = log_frameprob.shape
    if log_startprob.shape != (n_states,):
        raise ValueError(
            f"log_startprob has shape {log_startprob.shape}, expected {(n_states,)}"
        )
    if n_obs > 1 and log_transmat.shape != (n_obs - 1, n_states, n_states):
        raise ValueError(
            f"log_transmat has shape {log_transmat.shape}, "
            f"expected {(n_obs - 1, n_states, n_states)}"
        )
    return n_obs, n_states


def forward(log_startprob, log_transmat, log_frameprob):
    """Run the forward pass.

    Parameters
    ----------
    log_startprob : ndarray of shape (n_states,)
        Log initial-state distribution.
    log_transmat : ndarray of shape (n_obs - 1, n_states, n_states)
        ``log_transmat[t, i, j]`` is ``log P(z_{t+1}=j | z_t=i)``.
    log_frameprob : ndarray of shape (n_obs, n_states)
        ``log_frameprob[t, i]`` is the log emission likelihood ``log P(x_t | z_t=i)``.

    Returns
    -------
    log_alpha : ndarray of shape (n_obs, n_states)
    log_likelihood : float
    """
    n_obs, n_states = _check_shapes(log_startprob, log_transmat, log_frameprob)
    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_startprob + log_frameprob[0]
    for t in range(1, n_obs):
        # work = log_alpha[t-1, i] + log_transmat[t-1, i, j]
        work = log_alpha[t - 1][:, None] + log_transmat[t - 1]
        log_alpha[t] = logsumexp(work, axis=0) + log_frameprob[t]
    return log_alpha, logsumexp(log_alpha[-1])


def backward(log_startprob, log_transmat, log_frameprob):
    """Run the backward pass, returning ``log_beta`` of shape (n_obs, n_states)."""
    n_obs, n_states = _check_shapes(log_startprob, log_transmat, log_frameprob)
    log_beta = np.zeros((n_obs, n_states))
    for t in range(n_obs - 2, -1, -1):
        # work = log_transmat[t, i, j] + log_frameprob[t+1, j] + log_beta[t+1, j]
        work = log_transmat[t] + (log_frameprob[t + 1] + log_beta[t + 1])[None, :]
        log_beta[t] = logsumexp(work, axis=1)
    return log_beta


def posteriors(log_alpha, log_beta, log_transmat, log_frameprob, log_likelihood):
    """Compute smoothed state and transition posteriors.

    Returns
    -------
    gamma : ndarray of shape (n_obs, n_states)
        ``gamma[t, i] = P(z_t=i | x_{1:T})``.
    xi_sum : ndarray of shape (n_states, n_states) or None
        ``sum_t P(z_t=i, z_{t+1}=j | x_{1:T})`` aggregated over time. ``None``
        when the sequence has a single observation.
    xi : ndarray of shape (n_obs - 1, n_states, n_states) or None
        Per-step transition posteriors (needed for the non-homogeneous M-step).
    """
    log_gamma = log_alpha + log_beta - log_likelihood
    gamma = np.exp(log_gamma)
    # Guard against tiny numerical drift.
    gamma /= gamma.sum(axis=1, keepdims=True)

    n_obs = log_alpha.shape[0]
    if n_obs < 2:
        return gamma, None, None

    log_xi = (
        log_alpha[:-1, :, None]
        + log_transmat
        + (log_frameprob[1:] + log_beta[1:])[:, None, :]
        - log_likelihood
    )
    xi = np.exp(log_xi)
    return gamma, xi.sum(axis=0), xi


def viterbi(log_startprob, log_transmat, log_frameprob):
    """Maximum a posteriori state sequence via the Viterbi algorithm.

    Returns ``(state_sequence, log_prob_of_path)``.
    """
    n_obs, n_states = _check_shapes(log_startprob, log_transmat, log_frameprob)
    delta = np.empty((n_obs, n_states))
    backptr = np.zeros((n_obs, n_states), dtype=int)

    delta[0] = log_startprob + log_frameprob[0]
    for t in range(1, n_obs):
        work = delta[t - 1][:, None] + log_transmat[t - 1]
        backptr[t] = np.argmax(work, axis=0)
        delta[t] = work[backptr[t], np.arange(n_states)] + log_frameprob[t]

    states = np.empty(n_obs, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    log_prob = float(delta[-1, states[-1]])
    for t in range(n_obs - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states, log_prob
