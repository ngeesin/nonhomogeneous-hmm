import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from nhmm.emissions import CategoricalEmissions, GaussianEmissions


def test_gaussian_diag_loglik_matches_scipy():
    em = GaussianEmissions(2, 2, covariance_type="diag")
    em.means_ = np.array([[0.0, 0.0], [1.0, -1.0]])
    em.covars_ = np.array([[1.0, 2.0], [0.5, 1.5]])
    Y = np.array([[0.2, -0.3], [1.1, -0.8]])

    ll = em.log_likelihood(Y)
    for k in range(2):
        expected = norm.logpdf(Y, em.means_[k], np.sqrt(em.covars_[k])).sum(axis=1)
        assert np.allclose(ll[:, k], expected)


def test_gaussian_full_loglik_matches_scipy():
    em = GaussianEmissions(1, 2, covariance_type="full")
    em.means_ = np.array([[0.5, -0.5]])
    cov = np.array([[1.0, 0.3], [0.3, 0.8]])
    em.covars_ = cov[None]
    Y = np.array([[0.1, 0.2], [-0.4, 0.6]])
    ll = em.log_likelihood(Y)
    expected = multivariate_normal(em.means_[0], cov).logpdf(Y)
    assert np.allclose(ll[:, 0], expected)


def test_gaussian_mstep_recovers_parameters():
    rng = np.random.default_rng(0)
    # Pure responsibilities (hard assignment) -> should recover sample stats.
    Y = np.vstack([rng.normal(5.0, 1.0, (500, 1)), rng.normal(-2.0, 0.5, (500, 1))])
    gamma = np.zeros((1000, 2))
    gamma[:500, 0] = 1.0
    gamma[500:, 1] = 1.0

    em = GaussianEmissions(2, 1, covariance_type="diag", min_covar=0.0)
    em.m_step(Y, gamma)
    assert em.means_[0, 0] == pytest.approx(Y[:500].mean(), abs=1e-6)
    assert em.means_[1, 0] == pytest.approx(Y[500:].mean(), abs=1e-6)


def test_categorical_mstep_and_loglik():
    em = CategoricalEmissions(2, 3)
    Y = np.array([0, 1, 2, 0, 1])
    gamma = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    )
    em.m_step(Y, gamma)
    assert np.allclose(em.emissionprob_.sum(axis=1), 1.0)

    ll = em.log_likelihood(Y)
    assert ll.shape == (5, 2)
    assert np.allclose(ll, np.log(em.emissionprob_[:, Y].T))
