# direchilet_mle -  
# This function estimates concentration parameters of Dirichlet distribution, using maximum likelihood. 
# The calcualation based on paper "Estimating a Dirichlet distribution" (Thomas P. Minka, 2000, 2012)
# Reference link: http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/.
# Author: Adapted from opensource code on github https://github.com/ericsuh/dirichlet by David Yin
# Date Jul 28, 2020

import sys
import numpy as np
import scipy as sp
import scipy.stats as stats
from numpy import (
    arange,
    array,
    asanyarray,
    asarray,
    diag,
    exp,
    isscalar,
    log,
    ndarray,
    ones,
    vstack,
    zeros,
)
from numpy.linalg import norm
from scipy.special import gammaln, polygamma, psi

MAXINT = sys.maxsize

euler = -1 * psi(1)  # Euler-Mascheroni constant

def direchlet_mle(D, tol=1e-7, method="meanprecision", maxiter=None):
    """Iteratively computes maximum likelihood Dirichlet distribution
    for an observed data set, i.e. a for which log p(D|a) is maximum.
    Parameters
    ----------
    D : input direchlet distributed samples, (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    method : string
        One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
        which to find MLE Dirichlet distribution. Default is
        ``'meanprecision'``, which is faster.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    a : (K,) shape array
        Maximum likelihood parameters for Dirichlet distribution."""

    if method == "meanprecision":
        return _meanprecision(D, tol=tol, maxiter=maxiter)
    else:
        return _fixedpoint(D, tol=tol, maxiter=maxiter)

def _meanprecision(D, tol=1e-7, maxiter=None):
    """Mean/precision method for MLE of Dirichlet distribution
    Uses alternating estimations of mean and precision.
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    a : (K,) shape array
        Estimated parameters for Dirichlet distribution."""
    # remove observations with elements equal to 0
    D = D[np.all(D[:]>0, axis=1)]
    
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)
    s0 = a0.sum()
    if s0 < 0:
        a0 = a0 / s0
        s0 = 1
    elif s0 == 0:
        a0 = ones(a0.shape) / len(a0)
        s0 = 1
    m0 = a0 / s0

    # Start updating
    if maxiter is None:
        maxiter = MAXINT
    for i in range(maxiter):
        a1 = _fit_s(D, a0, logp, tol=tol, maxiter = maxiter)
        s1 = sum(a1)
        a1 = _fit_m(D, a1, logp, tol=tol, maxiter= maxiter)
        m = a1 / s1
        # Much faster convergence than with the more obvious condition
        # `norm(a1-a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, " f"values are {a1}."
    )

def _trigamma(x):
    return polygamma(1, x)


def _ipsi(y, tol=1.48e-9, maxiter=10):
    """Inverse of psi (digamma) using Newton's method. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).
    
    Parameters
    ----------
    y : (K,) shape array
        y-values of psi(x)
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 10.
    Returns
    -------
    (K,) shape array
        Approximate x for psi(x)."""
    y = asanyarray(y, dtype="float")
    x0 = np.piecewise(
        y,
        [y >= -2.22, y < -2.22],
        [(lambda x: exp(x) + 0.5), (lambda x: -1 / (x + euler))],
    )
    for i in range(maxiter):
        x1 = x0 - (psi(x0) - y) / _trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"value is {x1}")

def _fit_s(D, a0, logp, tol=1e-7, maxiter=1000):
    """Update parameters via MLE of precision with fixed mean
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a0 : (K,) shape array
        Current parameters for Dirichlet distribution
    logp : (K,) shape array
        Mean of log-transformed D across N observations
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 1000.
    Returns
    -------
    (K,) shape array
        Updated parameters for Dirichlet distribution."""
    s1 = a0.sum()
    m = a0 / s1
    mlogp = (m * logp).sum()
    for i in range(maxiter):
        s0 = s1
        g = psi(s1) - (m * psi(s1 * m)).sum() + mlogp
        h = _trigamma(s1) - ((m ** 2) * _trigamma(s1 * m)).sum()

        if g + s1 * h < 0:
            s1 = 1 / (1 / s0 + g / h / (s0 ** 2))
        if s1 <= 0:
            s1 = s0 * exp(-g / (s0 * h + g))  # Newton on log s
        if s1 <= 0:
            s1 = 1 / (1 / s0 + g / ((s0 ** 2) * h + 2 * s0 * g))  # Newton on 1/s
        if s1 <= 0:
            s1 = s0 - g / h  # Newton
        if s1 <= 0:
            raise NotConvergingError(f"Unable to update s from {s0}")

        a = s1 * m
        if abs(s1 - s0) < tol:
            return a

    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"s is {s1}")


def _fit_m(D, a0, logp, tol=1e-7, maxiter=1000):
    """Update parameters via MLE of mean with fixed precision s
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a0 : (K,) shape array
        Current parameters for Dirichlet distribution
    logp : (K,) shape array
        Mean of log-transformed D across N observations
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is 1000.
    Returns
    -------
    (K,) shape array
        Updated parameters for Dirichlet distribution."""
    s = a0.sum()
    for i in range(maxiter):
        m = a0 / s
        a1 = _ipsi(logp + (m * (psi(a0) - logp)).sum())
        a1 = a1 / a1.sum() * s

        if norm(a1 - a0) < tol:
            return a1
        a0 = a1

    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"s is {s}")

    
def loglikelihood(D, a):
    """Compute log likelihood of Dirichlet distribution, i.e. log p(D|a).
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    a : (K,) shape array
        Parameters for the Dirichlet distribution.
    Returns
    -------
    logl : float
        The log likelihood of the Dirichlet distribution"""
    N, K = D.shape
    logp = log(D).mean(axis=0)
    return N * (gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * logp).sum())

    

def _init_a(D):
    """Initial guess for Dirichlet alpha parameters given data D
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    Returns
    -------
    (K,) shape array
        Crude guess for parameters of Dirichlet distribution."""
    E = D.mean(axis=0)
    E2 = (D ** 2).mean(axis=0)
    return ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E

    
def _fixedpoint(D, tol=1e-7, maxiter=None):
    """Simple fixed point iteration method for MLE of Dirichlet distribution
    Parameters
    ----------
    D : (N, K) shape array
        ``N`` is the number of observations, ``K`` is the number of
        parameters for the Dirichlet distribution.
    tol : float
        If Euclidean distance between successive parameter arrays is less than
        ``tol``, calculation is taken to have converged.
    maxiter : int
        Maximum number of iterations to take calculations. Default is
        ``sys.maxint``.
    Returns
    -------
    a : (K,) shape array
        Fixed-point estimated parameters for Dirichlet distribution."""
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)

    # Start updating
    if maxiter is None:
        maxiter = MAXINT
    for i in range(maxiter):
        a1 = _ipsi(psi(a0.sum()) + logp)
        # Much faster convergence than with the more obvious condition
        # `norm(a1-a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        "Failed to converge after {} iterations, values are {}.".format(maxiter, a1)
    )