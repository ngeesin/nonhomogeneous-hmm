"""nhmm: non-homogeneous hidden Markov models with covariate-driven transitions.

The transition probabilities between hidden states are modelled as a function of
time-varying external covariates through a softmax (multinomial-logistic) model,
generalising the classical homogeneous HMM.

Example
-------
>>> import numpy as np
>>> from nhmm import NonHomogeneousHMM
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((200, 1))          # one covariate
>>> Y = rng.standard_normal((200, 1))          # toy observations
>>> model = NonHomogeneousHMM(n_states=2, random_state=0).fit(Y, X)
>>> states = model.predict(Y, X)
"""

from .emissions import (
    BaseEmissions,
    CategoricalEmissions,
    GaussianEmissions,
)
from .model import ConvergenceMonitor, ForwardSimulation, NonHomogeneousHMM
from .transitions import SoftmaxTransitions

__all__ = [
    "NonHomogeneousHMM",
    "ForwardSimulation",
    "ConvergenceMonitor",
    "SoftmaxTransitions",
    "BaseEmissions",
    "GaussianEmissions",
    "CategoricalEmissions",
]

__version__ = "0.1.0"
