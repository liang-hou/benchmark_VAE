"""This module is the implementation of the Wasserstein Autoencoder proposed in 
(https://arxiv.org/abs/1711.01558).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .otm_config import OTM_Config
from .otm_model import OTM

__all__ = ["OTM", "OTM_Config"]
