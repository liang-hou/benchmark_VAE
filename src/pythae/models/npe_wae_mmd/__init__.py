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

from .npe_wae_mmd_config import NPE_WAE_MMD_Config
from .npe_wae_mmd_model import NPE_WAE_MMD

__all__ = ["NPE_WAE_MMD", "NPE_WAE_MMD_Config"]
