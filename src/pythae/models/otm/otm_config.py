from dataclasses import field
from typing import List, Union

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..ae import AEConfig


@dataclass
class OTM_Config(AEConfig):
    """Optimal transport model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        num_samples: (int): The number of training samples.
        reconstruction_loss_scale (float): Parameter scaling the reconstruction loss. Default: 1
    """

    num_samples: int = 1
    reconstruction_loss_scale: float = 1.0
    update_steps: int = 1
