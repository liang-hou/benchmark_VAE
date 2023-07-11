import os
from typing import Optional

import scipy as sp
import torch
import torch.nn.functional as F
from pyexpat import model

from ...data.datasets import BaseDataset
from ..ae import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .otm_config import OTM_Config


class OTM(AE):
    """Hungarian Autoencoder model.

    Args:
        model_config (OTM_Config): The Autoencoder configuration setting the main parameters
            of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: OTM_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "OTM"

        self.update_steps = self.model_config.update_steps

        self.reconstruction_loss_scale = self.model_config.reconstruction_loss_scale

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        idx = inputs["index"]

        z = self.encoder(idx).embedding.detach()
        recon_x = self.decoder(z)["reconstruction"]

        epoch = kwargs.get('epoch', None)
        if epoch is not None and epoch % self.update_steps == 0:
            loss, cost = self.loss_function(recon_x, x, z, idx=idx)
        else:
            loss, cost = self.loss_function(recon_x, x, z)

        output = ModelOutput(
            loss=loss, recon_x=recon_x, z=z, cost=cost
        )

        return output

    def loss_function(self, recon_x, x, z, idx=None):

        if idx is not None:
            cost = (torch.cdist(x.reshape(x.shape[0], -1), recon_x.reshape(x.shape[0], -1), p=2) ** 2).detach().cpu().numpy()
            row_ind, col_ind = sp.optimize.linear_sum_assignment(cost)  # Hungarian algorithm for assignment

            assign_loss = cost[row_ind, col_ind].mean()

            recon_x = recon_x[col_ind]
            self.assign_latents(idx, col_ind)
        else:
            assign_loss = 0

        recon_loss = self.reconstruction_loss_scale * F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        return (
            recon_loss.mean(dim=0), assign_loss
        )

    def assign_latents(self, batch_index, assign_index):
        self.encoder.assign_latents(batch_index, assign_index)
