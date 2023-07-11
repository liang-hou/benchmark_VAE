import torch
import torch.nn as nn

from ...base import BaseAEConfig
from ...base.base_utils import ModelOutput
from ..base_architectures import BaseEncoder


class Encoder_NP(BaseEncoder):
    
    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.num_samples = args.num_samples
        self.latent_dim = args.latent_dim

        self.embedding = nn.Embedding(self.num_samples, args.latent_dim)

    def forward(self, idx: torch.Tensor):
        output = ModelOutput()

        output["embedding"] = self.embedding(idx)

        return output

    def assign_latents(self, batch_index, assign_index):
        self.embedding.weight.data[batch_index] = self.embedding.weight.data[batch_index][assign_index]
