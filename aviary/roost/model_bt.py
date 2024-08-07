from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor, nn

from aviary.core import BaseModelClass
from aviary.roost.model import Roost
from aviary.networks import ResidualNetwork, SimpleNetwork
from aviary.segments import MessageLayer, WeightedAttentionPooling

if TYPE_CHECKING:
    from collections.abc import Sequence

class RoostWithBarlowTwins(Roost):
    def forward_with_masking(
        self,
        elem_weights: Tensor,
        elem_fea: Tensor,
        self_idx: LongTensor,
        nbr_idx: LongTensor,
        cry_elem_idx: LongTensor,
        mask_idx: int = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass with an optional masking of one node."""
        if mask_idx is not None:
            elem_fea_masked = elem_fea.clone()
            elem_fea_masked[mask_idx] = 0  # Mask the selected node
        else:
            elem_fea_masked = elem_fea

        # Compute the output for both masked and unmasked inputs
        output_unmasked = super().forward(elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx)
        output_masked = super().forward(elem_weights, elem_fea_masked, self_idx, nbr_idx, cry_elem_idx)

        return output_unmasked, output_masked

    def barlow_twins_loss(self, z_unmasked, z_masked, lambda_param=0.005):
        z_unmasked = (z_unmasked - z_unmasked.mean(0)) / z_unmasked.std(0)
        z_masked = (z_masked - z_masked.mean(0)) / z_masked.std(0)

        N, D = z_unmasked.size()
        c = torch.mm(z_unmasked.T, z_masked) / N

        c_diff = (c - torch.eye(D, device=c.device)).pow(2)
        on_diag = torch.diagonal(c_diff).sum()
        off_diag = (c_diff.sum() - on_diag)

        loss = on_diag + lambda_param * off_diag
        return loss

    def pretrain_step(self, batch, optimizer, lambda_param=0.005):
        elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx = batch
        mask_idx = torch.randint(0, elem_fea.size(0), (1,)).item()

        z_unmasked, z_masked = self.forward_with_masking(
            elem_weights, elem_fea, self_idx, nbr_idx, cry_elem_idx, mask_idx
        )

        loss = self.barlow_twins_loss(z_unmasked[0], z_masked[0], lambda_param)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
