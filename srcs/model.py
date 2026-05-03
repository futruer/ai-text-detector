"""
Linear + CRF model for token-level binary classification.
"""

import torch
import torch.nn as nn
from torchcrf import CRF
from typing import Union, List


class LinearCRF(nn.Module):
    """Linear layer followed by CRF for sequence labeling.

    Architecture:
        Input (512, 768) -> Linear(768, 2) -> CRF(num_tags=2)

    Label scheme:
        0 = Human
        1 = AI/Machine
        -100 = Ignore (padding, [CLS], [SEP])
    """

    def __init__(self, input_dim: int = 768, num_tags: int = 2):
        super().__init__()

        # Linear projection: 768 -> 2
        self.fc = nn.Linear(input_dim, num_tags)

        # Xavier initialization for linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        features: torch.Tensor,
        tags: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> Union[torch.Tensor, List[List[int]]]:
        """Forward pass.

        Training mode (tags is not None):
            Returns: Negative log likelihood loss (scalar)

        Inference mode (tags is None):
            Returns: List of decoded tag sequences (Viterbi)
        """
        emissions = self.fc(features)  # [batch_size, seq_len, 2]

        if tags is not None:
            # CRF doesn't support -100 labels, convert to valid tags
            tags_for_crf = tags.clone()
            tags_for_crf[tags == -100] = 0

            # CRF expects mask as byte tensor
            mask_byte = mask.byte() if mask is not None else None
            if mask_byte is not None:
                mask_byte = mask_byte.clone()
                # Set mask to 0 where labels are -100
                mask_byte[tags == -100] = 0
                # CRF requires first timestep mask to be 1 (do this after -100 masking)
                mask_byte[:, 0] = 1

            loss = -self.crf(
                emissions,
                tags_for_crf,
                mask=mask_byte,
                reduction='mean'
            )
            return loss
        else:
            # Inference: return Viterbi decoded sequences
            mask_byte = mask.byte() if mask is not None else None
            if mask_byte is not None:
                mask_byte = mask_byte.clone()
                # CRF requires first timestep mask to be 1
                mask_byte[:, 0] = 1
            decoded = self.crf.decode(emissions, mask=mask_byte)
            return decoded