# -*- coding: utf-8 -*-

from typing import List, Union

import torch
import torch.nn as nn


def to_one_hot(x: torch.Tensor, n_classes: int = -1):
    """Converts the label into one-hot encoded ground truth.
    It uses PyTorch's one_hot() function results and converts it
    from (channel-last) into (channel-first) order"""
    output_type = x.dtype
    output_shape = x.shape

    remove_extra_dim = False
    if len(output_shape) < 3:
        remove_extra_dim = True
        x = torch.unsqueeze(x, dim=0)

    # First we perform the one-hot encoding
    # results in shape (batch, x, y, c)
    tmp = torch.nn.functional.one_hot(x.type(torch.int64), num_classes=n_classes)

    # Now we move the last axis to the second position
    # results in shape (batch, c, x, y)
    tmp = torch.moveaxis(tmp, -1, 1).type(output_type)

    if remove_extra_dim:
        tmp = torch.squeeze(tmp, dim=0)

    return tmp


class DiceLoss(nn.Module):
    def __init__(
        self,
        classes: Union[int, List[int]],
        smoothing_factor: float = 1.0,
    ):
        """
        Calculates the dice-sorensen score on a given input wrt a target

        :param classes: If an int: the amount of classes,
                        if a list of ints: the class indices to calculate a dice for
        :param smoothing_factor: A value that we add to either side of the division
                                 to ensure that we don't divide by zero, or reach a
                                 loss of 0.0, which would cause a loss of gradients
        """
        super(DiceLoss, self).__init__()
        if isinstance(classes, list):
            self.classes = classes
        elif isinstance(classes, int):
            self.classes = list(range(classes))
        else:
            raise RuntimeError(
                f"DiceLoss.classes has to be either an integer, or list of integers.\n{self.classses}"
            )

        self.smoothing_factor = smoothing_factor

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if len(self.classes) > y_hat.shape[1]:
            raise RuntimeError(
                f"The amount of classes ({len(self.classes)}) to calculate is larger than "
                f"the amount of classes in the prediction {y_hat.shape[1]}"
            )

        scores = torch.zeros(y.shape[0]).to(y)
        for c in self.classes:
            scores += self._calculate_single_class(y_hat[:, c, ...], y[:, c, ...])

        return torch.mean(torch.div(scores, len(self.classes)))

    def _calculate_single_class(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        pred_flat = torch.flatten(y_hat, start_dim=1)
        targ_flat = torch.flatten(y, start_dim=1)

        intersection = torch.sum(pred_flat * targ_flat, dim=-1)
        divider = torch.sum(pred_flat, dim=-1) + torch.sum(targ_flat, dim=-1)

        dice = 2.0 * intersection + self.smoothing_factor
        dice /= divider + self.smoothing_factor

        return torch.sub(1.0, dice)

