from __future__ import annotations

from typing import Optional

import torch
from torch import long, nn
from torchmetrics import Dice

from kornia.core import Tensor, tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.utils.one_hot import one_hot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, Sequence
from torch import Tensor



        

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
    




# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py


def focal_loss(
    pred: Tensor,
    target: Tensor,
    alpha: Optional[float],
    gamma: float = 2.0,
    reduction: str = "none",
    weight: Optional[Tensor] = None,
) -> Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        pred: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Return:
        the computed loss.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> output = focal_loss(pred, target, **kwargs)
        >>> output.backward()
    """

    KORNIA_CHECK_SHAPE(pred, ["B", "C", "*"])
    out_size = (pred.shape[0],) + pred.shape[2:]
    target = target.argmax(dim=1)
    KORNIA_CHECK(
        (pred.shape[0] == target.shape[0] and target.shape[1:] == pred.shape[2:]),
        f"Expected target size {out_size}, got {target.shape}",
    )
    KORNIA_CHECK(
        pred.device == target.device,
        f"pred and target must be in the same device. Got: {pred.device} and {target.device}",
    )

    # create the labels one hot tensor
    
    target_one_hot: Tensor = one_hot(target, num_classes=pred.shape[1], device=pred.device, dtype=pred.dtype)

    # compute softmax over the classes axis
    log_pred_soft: Tensor = pred.log_softmax(1)

    # compute the actual focal loss
    loss_tmp: Tensor = -torch.pow(1.0 - log_pred_soft.exp(), gamma) * log_pred_soft * target_one_hot

    num_of_classes = pred.shape[1]
    broadcast_dims = [-1] + [1] * len(pred.shape[2:])
    if alpha is not None:
        alpha_fac = tensor([1 - alpha] + [alpha] * (num_of_classes - 1), dtype=loss_tmp.dtype, device=loss_tmp.device)
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss_tmp = alpha_fac * loss_tmp

    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f"weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}",
        )
        KORNIA_CHECK(
            weight.device == pred.device,
            f"weight and pred must be in the same device. Got: {weight.device} and {pred.device}",
        )

        weight = weight.view(broadcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Shape:
        - Pred: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha: Optional[float], gamma: float = 2.0, reduction: str = "none", weight: Optional[Tensor] = None
    ) -> None:
        super().__init__()
        self.alpha: Optional[float] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.weight: Optional[Tensor] = weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return focal_loss(pred, target, self.alpha, self.gamma, self.reduction, self.weight)


class DiceLoss(nn.Module):
    def __init__(self,num_classes=4,ignore_index=0,average="macro"):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5
        
    def forward(self, inputs, targets):
        predict = F.softmax(inputs, dim=1)

        intersection = torch.sum(predict * targets,(4,3,2))  # compute the intersection score per class
        union = torch.sum(predict, (4,3,2)) + torch.sum(targets,(4,3,2))  # compute the sum per class
        
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # dice score per class

        # this is actualy Macro menthod - calculating the ratio per class and averaging 
        dice_loss = 1 - torch.mean(dice_coef[0][-3:])  # take only classes 1,2,3

        return dice_loss
        
