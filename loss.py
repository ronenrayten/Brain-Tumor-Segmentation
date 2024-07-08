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
    


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        # use standard CE loss without reducion as basis
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.CE = nn.CrossEntropyLoss(reduction='none',weight=self.weight)

    def forward(self, input, target):
        '''
       Compute the focal loss between `inputs` and `targets`.
        
        Parameters:
        - inputs: predictions for each example (a tensor of shape (batch_size, num_classes, depth, height, width)).
        - targets: ground truth labels for each example (a one-hot encoded tensor of shape (batch_size, num_classes, depth, height, width)).
        
        Returns:
        - loss: the computed focal loss.
        '''
        # Convert targets from one-hot encoding to class indices
        targets_indices = torch.argmax(target, dim=1)  # Shape: (batch_size, depth, height, width)

        # Flatten the input and target tensors to shape (batch_size * depth * height * width, num_classes) and (batch_size * depth * height * width), respectively
        inputs_flat = input.view(-1, input.size(1))
        targets_indices_flat = targets_indices.view(-1)

        # Compute the cross-entropy loss
        ce_loss = self.CE(inputs_flat, targets_indices_flat)

        # Compute the probabilities
        probs = torch.exp(-ce_loss)

        # Compute the focal loss
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self,num_classes=4,ignore_index=0,average="macro"):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-5
        
    def forward(self, inputs, targets):
        predict = F.log_softmax(inputs, dim=1).exp()

        intersection = torch.sum(predict * targets,(4,3,2))  # compute the intersection score per class
        union = torch.sum(predict, (4,3,2)) + torch.sum(targets,(4,3,2))  # compute the sum per class
        
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # dice score per class

        # this is actualy Macro menthod - calculating the ratio per class and averaging 
        dice_loss = 1 - torch.mean(dice_coef[0][:])  # take all classes

        return dice_loss
        

class CombinedDiceFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2,reduction='mean',weight=None):
        super(CombinedDiceFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma,reduction='mean',weight=weight)
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        combined_loss = focal_loss + dice_loss
        return combined_loss
