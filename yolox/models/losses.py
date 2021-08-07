#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2019-2021 Intflow Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# Focal loss implementation inspired by
# https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
# https://github.com/doiken23/pytorch_toolbox/blob/master/focalloss2d.py
class MultiClassBCELoss(nn.Module):
    def __init__(self,
                 use_focal_weights=True,
                 focus_param=2,
                 balance_param=0.25,
                 reduction='none'
                 ):
        super().__init__()

        #self.use_weight_mask = use_weight_mask
        self.nll_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.use_focal_weights = use_focal_weights
        self.focus_param = focus_param
        self.balance_param = balance_param
        
    def forward(self,
                outputs,
                targets):
        # inputs and targets are assumed to be BatchxClasses
        assert len(outputs.shape) == len(targets.shape)
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(1) == targets.size(1)
        
        ## weights are assumed to be BatchxClasses
        #assert outputs.size(0) == weights.size(0)
        #assert outputs.size(1) == weights.size(1)

        #if self.use_weight_mask:
        #    bce_loss = F.binary_cross_entropy_with_logits(input=outputs,
        #                                                  target=targets)            
        #else:
        bce_loss = self.nll_loss(input=outputs,
                                 target=targets)
        
        if self.use_focal_weights:
            logpt = - bce_loss
            pt    = torch.exp(logpt)

            focal_loss = -((1 - pt) ** self.focus_param) * logpt
            balanced_focal_loss = self.balance_param * focal_loss
            
            return balanced_focal_loss
        else:
            return bce_loss 