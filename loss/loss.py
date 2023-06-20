import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coef(target, pred):
    eps = 0.0001
    y_pred_f = pred.flatten(2)
    y_true_f = target.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def focal_dice_loss(pred, target, focal_weight=0.5):
    focal = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    return loss

def focal_loss(pred, target, alpha=.25, gamma=2) : 
    bce = F.binary_cross_entropy_with_logits(pred, target)
    bce_exp = torch.exp(-bce)
    loss = alpha * (1-bce_exp)**gamma * bce
    return loss
