"""
MLoRE Loss Functions for MPCompress Framework

This module provides multi-task loss functions for training MLoRE models,
including task-specific losses and compression losses (bpp, mse).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# RFC code has been migrated to mpcompress, no need for external RFC dependency


__all__ = [
    "MultiTaskLoss",
    "MLoRECodingLoss",
    "CrossEntropyLoss",
    "BalancedBinaryCrossEntropyLoss",
    "L1Loss",
    "SiLogLoss",
    "get_task_loss",
]


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with ignore regions.
    
    Args:
        ignore_index: Label index to ignore (default: 255)
        class_weight: Optional class weights for balancing
        balanced: Whether to use balanced weighting
    """
    
    def __init__(self, ignore_index=255, class_weight=None, balanced=False):
        super().__init__()
        self.ignore_index = ignore_index
        if balanced:
            assert class_weight is None
        self.balanced = balanced
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None
    
    def forward(self, out, label, reduction='mean'):
        if len(label.shape) > 1:
            label = torch.squeeze(label, dim=1).long()
        
        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w_pos = num_labels_neg / num_total
            class_weight = torch.stack((1. - w_pos, w_pos), dim=0)
            loss = F.cross_entropy(
                out, label, weight=class_weight, 
                ignore_index=self.ignore_index, reduction='none'
            )
        else:
            loss = F.cross_entropy(
                out, label,
                weight=self.class_weight,
                ignore_index=self.ignore_index,
                reduction='none'
            )
        
        if reduction == 'mean':
            n_valid = (label != self.ignore_index).sum()
            return (loss.sum() / max(n_valid, 1)).float()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss


class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    
    Used for edge detection and binary segmentation tasks.
    
    Args:
        pos_weight: Weight for positive samples (default: computed from data)
        ignore_index: Label index to ignore
    """
    
    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index
    
    def forward(self, output, label, reduction='mean'):
        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)
        
        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return torch.tensor(0.0, device=output.device)
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        
        factor = 1. / (1 - w)
        
        loss = F.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w * factor,
            reduction=reduction
        )
        loss /= factor
        return loss


class L1Loss(nn.Module):
    """
    L1 loss with ignore regions.
    
    Used for surface normal estimation and depth estimation.
    
    Args:
        normalize: Whether to normalize predictions (for surface normals)
        ignore_index: Value to ignore in labels
        ignore_invalid_area: Whether to ignore invalid regions
    """
    
    def __init__(self, normalize=False, ignore_index=0, ignore_invalid_area=True):
        super().__init__()
        self.normalize = normalize
        self.ignore_invalid_area = ignore_invalid_area
        if ignore_invalid_area:
            self.ignore_index = ignore_index
    
    def forward(self, out, label, reduction='mean'):
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        
        if self.ignore_invalid_area:
            mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        else:
            mask = torch.ones_like(label).all(dim=1, keepdim=True)
        
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        
        if reduction == 'mean':
            return F.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return F.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return F.l1_loss(masked_out, masked_label, reduction='none')


class SiLogLoss(nn.Module):
    """
    Scale-invariant logarithmic loss for depth estimation.
    
    Args:
        lambd: Balance factor (default: 0.5)
    """
    
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, pred, target):
        pred = F.relu(pred) + 1e-9
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() -
            self.lambd * torch.pow(diff_log.mean(), 2)
        )
        return loss


def get_task_loss(task, p=None, **kwargs):
    """
    Factory function to get loss function for a specific task.
    
    Args:
        task: Task name
        p: Configuration dict (optional)
        **kwargs: Additional arguments for loss construction
        
    Returns:
        Loss module
    """
    ignore_index = kwargs.get('ignore_index', 255)
    
    if task == 'edge':
        pos_weight = kwargs.get('pos_weight', kwargs.get('edge_w', 0.95))
        return BalancedBinaryCrossEntropyLoss(
            pos_weight=pos_weight, 
            ignore_index=ignore_index
        )
    
    elif task in ['semseg', 'human_parts', 'scene']:
        return CrossEntropyLoss(ignore_index=ignore_index)
    
    elif task == 'normals':
        return L1Loss(normalize=True, ignore_index=ignore_index)
    
    elif task == 'sal':
        return CrossEntropyLoss(balanced=True, ignore_index=ignore_index)
    
    elif task == 'depth':
        ignore_invalid = kwargs.get('ignore_invalid_area_depth', True)
        return L1Loss(ignore_invalid_area=ignore_invalid, ignore_index=0)
    
    else:
        # Default to cross entropy
        return CrossEntropyLoss(ignore_index=ignore_index)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss aggregation.
    
    Combines losses from multiple tasks with configurable weights,
    plus optional load balancing loss for routing mechanisms.
    
    Args:
        p: Configuration dict
        tasks: List of task names
        loss_ft: ModuleDict mapping task names to loss functions
        loss_weights: Dict mapping task names to loss weights
    """
    
    def __init__(self, p, tasks, loss_ft, loss_weights):
        super().__init__()
        assert set(tasks) == set(loss_ft.keys())
        
        self.p = p
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.cv_weight = loss_weights.get('load_balancing', 0.0)
    
    def forward(self, pred, gt, tasks=None):
        """
        Compute multi-task loss.
        
        Args:
            pred: Dict of predictions per task
            gt: Dict of ground truth per task
            tasks: List of tasks to compute loss for (default: all)
            
        Returns:
            Dict with 'total' loss and per-task losses
        """
        if tasks is None:
            tasks = self.tasks
        
        out = {}
        for task in tasks:
            if task in pred and task in gt:
                out[task] = self.loss_ft[task](pred[task], gt[task])
        
        # Compute weighted total
        out['total'] = torch.sum(torch.stack([
            self.loss_weights.get(t, 1.0) * out[t] 
            for t in tasks if t in out
        ]))
        
        # Add load balancing loss if routing info present
        if 'route_1_prob' in pred and self.cv_weight > 0:
            loss_cv = self._compute_cv_loss(pred)
            out['total'] = out['total'] + loss_cv * self.cv_weight
            out['cv'] = loss_cv
        
        # Add auxiliary losses from model
        for k in pred:
            if 'loss' in k and k not in out:
                out[k] = pred[k]
                weight = self.loss_weights.get(k, 1.0)
                out['total'] = out['total'] + pred[k] * weight
        
        return out
    
    def _compute_cv_loss(self, pred):
        """Compute coefficient of variation loss for load balancing."""
        loss_cv = 0
        
        for route_key in ['route_1_prob', 'route_2_prob']:
            if route_key in pred:
                for task_route in pred[route_key]:
                    task_route_list = [t_r for t_r in task_route.values()]
                    task_route_tensor = torch.cat(task_route_list, dim=0)
                    task_route_tensor = torch.mean(task_route_tensor, dim=0).reshape(-1)
                    loss = (torch.std(task_route_tensor) / (torch.mean(task_route_tensor) + 1e-8)) ** 2
                    loss_cv = loss_cv + loss
        
        return loss_cv


class MLoRECodingLoss(nn.Module):
    """
    Combined loss for MLoRE coding training.
    
    Combines multi-task losses with compression losses (bpp and mse).
    
    Args:
        p: Configuration dict
        tasks: List of task names
        loss_weights: Dict of loss weights including:
            - Task weights (e.g., 'semseg': 1.0)
            - 'bpp_loss': Weight for bits-per-pixel loss
            - 'mse_loss': Weight for reconstruction MSE loss
            - 'load_balancing': Weight for routing CV loss
    """
    
    def __init__(self, p, tasks, loss_weights=None):
        super().__init__()
        
        self.p = p
        self.tasks = tasks
        
        # Default weights
        default_weights = {
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'edge': 50.0,
            'normals': 10.0,
            'depth': 1.0,
            'scene': 1.0,
            'load_balancing': 0.0001,
            'bpp_loss': 1.0,
            'mse_loss': 100.0,
        }
        
        if loss_weights is not None:
            default_weights.update(loss_weights)
        self.loss_weights = default_weights
        
        # Create task-specific losses
        ignore_index = getattr(p, 'ignore_index', 255)
        edge_w = default_weights.get('edge_w', 0.95)
        
        self.loss_ft = nn.ModuleDict({
            task: get_task_loss(
                task, 
                ignore_index=ignore_index,
                edge_w=edge_w
            )
            for task in tasks
        })
        
        # Create multi-task loss aggregator
        self.multi_task_loss = MultiTaskLoss(
            p, tasks, self.loss_ft, self.loss_weights
        )
    
    def forward(self, pred, gt, tasks=None):
        """
        Compute combined coding loss.
        
        Args:
            pred: Dict with task predictions and 'bpp_loss', 'mse_loss'
            gt: Dict of ground truth per task
            tasks: List of tasks to compute loss for
            
        Returns:
            Dict with 'total' loss and all component losses
        """
        if tasks is None:
            tasks = self.tasks
        
        # Compute multi-task loss
        out = self.multi_task_loss(pred, gt, tasks)
        
        # Add compression losses
        if 'bpp_loss' in pred:
            bpp_loss = pred['bpp_loss']
            out['bpp_loss'] = bpp_loss
            out['total'] = out['total'] + bpp_loss * self.loss_weights['bpp_loss']
        
        if 'mse_loss' in pred:
            mse_loss = pred['mse_loss']
            out['mse_loss'] = mse_loss
            out['total'] = out['total'] + mse_loss * self.loss_weights['mse_loss']
        
        return out


# Wrapper to use RFC's original loss implementation
class RFCMultiTaskLoss(nn.Module):
    """
    Wrapper to use RFC's original MultiTaskLoss class.
    
    Args:
        p: Configuration dict
        tasks: List of task names
    """
    
    def __init__(self, p, tasks=None):
        super().__init__()
        
        if tasks is None:
            tasks = list(p.TASKS.NAMES)
        
        # Import migrated loss
        from mpcompress.losses.loss_schemes import MultiTaskLoss as RFCLoss
        from mpcompress.utils.mlore_common_config import get_loss
        
        loss_ft = nn.ModuleDict({
            task: get_loss(p, task) for task in tasks
        })
        loss_weights = p.get('loss_kwargs', {}).get('loss_weights', {})
        
        self.loss = RFCLoss(p, tasks, loss_ft, loss_weights)
        self.tasks = tasks
    
    def forward(self, pred, gt, tasks=None):
        """Forward pass."""
        if tasks is None:
            tasks = self.tasks
        return self.loss(pred, gt, tasks)




