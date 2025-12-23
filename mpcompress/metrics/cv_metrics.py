from typing import List, Dict
import torch
import numpy as np


class TopKAccuracyMetric:
    """Metric for computing top-K accuracy.

    This metric tracks the accuracy of predictions where the correct class
    appears in the top-K predicted classes. It accumulates results across
    multiple batches and computes the final accuracy percentage.
    """

    def __init__(self, topk: List[int] = [1, 5]):
        """Initialize TopKAccuracyMetric.

        Args:
            topk (List[int], optional): List of K values to compute accuracy for.
                The list will be sorted in ascending order. Defaults to [1, 5].
        """
        self.topk = sorted(topk)  # Ensure ascending order, e.g., [1, 5]
        self.correct_counts = {k: 0 for k in self.topk}
        self.total_samples = 0

    def update(self, predictions: List[List[int]], targets: List[int]):
        """Update the metric with a batch of predictions and targets.

        Args:
            predictions (List[List[int]]): List of prediction lists, where each
                inner list contains class indices sorted by confidence (highest first).
            targets (List[int]): List of ground truth class indices.
        """
        for pred, target in zip(predictions, targets):
            for k in self.topk:
                if target in pred[:k]:
                    self.correct_counts[k] += 1
            self.total_samples += 1

    def compute(self) -> Dict[str, float]:
        """Compute the top-K accuracy metrics.

        Returns:
            Dict[str, float]: Dictionary mapping metric names to accuracy percentages.
                Keys are in the format "top-{k}" (e.g., "top-1", "top-5").
                Returns zeros if no samples have been processed.
        """
        if self.total_samples == 0:
            return {f"top-{k}": 0.0 for k in self.topk}

        return {
            f"top-{k}": (self.correct_counts[k] / self.total_samples) * 100
            for k in self.topk
        }


class MeanIoUMetric:
    """Metric for computing mean Intersection over Union (mIoU).

    This metric computes the mean IoU across all classes for semantic segmentation
    tasks. It accumulates a confusion matrix across batches and computes the
    per-class IoU, then takes the mean.
    """

    def __init__(self, num_classes: int = 21):
        """Initialize MeanIoUMetric.

        Args:
            num_classes (int, optional): Number of classes in the segmentation task.
                Defaults to 21.
        """
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, preds, target):
        """Update the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor or np.ndarray): Predicted class indices.
                Shape: [B, H, W] or [H, W].
            target (torch.Tensor or np.ndarray): Ground truth class indices.
                Shape: [B, H, W] or [H, W]. Must match preds shape.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        self.hist += self.fast_hist(target, preds, self.num_classes)

    def compute(self):
        """Compute the mean IoU metric.

        Returns:
            average (dict): Dictionary containing:

                - "mIoU" (float): Mean Intersection over Union across all classes.
        """
        iou = self.per_class_iou(self.hist)
        mean_iou = np.nanmean(iou)
        return {"mIoU": mean_iou}

    def fast_hist(self, label, prediction, n):
        """Compute confusion matrix efficiently.

        Args:
            label (np.ndarray): Ground truth class indices.
            prediction (np.ndarray): Predicted class indices. Must match label shape.
            n (int): Number of classes.

        Returns:
            hist (np.ndarray): Confusion matrix of shape [n, n] where entry [i, j] is the
                count of pixels with true class i and predicted class j.
        """
        k = (label >= 0) & (label < n)
        return np.bincount(
            n * label[k].astype(int) + prediction[k].astype(int), minlength=n * n
        ).reshape(n, n)

    def per_class_iou(self, hist):
        """Compute Intersection over Union for each class.

        Args:
            hist (np.ndarray): Confusion matrix of shape [n, n].

        Returns:
            iou (np.ndarray): Per-class IoU values of shape [n]. IoU for class i is:
                IoU_i = hist[i, i] / (sum(hist[i, :]) + sum(hist[:, i]) - hist[i, i])
        """
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        return iou
