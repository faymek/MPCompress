from typing import List, Dict
import torch
import numpy as np


class TopKAccuracyMetric:
    def __init__(self, topk: List[int] = [1, 5]):
        self.topk = sorted(topk)  # 确保升序排列，如 [1, 5]
        self.correct_counts = {k: 0 for k in self.topk}
        self.total_samples = 0

    def update(self, predictions: List[List[int]], targets: List[int]):
        for pred, target in zip(predictions, targets):
            for k in self.topk:
                if target in pred[:k]:
                    self.correct_counts[k] += 1
            self.total_samples += 1

    def compute(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {f"top-{k}": 0.0 for k in self.topk}

        return {
            f"top-{k}": (self.correct_counts[k] / self.total_samples) * 100
            for k in self.topk
        }


class MeanIoUMetric:
    def __init__(self, num_classes: int = 21):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, preds, target):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        self.hist += self.fast_hist(target, preds, self.num_classes)

    def compute(self):
        iou = self.per_class_iou(self.hist)
        mean_iou = np.nanmean(iou)
        return {"mIoU": mean_iou}

    def fast_hist(self, label, prediction, n):
        k = (label >= 0) & (label < n)
        return np.bincount(
            n * label[k].astype(int) + prediction[k].astype(int), minlength=n * n
        ).reshape(n, n)

    def per_class_iou(self, hist):
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        return iou
