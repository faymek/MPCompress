# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import glob
import json
import numpy as np
import torch
from PIL import Image
import pdb

class ClassificationMeter(object):
    def __init__(self, database):
        """
        Classification performance meter
        
        Args:
            database: Dataset name (e.g., 'CIFAR10', 'ImageNet')
            class_names: List of class names (optional)
        """
        if database == 'NYUD':
            self.n_classes = 27
            self.class_names = [
                'basement', 'bathroom', 'bedroom', 'bookstore', 'cafe',
                'classroom', 'computer_lab', 'conference_room', 'dinette',
                'dining_room', 'excercise_room', 'foyer', 'furniture_store',
                'home_office', 'home_storage', 'indoor_balcony', 'kitchen',
                'laundry_room', 'living_room', 'office', 'office_kitchen',
                'playroom', 'printer_room', 'reception_room', 'student_lounge',
                'study', 'study_room']
        else:
            raise NotImplementedError
            
        self.confusion_matrix = None
        self.total_samples = 0
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.total_samples = 0

    @torch.no_grad()
    def update(self, pred, target):
        """
        Update metrics with new batch of predictions
        
        Args:
            pred: Model predictions [N]
            target: Ground truth labels [N]
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
                        
        # Convert predictions to class indices
        pred_classes = pred
        
        # Update confusion matrix
        for t, p in zip(target, pred_classes):
            self.confusion_matrix[t, p] += 1
        
        self.total_samples += len(target)

    def get_score(self, verbose=True):
        """Compute and return evaluation metrics"""
        eval_result = {}
        
        # Calculate per-class and overall accuracy
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        
        # Per-class precision, recall, f1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Overall metrics
        accuracy = np.sum(tp) / self.total_samples
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        eval_result['accuracy'] = accuracy * 100
        #eval_result['avg_precision'] = avg_precision * 100
        #eval_result['avg_recall'] = avg_recall * 100
        #eval_result['avg_f1'] = avg_f1 * 100
        #eval_result['class_accuracy'] = (tp / np.sum(self.confusion_matrix, axis=1)) * 100
        #eval_result['confusion_matrix'] = self.confusion_matrix
        
        if verbose:
            print('\nClassification Metrics:')
            print('Overall Accuracy: {0:.2f}%'.format(eval_result['accuracy'])) 
            cls_accuracy = (tp / np.sum(self.confusion_matrix, axis=1)) * 100           
            if self.class_names is not None and len(self.class_names) == self.n_classes:
                print('\nPer-class Accuracy:')
                for i, name in enumerate(self.class_names):
                    spaces = ' ' * (20 - len(name))
                    print('{0:s}{1:s}{2:.2f}%'.format(
                        name, spaces, cls_accuracy[i]))
        
        return eval_result