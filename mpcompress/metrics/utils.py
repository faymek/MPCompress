import torch
import pandas as pd
import numpy as np


class DictAverageMeter:
    """Averaging meter for dictionary records.

    This class accumulates numeric values from dictionary records and computes
    their averages. It filters out non-numeric values and handles various numeric
    types including scalars, single-element tensors, and objects with arithmetic
    operations.

    Example:

        Input record: {"key1": value1, "key2": value2, ...}
        Average record: {"key1": avg_value1, "key2": avg_value2, ...}
    """

    def __init__(self):
        """Initialize DictAverageMeter.

        Creates an empty meter with no accumulated values.
        """
        self.meter = None
        self.count = 0

    def _only_keep_number_items(self, record):
        """Filter record to keep only numeric items.

        Args:
            record (dict): Input dictionary with potentially mixed types.

        Returns:
            dict: Dictionary containing only numeric values. Supports:
                - int and float
                - Single-element torch.Tensor (converted to Python scalar)
                - Objects with __add__ and __truediv__ methods
        """
        new_record = {}
        for key, value in record.items():
            if isinstance(value, (int, float)):
                new_record[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                new_record[key] = value.item()
            elif hasattr(value, "__add__") and hasattr(value, "__truediv__"):
                new_record[key] = value
        return new_record

    def update(self, record, n=1):
        """Update the meter with a new record.

        Args:
            record (dict): Dictionary of key-value pairs to accumulate.
                Non-numeric values are filtered out.
            n (int, optional): Number of samples this record represents.
                Used for weighted averaging. Defaults to 1.
        """
        record = self._only_keep_number_items(record)
        if self.meter is None:
            self.meter = record
        else:
            for key, value in record.items():
                self.meter[key] += value
        self.count += n

    def average(self):
        """Compute the average of all accumulated values.

        Returns:
            average (dict): Dictionary mapping keys to their average values.
                Each value is computed as: accumulated_value / count

        Raises:
            AttributeError: If no records have been updated yet (meter is None).
        """
        record_avg = {}
        for key in self.meter.keys():
            record_avg[key] = self.meter[key] / self.count
        return record_avg


class DataFrameRecords:
    """Record storage and averaging using pandas DataFrame.

    This class stores records in a pandas DataFrame with _id as the index,
    allowing records to be updated by their _id. It supports computing averages
    across all stored records.

    Example:

        Input record: {"_id": id1, "key1": value1, "key2": value2, ...}
        Average record: {"key1": avg_value1, "key2": avg_value2, ...}

    The record can be updated by _id. Implemented using pandas DataFrame,
    supporting row updates by _id.
    """

    def __init__(self):
        """Initialize DataFrameRecords.

        Creates an empty DataFrame for storing records. The _id field will
        be used as the DataFrame index.
        """
        self.df = pd.DataFrame()  # Use DataFrame to store data, with _id as index

    def __len__(self):
        """Get the number of records.

        Returns:
            length (int): Number of records stored in the DataFrame.
        """
        return len(self.df)

    def get_record_by_id(self, _id):
        """Retrieve a record by its _id.

        Args:
            _id (str): The identifier of the record to retrieve.

        Returns:
            record (dict): Dictionary representation of the record with the given _id.

        Raises:
            KeyError: If the _id does not exist in the DataFrame.
        """
        return self.df.loc[_id].to_dict()

    def update(self, record):
        """Update or insert a record by _id.

        If a record with the given _id exists, it will be updated with new values.
        Otherwise, a new record will be inserted. The _id field is used as the
        DataFrame index and is removed from the record data.

        Args:
            record (dict): Dictionary containing "_id" and other key-value pairs.
                The "_id" field is required and will be used as the DataFrame index.

        Raises:
            AssertionError: If the record does not contain "_id" key.
            KeyError: If the _id does not exist in the DataFrame.
        """
        assert "_id" in record, "record must contain _id"
        _id = record["_id"]

        # Create new row data (exclude _id, as _id is used as index)
        record.pop("_id")

        if self.df.empty:
            # If DataFrame is empty, create it directly
            self.df = pd.DataFrame([record], index=[_id])
        elif _id in self.df.index:
            # If _id already exists, update the row
            for key, value in record.items():
                self.df.at[_id, key] = value
        else:
            # If _id does not exist, add a new row
            new_df = pd.DataFrame([record], index=[_id])
            self.df = pd.concat([self.df, new_df])

    def average(self):
        """Compute the average of all numeric columns across all records.

        Returns:
            average (dict): Dictionary mapping column names to their mean values.
                Only numeric columns are included in the result.
        """
        return self.df.mean().to_dict()


# 计算两个边界框的 IoU
def compute_iou(box1, box2):
    # box 格式：[x_min, y_min, x_max, y_max]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

# 计算单个类别的平均精度（AP）
def compute_ap_for_class(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0.0
    # 将预测结果按照置信度降序排序
    sorted_inds = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in sorted_inds]
    pred_scores = [pred_scores[i] for i in sorted_inds]
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    assigned = np.zeros(len(gt_boxes))  # 标记每个 gt 是否已匹配

    for i, pbox in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt = -1
        for j, gtbox in enumerate(gt_boxes):
            iou = compute_iou(pbox, gtbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        if best_iou >= iou_threshold and assigned[best_gt] == 0:
            tp[i] = 1
            assigned[best_gt] = 1
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (len(gt_boxes) + 1e-6)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    # 计算 AP（使用简单的数值积分）
    ap = 0.0
    prev_rec = 0.0
    for r, p in zip(recalls, precisions):
        ap += p * (r - prev_rec)
        prev_rec = r
    return ap

def compute_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, num_classes, iou_threshold=0.5):
    """
    计算多个类别的平均精度均值（mAP）
    
    Args:
        pred_boxes: 预测的边界框 numpy array of shape (N, 4)
        pred_labels: 预测的标签 numpy array of shape (N,)
        pred_scores: 预测的置信度 numpy array of shape (N,)
        gt_boxes: 真实的边界框 numpy array of shape (M, 4)
        gt_labels: 真实的标签 numpy array of shape (M,)
        num_classes: 类别数量（不包括背景）
        iou_threshold: IoU阈值
    
    Returns:
        mAP: 所有类别的平均AP
        ap_per_class: 每个类别的AP字典
    """
    ap_per_class = {}
    valid_classes = []
    
    # 获取所有存在的类别
    all_classes = set(gt_labels.tolist()) | set(pred_labels.tolist())
    
    for class_id in all_classes:
        if class_id == 0:  # 跳过背景类
            continue
            
        # 获取该类别的预测和真实框
        pred_mask = pred_labels == class_id
        gt_mask = gt_labels == class_id
        
        class_pred_boxes = pred_boxes[pred_mask]
        class_pred_scores = pred_scores[pred_mask]
        class_gt_boxes = gt_boxes[gt_mask]
        
        # 计算该类别的AP
        ap = compute_ap_for_class(class_pred_boxes, class_pred_scores, class_gt_boxes, iou_threshold)
        ap_per_class[class_id] = ap
        
        # 只有当该类别有真实标注时才计入mAP计算
        if len(class_gt_boxes) > 0:
            valid_classes.append(class_id)
    
    # 计算mAP（只计算有真实标注的类别）
    if len(valid_classes) > 0:
        mAP = np.mean([ap_per_class[class_id] for class_id in valid_classes])
    else:
        mAP = 0.0
    
    return mAP, ap_per_class

def compute_map_at_multiple_ious(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, num_classes):
    """
    计算在多个IoU阈值下的mAP（类似COCO评估）
    
    Returns:
        mAP_50: IoU=0.5时的mAP
        mAP_75: IoU=0.75时的mAP  
        mAP_50_95: IoU从0.5到0.95（步长0.05）的平均mAP
    """
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    mAPs = []
    for iou_thresh in iou_thresholds:
        mAP, _ = compute_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, num_classes, iou_thresh)
        mAPs.append(mAP)
    
    mAP_50 = mAPs[0]  # IoU=0.5
    mAP_75 = mAPs[5]  # IoU=0.75
    mAP_50_95 = np.mean(mAPs)  # 平均mAP
    
    return mAP_50, mAP_75, mAP_50_95, mAPs


# 指标2
def calculate_video_metrics(model, input_video, gt_video, device, csv_path=None):
    """计算视频间指标"""
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    metrics = {
        'psnr_original': [],
        'lpips_original': [],
        'psnr_enhanced': [],
        'lpips_enhanced': []
    }
    
    # 同时读取两个视频
    input_gen = load_video(input_video)
    gt_gen = load_video(gt_video)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    with torch.no_grad():
        for input_frame, gt_frame in zip(input_gen, gt_gen):
            # 转换颜色空间和Tensor
            input_tensor = transform(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            gt_tensor = transform(cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            
            # 处理帧
            enhanced_tensor = model(input_tensor)
            
            # 计算指标
            metrics['psnr_original'].append(psnr(gt_tensor.cpu().numpy(), input_tensor.cpu().numpy(), data_range=1.0))
            metrics['lpips_original'].append(lpips_model(gt_tensor, input_tensor).item())
            metrics['psnr_enhanced'].append(psnr(gt_tensor.cpu().numpy(), enhanced_tensor.cpu().numpy(), data_range=1.0))
            metrics['lpips_enhanced'].append(lpips_model(gt_tensor, enhanced_tensor).item())
    
    # 计算平均指标
    avg_metrics = {k: sum(v)/len(v) for k, v in metrics.items()}
    
    # 保存到CSV
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Original', 'Enhanced'])
            writer.writerow(['PSNR', avg_metrics['psnr_original'], avg_metrics['psnr_enhanced']])
            writer.writerow(['LPIPS', avg_metrics['lpips_original'], avg_metrics['lpips_enhanced']])
    
    return avg_metrics

    
def load_detection_model(device):
    """加载目标检测模型"""
    import torchvision
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained=True).to(device)
    model.eval()
    return model
