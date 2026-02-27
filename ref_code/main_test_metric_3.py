import torch
import torchvision
import numpy as np
import os
from tqdm import tqdm
from test_utils.metrics import compute_map, compute_ap_for_class
from test_utils.detect import load_detection_model
from test_utils.parser import parse_calcu_metric_args

def detect_frames(args):
    """检测并计算指标"""
    # 设置设备
    device = 'cuda:0' if args.device == 'cuda' else 'cpu'
    
    # 加载模型
    model = load_detection_model(device)
    
    # 定义目标检测的类别
    num_classes = 90  # COCO数据集有80个类别+背景
    target_classes = [1, 3]  # 本任务只关注汽车与行人
    
    # 初始化统计数据结构
    class_ap_totals = {class_id: 0.0 for class_id in target_classes}
    class_counts = {class_id: 0 for class_id in target_classes}
    class_frame_aps = {class_id: [] for class_id in target_classes}
    frame_map = []
    frame_map_per_class = {class_id: [] for class_id in target_classes}
    
    # 加载标签
    label_path = args.gt_dir
    label_list = sorted([f for f in os.listdir(label_path) if f.endswith('.pt')])
    
    # 主循环处理每一帧
    for gt_name in tqdm(os.listdir(label_path), desc="Processing enhanced frames"):
        # 加载增强后的帧
        frame_path = os.path.join(args.frame_dir, gt_name.replace(".pt", ".png"))
        frame_tensor = torchvision.io.read_image(frame_path).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)
        
        # 加载ground truth
        gt_id = os.path.join(label_path, gt_name)
        gt = torch.load(gt_id)
        gt_video_boxes = gt['boxes'].numpy()
        gt_video_labels = gt['labels'].numpy()
        
        # 进行目标检测
        with torch.no_grad():
            prediction = model(frame_tensor)
        
        # 解析预测结果
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        # 应用置信度阈值
        mask = (scores > 0.5) & ((labels == 3) | (labels == 1))
        origin_video_boxes = boxes[mask]
        origin_video_labels = labels[mask]
        origin_video_scores = scores[mask]
        
        # 计算指标
        if len(origin_video_boxes) > 0 and len(gt_video_boxes) > 0:
            target_pred_mask = np.isin(origin_video_labels, target_classes)
            target_gt_mask = np.isin(gt_video_labels, target_classes)
            
            if np.any(target_pred_mask) and np.any(target_gt_mask):
                mAP, ap_per_class = compute_map(
                    origin_video_boxes[target_pred_mask], 
                    origin_video_labels[target_pred_mask], 
                    origin_video_scores[target_pred_mask],
                    gt_video_boxes[target_gt_mask], 
                    gt_video_labels[target_gt_mask], 
                    num_classes, iou_threshold=0.7
                )
                frame_map.append(mAP)
                
                for class_id in target_classes:
                    if class_id in ap_per_class:
                        frame_map_per_class[class_id].append(ap_per_class[class_id])
            else:
                frame_map.append(0.0)
                for class_id in target_classes:
                    if np.any(gt_video_labels == class_id):
                        frame_map_per_class[class_id].append(0.0)
        else:
            frame_map.append(0.0)
            for class_id in target_classes:
                if len(gt_video_boxes) > 0 and np.any(gt_video_labels == class_id):
                    frame_map_per_class[class_id].append(0.0)
        
        # 计算每个类别的AP
        for class_id in target_classes:
            pred_mask = origin_video_labels == class_id
            gt_mask = gt_video_labels == class_id
            
            if np.any(pred_mask) and np.any(gt_mask):
                ap = compute_ap_for_class(
                    origin_video_boxes[pred_mask], 
                    origin_video_scores[pred_mask], 
                    gt_video_boxes[gt_mask]
                )
                class_ap_totals[class_id] += ap
                class_counts[class_id] += 1
                class_frame_aps[class_id].append(ap)
            elif np.any(gt_mask):
                ap = 0.0
                class_ap_totals[class_id] += ap
                class_counts[class_id] += 1
                class_frame_aps[class_id].append(ap)
    
    # 打印结果
    print_results(frame_map, frame_map_per_class, 
                 class_ap_totals, class_counts, class_frame_aps, target_classes)

def print_results(frame_map, frame_map_per_class, 
                 class_ap_totals, class_counts, class_frame_aps, target_classes):
    """打印结果统计"""
    print("\n=== 整体评估结果（基于单图平均）===")
    if frame_map:
        overall_mAP = np.mean(frame_map)
        mAP_std = np.std(frame_map)
        print(f"Overall mAP@0.7: {overall_mAP:.4f}")
        print(f"mAP 标准差: {mAP_std:.4f}")
        print(f"mAP 最高值: {max(frame_map):.4f}")
        print(f"mAP 最低值: {min(frame_map):.4f}")
        print(f"mAP 中位数: {np.median(frame_map):.4f}")
        
        print(f"\n各类别整体mAP统计:")
        for class_id in target_classes:
            if frame_map_per_class[class_id]:
                class_maps = frame_map_per_class[class_id]
                class_mean_map = np.mean(class_maps)
                class_std_map = np.std(class_maps)
                print(f"  类别 {class_id}: mAP={class_mean_map:.4f}, std={class_std_map:.4f}, 样本数={len(class_maps)}")
    
    print(f"\n=== 各类别详细评估结果 ===")
    for class_id in target_classes:
        if class_counts[class_id] > 0:
            class_avg_ap = class_ap_totals[class_id] / class_counts[class_id]
            class_std_ap = np.std(class_frame_aps[class_id]) if len(class_frame_aps[class_id]) > 1 else 0.0
            
            print(f"\n类别 {class_id}:")
            print(f"  样本数量: {class_counts[class_id]}")
            print(f"  平均 AP: {class_avg_ap:.4f}")
            print(f"  AP 标准差: {class_std_ap:.4f}")
            
            if class_frame_aps[class_id]:
                class_aps = class_frame_aps[class_id]
                print(f"  最高 AP: {max(class_aps):.4f}")
                print(f"  最低 AP: {min(class_aps):.4f}")
                print(f"  中位数 AP: {np.median(class_aps):.4f}")
    
    total_samples = sum(class_counts.values())
    if total_samples > 0:
        weighted_avg_ap = sum(class_ap_totals.values()) / total_samples
        print(f"\n=== 所有关注类别整体统计 ===")
        print(f"总样本数: {total_samples}")
        print(f"加权平均 AP: {weighted_avg_ap:.4f}")

if __name__ == '__main__':
    args = parse_calcu_metric_args()
    detect_frames(args)