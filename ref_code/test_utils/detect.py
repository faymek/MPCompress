import torch
import torchvision
import cv2

def detect_img(frame, device, net, model, threshold=0.3):
    frame_npy = frame.to_ndarray(format='rgb24')
    frame_tensor = torch.from_numpy(frame_npy).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    frame_tensor = net.postprocess(frame_tensor)

    # 进行预测
    with torch.no_grad():
        prediction = model(frame_tensor)

    # 获取预测的边界框、标签以及分数
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # 设置置信度阈值
    mask = (scores > threshold) & ((labels == 3) | (labels == 1))
    filtered_boxes = boxes[mask]
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]
    
    return filtered_boxes, filtered_labels, filtered_scores


def detect_img_from_id(image_id, model, threshold=0.3):
    img = cv2.imread(image_id)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_transposed = img_rgb.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_rgb_transposed).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # 增加batch维度

    # 进行预测
    with torch.no_grad():
        prediction = model(img_tensor.cuda())

    # 获取预测的边界框、标签以及分数
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # 设置置信度阈值
    mask = (scores > threshold) & ((labels == 3) | (labels == 1))
    filtered_boxes = boxes[mask]
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]
    
    return filtered_boxes, filtered_labels, filtered_scores

def load_detection_model(device):
    """加载目标检测模型"""
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained=True).to(device)
    model.eval()
    return model