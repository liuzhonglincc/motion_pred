import torch
import numpy as np
def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU值
    参数:
        box1 (Tensor): 第一个边界框，形状为[N, 4]，包含N个边界框的坐标信息（x_min, y_min, x_max, y_max）
        box2 (Tensor): 第二个边界框，形状为[N, 4]，包含N个边界框的坐标信息（x_min, y_min, x_max, y_max）
    返回:
        iou (Tensor): IoU值，形状为[N]
    """
    # 计算交集的左上角和右下角坐标
    intersect_tl = torch.max(box1[:, :2], box2[:, :2])
    intersect_br = torch.min(box1[:, 2:], box2[:, 2:])

    # 计算交集的宽度和高度
    intersect_wh = torch.clamp(intersect_br - intersect_tl, min=0)

    # 计算交集面积
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]

    # 计算并集面积
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - intersect_area

    # 计算IoU值
    iou = intersect_area / union_area

    return iou

def calculate_giou(box1, box2):
    iou = calculate_iou(box1, box2)

    # 计算并集的面积
    union_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) + \
                (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 计算最小闭包框的宽度和高度
    enclosing_wh = torch.clamp(box1[:, 2:] - box1[:, :2], min=0) + \
                torch.clamp(box2[:, 2:] - box2[:, :2], min=0)

    # 计算最小闭包框的面积
    enclosing_area = enclosing_wh[:, 0] * enclosing_wh[:, 1]

    # 计算GIoU
    giou = iou - (enclosing_area - union_area) / enclosing_area

    return giou

def calculate_diou(box1, box2):
    iou = calculate_iou(box1, box2)

    # 计算框的中心点坐标
    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2

    # 计算框的对角线距离的平方
    dist_sq = torch.sum((center1 - center2) ** 2, dim=1)

    # 计算最小闭包框的对角线距离的平方
    enclosing_diag_sq = torch.sum((torch.max(box1[:, :2], box2[:, :2]) - torch.min(box1[:, 2:], box2[:, 2:])) ** 2, dim=1)

    # 计算DIoU
    diou = iou - dist_sq / enclosing_diag_sq

    return diou

def calculate_ciou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    iou = calculate_iou(box1, box2)

    C_xx1 = torch.min(b1_x1, b2_x1)
    C_yy1 = torch.min(b1_y1, b2_y1)
    C_xx2 = torch.max(b1_x2, b2_x2)
    C_yy2 = torch.max(b1_y2, b2_y2)

    # DISTANCE
    center_b_x = (b1_x1 + b1_x2) / 2
    center_b_y = (b1_y1 + b1_y2) / 2
    center_gtb_x = (b2_x1 + b2_x2) / 2
    center_gtb_y = (b2_y1 + b2_y2) / 2
    C_area = (C_xx2 - C_xx1) * (C_yy2 - C_yy1)
    Distance = (center_gtb_x - center_b_x) ** 2 + (center_gtb_y - center_b_y) ** 2
    Distance_area = Distance / (C_area ** 2)

    # aspect ratio
    pred_w = b1_y2 - b1_y1
    pred_h = b1_x2 - b1_x1
    gt_w = b2_y2 - b2_y1
    gt_h = b2_x2 - b2_x1
    v = (4 / (torch.tensor(np.pi) ** 2)) * (torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h)) ** 2
    # 训练ciou出现nan, 就是因为预测的box和gt完全一样，iou=1 v=0, 这里除数为0，alpha无限大，  使用指数函数避免除数为0
    alpha = torch.zeros(v.shape[0], device=v.device)
    mask = iou != 1.0
    alpha[mask] = v[mask] / ((1 - iou[mask]) + v[mask])

    ciou = iou - Distance_area - alpha * v
    return ciou

def calculate_eiou(boxes1, boxes2):
    """
    计算两组框的EIoU损失
    Args:
        boxes1: 第一组框的坐标，shape 为 [N, 4]，其中 N 是框的数量
        boxes2: 第二组框的坐标，shape 为 [N, 4]，其中 N 是框的数量
    Returns:
        eiou_loss: EIoU 损失值，shape 为 [N]
    """
    eps = 1e-7
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # 计算两组框的IoU
    iou = calculate_iou(boxes1, boxes2)
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared

    rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
    rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
    cw2 = cw ** 2 + eps
    ch2 = ch ** 2 + eps
    eiou = iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)

    return eiou

def calculate_alpha_ciou(box1, box2):
    eps=1e-7
    alpha = 4
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter/union + eps, alpha)

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex heigh
    c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal
    rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
    rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
    rho2 = ((rho_x ** 2 + rho_y ** 2) / 4) ** alpha  # center distance

    v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha_ciou = v / ((1 + eps) - inter / union + v)
    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU

def calculate_siou(boxes1, boxes2):
    # 计算两组框的IoU
    iou = calculate_iou(boxes1, boxes2)
    # Compute SIoU terms
    si = torch.min(boxes1[:, 2], boxes2[:, 2]) - torch.max(boxes1[:, 0], boxes2[:, 0]) + 1
    sj = torch.min(boxes1[:, 3], boxes2[:, 3]) - torch.max(boxes1[:, 1], boxes2[:, 1]) + 1
    s_union = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1) + \
              (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    s_intersection = si * sj

    # Compute SCYLLA-IoU
    siou = iou - (s_intersection / s_union)

    return siou

def convert_boxes(boxes):
    # boxes 维度 Nx4, 为 (center_x, center_y, w, h)
    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    left = center_x - (w / 2)
    top = center_y - (h / 2)
    right = center_x + (w / 2)
    bottom = center_y + (h / 2)

    # converted_boxes 维度 Nx4, 为 (x_min, y_min, x_max, y_max)
    converted_boxes = torch.stack((left, top, right, bottom), dim=1)
    return converted_boxes
