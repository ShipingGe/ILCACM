import torch


def iou_1d(boxes_a, boxes_b):
    assert boxes_a.size(-1) == 2
    assert boxes_b.size(-1) == 2

    left = torch.max(boxes_a[..., 0], boxes_b[..., 0])
    right = torch.min(boxes_a[..., 1], boxes_b[..., 1])

    intersection = torch.clamp(right - left, min=0)
    union = (boxes_a[..., 1] - boxes_a[..., 0]) + (boxes_b[..., 1] - boxes_b[..., 0]) - intersection

    return intersection / union


def generalized_iou_1d(boxes_a, boxes_b):
    iou = iou_1d(boxes_a, boxes_b)

    left = torch.min(boxes_a[..., 0], boxes_b[..., 0])
    right = torch.max(boxes_a[..., 1], boxes_b[..., 1])

    area_c = right - left

    return iou - (area_c - torch.clamp(right - left, min=0)) / area_c


def giou_loss_1d(boxes_a, boxes_b, reduction='mean'):
    giou = generalized_iou_1d(boxes_a, boxes_b)
    loss = 1 - giou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
