#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2019-2021 Intflow Inc. All rights reserved.
# --Based on YOLOX made by Megavii Inc.--

import numpy as np
import cv2
import torch
import torchvision
import math
from retinanet._C import iou as iou_cuda
from apex import amp
from Rotated_IoU.oriented_iou_loss import cal_diou, cal_giou


__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "rbboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "rotate_boxes",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess_nms(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]##
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):##
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 6 : 6 + num_classes], 1, keepdim=True)
        conf_mask = (torch.sqrt(image_pred[:, 5] * class_conf.squeeze() + 1e-6) >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :6], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue##
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 5] * detections[:, 6],
            detections[:, 7],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))##
    return output


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        obj_conf = image_pred[:, 4].unsqueeze(-1)
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        rad_sin = image_pred[:,  -2].unsqueeze(-1)
        rad_cos = image_pred[:,  -1].unsqueeze(-1)
        rad = torch.atan2(rad_sin,rad_cos)
        conf_mask = (torch.sqrt(obj_conf.squeeze() * class_conf.squeeze() + 1e-6) >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :4], obj_conf, class_conf, class_pred.float(), rad), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue##
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            torch.sqrt(detections[:, 4] * detections[:, 5] + 1e-14),
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:4], bboxes_b[:, 2:4])
        area_a = torch.prod(bboxes_a[:, 2:4] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:4] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:4] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:4] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:4] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:4] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:4], 1)
        area_b = torch.prod(bboxes_b[:, 2:4], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def rbboxes_iou(target, pred, iou_type="diou", calc_type="smallest"):
    if target.shape[1] != 6 or pred.shape[1] != 6:
        raise IndexError

    rad_target = torch.atan2(target[:,4],target[:,5]).unsqueeze(-1)
    rad_pred = torch.atan2(pred[:,4],pred[:,5]).unsqueeze(-1)
    _target = torch.cat((target[:,:4],rad_target),dim=-1)
    _pred = torch.cat((pred[:,:4],rad_pred),dim=-1)
    
    n = _target.shape[0]
    m = _pred.shape[0]
    _target = _target[:,None,:] * torch.ones((m,5)).to(_target.device)
    _pred = torch.ones((n,5)).to(_pred.device)[:,None,:] * _pred
    
    if iou_type == "diou": #defualt as DIoU
        loss, iou = cal_diou(_pred, _target, calc_type)
    elif iou_type == "giou":
        loss, iou = cal_giou(_pred, _target, calc_type)

    del _pred, _target, loss
    return iou


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def rotate_box(rbbox):
    cx, cy, width, height, theta = rbbox

    xmin, ymin = cx - (width - 1) / 2, cy - (height - 1) / 2

    xy1 = xmin, ymin
    xy2 = xmin, ymin + height - 1
    xy3 = xmin + width - 1, ymin + height - 1
    xy4 = xmin + width - 1, ymin

    cents = np.array([cx, cy])

    corners = np.stack([xy1, xy2, xy3, xy4])

    u = np.stack([np.cos(theta), -np.sin(theta)])
    l = np.stack([np.sin(theta), np.cos(theta)])
    R = np.vstack([u, l])

    corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents

    return corners.reshape(-1).tolist()


def rotate_boxes(rbboxes):

    corners_list = []
    for rbbox in rbboxes:
        xmin, ymin, width, height, rad = rbbox

        xy1 = xmin, ymin
        xy2 = xmin, ymin + height - 1
        xy3 = xmin + width - 1, ymin + height - 1
        xy4 = xmin + width - 1, ymin

        cents = np.array([xmin + (width - 1) / 2, ymin + (height - 1) / 2])

        corners = np.stack([xy1, xy2, xy3, xy4])

        u = np.stack([np.cos(rad), -np.sin(rad)])
        l = np.stack([np.sin(rad), np.cos(rad)])
        R = np.vstack([u, l])

        corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents

        corners_list.append(corners)
    corners_list = np.array(corners_list)
    return corners_list

@amp.float_function
def order_points(pts):
    pts_reorder = []

    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, 0])
        xSorted = pt[idx, :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[torch.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        (br, tr) = rightMost[torch.argsort(D, descending=True), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))

    return torch.stack([p for p in pts_reorder])

def rotate_boxes_torch(boxes, points=False):
    '''
    Rotate target bounding boxes
    
    Input:  
        Target boxes (xmin_ymin, width_height, sin, cos)
    Output:
        boxes_axis (xmin_ymin, xmax_ymax, theta)
        boxes_rotated (xy0, xy1, xy2, xy3)
    '''

    u = torch.stack([boxes[:,5], boxes[:,4]], dim=1)
    l = torch.stack([-boxes[:,4], boxes[:,5]], dim=1)
    R = torch.stack([u, l], dim=1)

    if points:
        cents = torch.stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            boxes[:,2], boxes[:,1],
            boxes[:,2], boxes[:,3],
            boxes[:,0], boxes[:,3],
            boxes[:,-2],
            boxes[:,-1]],1)

    else:
        cents = torch.stack([boxes[:,0]+(boxes[:,2])/2, boxes[:,1]+(boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            (boxes[:,0]+boxes[:,2]), boxes[:,1],
            (boxes[:,0]+boxes[:,2]), (boxes[:,1]+boxes[:,3]),
            boxes[:,0], (boxes[:,1]+boxes[:,3]),
            boxes[:,-2],
            boxes[:,-1]],1)

    xy0R = torch.matmul(R,boxes_rotated[:,:2].transpose(1,0) - cents) + cents
    xy1R = torch.matmul(R,boxes_rotated[:,2:4].transpose(1,0) - cents) + cents
    xy2R = torch.matmul(R,boxes_rotated[:,4:6].transpose(1,0) - cents) + cents
    xy3R = torch.matmul(R,boxes_rotated[:,6:8].transpose(1,0) - cents) + cents

    xy0R = torch.stack([xy0R[i,:,i] for i in range(xy0R.size(0))])
    xy1R = torch.stack([xy1R[i,:,i] for i in range(xy1R.size(0))])
    xy2R = torch.stack([xy2R[i,:,i] for i in range(xy2R.size(0))])
    xy3R = torch.stack([xy3R[i,:,i] for i in range(xy3R.size(0))])

    boxes_rotated = order_points(torch.stack([xy0R,xy1R,xy2R,xy3R],dim = 1)).view(-1,8)
    
    return boxes_rotated

def rotated2rect(rbbox, width, height):
    corners = rotate_box(rbbox)
    x1, y1, x2, y2 = bounding_box_naive(corners)
    w = x2 - x1
    h = y2 - y1
    w_h = 0.5*w
    h_h = 0.5*h
    cx = x1 + w_h
    cy = y1 + h_h

    #resize by rad
    rad = abs(rbbox[-1])
    min_scale = 0.3 / np.deg2rad(45)
    bbox_scale = 1.0 - rad*min_scale
    w_h *= bbox_scale
    h_h *= bbox_scale
    
    x1 = np.max((0, cx-w_h))
    y1 = np.max((0, cy-h_h))
    x2 = np.min((width - 1, cx + np.max((0, w_h - 1))))
    y2 = np.min((height - 1, cy + np.max((0, h_h - 1))))

    return x1, y1, x2, y2

def rotated_reform(rbbox, width, height):
    cx, cy, w, h, rad = rbbox
    
    w_h = 0.5*w
    h_h = 0.5*h
    
    x1 = np.max((0, cx-w_h))
    y1 = np.max((0, cy-h_h))
    x2 = np.min((width - 1, cx + np.max((0, w_h - 1))))
    y2 = np.min((height - 1, cy + np.max((0, h_h - 1))))

    return x1, y1, x2, y2, rad

def bounding_box_naive(corners):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    points = np.array(corners).reshape(4,2).tolist()
    top_left_x = min(point[0] for point in points)
    top_left_y = min(point[1] for point in points)
    bot_right_x = max(point[0] for point in points)
    bot_right_y = max(point[1] for point in points)

    return [top_left_x, top_left_y, bot_right_x, bot_right_y]

def calc_bearing(corners1, corners2):
    lat1, long1 = corners1
    lat2, long2 = corners2
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.rad2deg(brng)

    return brng

def corners2rotatedbbox(corners):
    if len(corners) == 8:
        corners = np.resize(corners, (4,2))
    centre = np.mean(np.array(corners), 0)
    theta = calc_bearing(corners[0], corners[1])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    out_points = np.matmul(corners - centre, rotation) + centre
    x1, y1 = list(out_points[0,:])
    x2, y2 = list(out_points[2,:])
    return [x1, y1, x2, y2, theta]