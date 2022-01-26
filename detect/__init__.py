# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np

from typing import List
from copy import deepcopy

from .. import volume


class BaseBox(object):
    def __init__(self):
        self.min = None
        self.max = None
        self.score = 0

    def box_area(self):
        if self.min is None or self.max is None:
            return 0.
        return np.prod(self.max - self.min, axis=-1)


class BoundingBox(BaseBox):
    def __init__(self, init_list, score=0, cls=0):
        super().__init__()
        if isinstance(init_list, list):
            init_list = np.array(init_list, dtype=np.float32)
        init_list = np.reshape(init_list, [2, -1])
        self.min = np.array(init_list[0])
        self.max = np.array(init_list[1])
        self.score = score
        self.cls = cls

    def as_cnt_scale(self):
        cnt = (self.min + self.max) / 2
        scale = self.max - cnt
        return np.concatenate([cnt, scale], axis=-1)


class BoundingBoxGroup(BaseBox):
    def __init__(self, boxes: List[BoundingBox]=None):
        super().__init__()
        if boxes is None or not boxes:
            self.min = None
            self.max = None
            self.score = None
            self.cls = None
        else:
            self.min = np.array([b.min for b in boxes])
            self.max = np.array([b.max for b in boxes])
            self.score = np.array([b.score for b in boxes])
            self.cls = np.array([b.cls for b in boxes])

    def append(self, box: BoundingBox):
        self.min = np.append(self.min, [box.min], axis=0)
        self.max = np.append(self.max, [box.max], axis=0)
        self.score = np.append(self.score, [box.score], axis=0)
        self.cls = np.append(self.cls, [box.cls], axis=0)

    def remove(self, box_index):
        if not isinstance(box_index, list):
            box_index = [box_index, ]
        if self.size() == len(box_index):
            self.max = None
            self.min = None
            self.score = None
            self.cls = None
        else:
            self.max = np.delete(self.max, box_index, 0)
            self.min = np.delete(self.min, box_index, 0)
            self.score = np.delete(self.score, box_index, 0)
            self.cls = np.delete(self.cls, box_index, 0)

    def size(self):
        return self.score.shape[0] if self.score is not None else 0

    def index(self, box_index):
        return BoundingBox(np.append(np.array(self.min[box_index]), self.max[box_index]), self.score[box_index],
                           self.cls[box_index])

    def resort_by_score(self, asc=True):
        resort = np.argsort(self.score, axis=0)
        if asc:
            resort = resort[::-1, ...]
        self.min = self.min[resort, ...]
        self.max = self.max[resort, ...]
        self.score = self.score[resort]
        self.cls = self.cls[resort]

    def filter_by_cls(self, cls):
        cls_id = np.argwhere(self.cls == cls)
        boxes_list = list()
        for c_id in cls_id:
            boxes_list.append(self.index(c_id))
        return BoundingBoxGroup(boxes_list)


def compute_bounding_box_iou_broadcast(boxes_a: BaseBox, box_b: BoundingBox, as_iou=True) -> np.array:
    """
    Compute the IoU among target bounding box b and group bounding boxes a

    Args:
        boxes_a: a group of bounding boxes
        box_b: a bounding box
        as_iou: whether to return the results as IoU

    Returns:
        IoU between each box in group a and target box b

    Notes:
        Maintained by: Yu-Xiao Guo
        Last modified by: Yu-Xiao Guo
        Last used by: Yu-Xiao Guo
    """
    min_max = np.maximum(boxes_a.min, box_b.min)
    max_min = np.minimum(boxes_a.max, box_b.max)
    intersect = np.maximum(max_min - min_max, 0.)
    intersect_area = np.prod(intersect, axis=-1)
    if as_iou:
        boxes_iou = intersect_area / (-intersect_area + boxes_a.box_area() + box_b.box_area())
        return boxes_iou
    else:
        t_ab = intersect_area
        f_a = boxes_a.box_area() - intersect_area
        f_b = box_b.box_area() - intersect_area
        return np.stack([t_ab, f_b, f_a], axis=-1)


def non_maximum_suppression(boxes: BoundingBoxGroup, threshold):
    """
    Non maximum suppression. First, sort the boxes by its scores; Then, iteratively add the box by
        by comparing its IoU with all passed boxes if maximum IoU smaller than the threshold;

    Args:
        boxes: the boxes to be applied NMS
        threshold: the threshold to reject later box added into passed boxes

    Returns:
        the boxes passed NMS

    Notes:
        Maintained by: Yu-Xiao Guo
        Last modified by: Yu-Xiao Guo
        Last used by: Yu-Xiao Guo
    """
    resort_boxes = deepcopy(boxes)
    if not resort_boxes.size():
        return BoundingBoxGroup()
    resort_boxes.resort_by_score()
    nms_boxes = BoundingBoxGroup([resort_boxes.index(0)])
    for i in range(1, resort_boxes.size()):
        exist_iou = compute_bounding_box_iou_broadcast(nms_boxes, resort_boxes.index(i))
        if np.all(exist_iou < threshold):
            nms_boxes.append(resort_boxes.index(i))
    return nms_boxes


def group_iou_across_classes(gt_boxes: BoundingBoxGroup, pt_boxes: BoundingBoxGroup, as_iou=False):
    """
    Compute the IoU between two boxes group set. For each class in each round, iteratively select a box from gt_boxes
        and pick corresponding maximum IoU box in pt_boxes as pairs. After finished, compute IoU between pairs. The rest
        boxes in gt_boxes will be the false positive, while the rest boxes in pt_boxes will be the false negative.

    Args:
        gt_boxes: ground truth boxes
        pt_boxes: predict boxes
        as_iou: whether to return the result as IoU, or separately returning tp, fp, np

    Returns:
        If `as_iou` is True, return per-class IoU, else returning true positive, false positive, false negative instead

    Notes:
        Maintained by: Yu-Xiao Guo
        Last modified by: Yu-Xiao Guo
        Last used by: Yu-Xiao Guo
    """
    gt_cmp_boxes = deepcopy(gt_boxes)
    pt_cmp_boxes = deepcopy(pt_boxes)
    pair_mask = list()
    for gt_id in range(gt_cmp_boxes.size()):
        if not pt_cmp_boxes.size():
            continue
        cmp_iou = compute_bounding_box_iou_broadcast(pt_cmp_boxes, gt_cmp_boxes.index(gt_id), as_iou=True)
        cmp_max = np.argmax(cmp_iou)
        if cmp_iou[cmp_max] > 0:
            pair_mask.append([gt_id, cmp_max])
    t_ps = 0.
    f_ps = 0.
    n_ps = 0.
    for p_m in pair_mask:
        t_p, f_p, n_p = compute_bounding_box_iou_broadcast(pt_cmp_boxes.index(p_m[1]), gt_cmp_boxes.index(p_m[0]),
                                                           False)
        t_ps += t_p
        f_ps += f_p
        n_ps += n_p
    masked_gt = [p_m[0] for p_m in pair_mask]
    gt_cmp_boxes.remove(masked_gt)
    f_ps += np.sum(gt_cmp_boxes.box_area(), axis=0)
    masked_pt = [p_m[1] for p_m in pair_mask]
    pt_cmp_boxes.remove(masked_pt)
    n_ps += np.sum(pt_cmp_boxes.box_area(), axis=0)
    return t_ps, f_ps, n_ps


def mask_vol_with_boxes(vol_size: List, boxes: BoundingBoxGroup, vol_stride=1, vol_offset=0):
    vol = np.zeros(vol_size, np.int32)
    vol_meshgrid = volume.mesh_grid_nd(vox_size=vol_size)
    for box_id in range(boxes.size()):
        box = boxes.index(box_id)
        max_margin = (box.max + vol_offset) / vol_stride
        min_margin = (box.min + vol_offset) / vol_stride
        in_margin = np.logical_and(np.all(vol_meshgrid < max_margin, axis=-1),
                                   np.all(vol_meshgrid > min_margin, axis=-1))
        vol = np.where(in_margin, np.ones_like(vol) * box.cls, vol)
    return vol

