""" RetinaNet / EfficientDet Anchor Gen

Adapted for PyTorch from Tensorflow impl at
    https://github.com/google/automl/blob/6f6694cec1a48cdb33d5d1551a2d5db8ad227798/efficientdet/anchors.py

Hacked together by Ross Wightman, original copyright below
"""
# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Anchor definition.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
#import torchvision.ops.boxes as tvb
from torchvision.ops.boxes import batched_nms, remove_small_boxes
from typing import List

from effdet.object_detection import ArgMaxMatcher, FasterRcnnBoxCoder, BoxList, IouSimilarity, TargetAssigner
from .soft_nms import batched_soft_nms
from collections import defaultdict


# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5


def decode_box_outputs(rel_codes, anchors, output_xyxy: bool=False):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.

    """
    ycenter_a = (anchors[:, 0] + anchors[:, 2]) / 2
    xcenter_a = (anchors[:, 1] + anchors[:, 3]) / 2
    ha = anchors[:, 2] - anchors[:, 0]
    wa = anchors[:, 3] - anchors[:, 1]

    ty, tx, th, tw = rel_codes.unbind(dim=1)

    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    if output_xyxy:
        out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    return out


def clip_boxes_xyxy(boxes: torch.Tensor, size: torch.Tensor):
    boxes = boxes.clamp(min=0)
    size = torch.cat([size, size], dim=0)
    boxes = boxes.min(size)
    return boxes


def assign_boxes_to_classes(bounding_boxes, classes, scores):
    """
    Parameters:
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
    Returns:
       boxes_to_classes: defaultdict(list) containing mapping to bounding boxes and confidence scores to class
    """
    boxes_to_classes = defaultdict(list)
    for each_box, each_class, each_score in zip(bounding_boxes, classes, scores):
        if each_score >= 0.05:
            boxes_to_classes[each_class].append(
                np.array([each_box[0], each_box[1], each_box[2], each_box[3], each_score]))
    return boxes_to_classes


def normalise_coordinates(x1, y1, x2, y2, min_x, max_x, min_y, max_y):
    """
    Parameters:
       x1, y1, x2, y2: bounding box coordinates to normalise
       min_x,max_x,min_y,max_y: minimum and maximum bounding box values (min = 0, max = 1)
    Returns:
       Normalised bounding box coordinates (scaled between 0 and 1)
    """
    x1, y1, x2, y2 = (x1 - min_x) / (max_x - min_x), (y1 - min_y) / (max_y - min_y), (x2 - min_x) / (max_x - min_x), (
                y2 - min_y) / (max_y - min_y)
    return x1, y1, x2, y2


def confluence_nms(bounding_boxes, scores, classes, confluence_thr, gaussian, score_thr=0.05, sigma=0.5):
    """
    Parameters:
       bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
       classes: list of class identifiers (int value, e.g. 1 = person)
       scores: list of class confidence scores (0.0-1.0)
       confluence_thr: value between 0 and 2, with optimum from 0.5-0.8
       gaussian: boolean switch to turn gaussian decaying of suboptimal bounding box confidence scores (setting to False results in suppression of suboptimal boxes)
       score_thr: class confidence score
       sigma: used in gaussian decaying. A smaller value causes harsher decaying.
    Returns:
       output: A dictionary mapping class identity to final retained boxes (and corresponding confidence scores)
    """

    class_mapping = assign_boxes_to_classes(bounding_boxes, classes, scores)
    output = {}
    for each_class in class_mapping:
        dets = np.array(class_mapping[each_class])
        retain = []
        while dets.size > 0:
            max_idx = np.argmax(dets[:, 4], axis=0)
            dets[[0, max_idx], :] = dets[[max_idx, 0], :]
            retain.append(dets[0, :])
            x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]

            min_x = np.minimum(x1, dets[1:, 0])
            min_y = np.minimum(y1, dets[1:, 1])
            max_x = np.maximum(x2, dets[1:, 2])
            max_y = np.maximum(y2, dets[1:, 3])

            x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2, min_x, max_x, min_y, max_y)
            xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3], min_x, max_x,
                                                       min_y, max_y)

            md_x1, md_x2, md_y1, md_y2 = abs(x1 - xx1), abs(x2 - xx2), abs(y1 - yy1), abs(y2 - yy2)
            manhattan_distance = (md_x1 + md_x2 + md_y1 + md_y2)

            weights = np.ones_like(manhattan_distance)

            if (gaussian == True):
                gaussian_weights = np.exp(-((1 - manhattan_distance) * (1 - manhattan_distance)) / sigma)
                weights[manhattan_distance <= confluence_thr] = gaussian_weights[manhattan_distance <= confluence_thr]
            else:
                weights[manhattan_distance <= confluence_thr] = manhattan_distance[manhattan_distance <= confluence_thr]

            dets[1:, 4] *= weights
            to_reprocess = np.where(dets[1:, 4] >= score_thr)[0]
            dets = dets[to_reprocess + 1, :]
        output[each_class] = retain

    return output

def generate_detections(
        cls_outputs,
        cls_uncertainty_al: Optional[torch.Tensor],
        cls_uncertainty_ep: Optional[torch.Tensor],
        box_outputs,
        box_uncertainty_al: Optional[torch.Tensor],
        box_uncertainty_ep: Optional[torch.Tensor],
        anchor_boxes, indices, classes,
        img_scale: Optional[torch.Tensor], img_size: Optional[torch.Tensor],
        max_det_per_image: int = 100, soft_nms: bool = False, confluence=False,
        iou_threshold: float = 0.5,
        confluence_thr: float = 0.5,
        confluence_gaussian: bool = True,
        confluence_sigma: float = 0.5,
        confluence_score_thr: float = 0.05,
):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels.

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [max_det_per_image, 6],
            each row representing [x_min, y_min, x_max, y_max, score, class, box_uncertainty_al, box_uncertainty_ep, cls_uncertainty_al, cls_uncertainty_ep]
    """
    assert box_outputs.shape[-1] == 4
    assert anchor_boxes.shape[-1] == 4
    assert cls_outputs.shape[-1] == 1

    anchor_boxes = anchor_boxes[indices, :]

    # Appply bounding box regression to anchors, boxes are converted to xyxy
    # here since PyTorch NMS expects them in that form.
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes, output_xyxy=True)
    if img_scale is not None and img_size is not None:
        boxes = clip_boxes_xyxy(boxes, img_size / img_scale)  # clip before NMS better?

    scores = cls_outputs.sigmoid().squeeze(1).float()
    if soft_nms:
        top_detection_idx, soft_scores = batched_soft_nms(
            boxes, scores, classes, method_gaussian=True, iou_threshold=0.3, score_threshold=.001)
        scores[top_detection_idx] = soft_scores
    if not soft_nms and confluence:
        top_detection_idx = confluence_nms(boxes.tolist(), scores.tolist(), classes.tolist(), gaussian=confluence_gaussian,
                                           confluence_thr=confluence_thr, sigma=confluence_sigma, score_thr=confluence_score_thr)
        top_detection_idx = np.array(top_detection_idx[0])
        top_detection_idx = top_detection_idx[top_detection_idx[:, 4].argsort()[::-1]]
        top_detection_idx = torch.from_numpy(top_detection_idx).cuda()
    else:
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=iou_threshold)

    # keep only top max_det_per_image scoring predictions
    top_detection_idx = top_detection_idx[:max_det_per_image]
    if not confluence:
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None] + 1  # back to class idx with background class = 0
    else:
        boxes = top_detection_idx[:, :4]
        scores = top_detection_idx[:, 4:5]
        classes = torch.ones_like(scores, dtype=torch.int8)

    if cls_uncertainty_al is not None and cls_uncertainty_ep is not None and box_uncertainty_al is not None and box_uncertainty_ep is not None:
        cls_uncertainty_al = cls_uncertainty_al[top_detection_idx]
        cls_uncertainty_ep = cls_uncertainty_ep[top_detection_idx]
        box_uncertainty_al = box_uncertainty_al[top_detection_idx]
        box_uncertainty_ep = box_uncertainty_ep[top_detection_idx]

    if img_scale is not None:
        boxes = boxes * img_scale

    # FIXME add option to convert boxes back to yxyx? Otherwise must be handled downstream if
    # that is the preferred output format.

    # stack em and pad out to max_det_per_image if necessary
    num_det = len(top_detection_idx)
    if cls_uncertainty_al is not None and cls_uncertainty_ep is not None and box_uncertainty_al is not None and box_uncertainty_ep is not None:
        detections = torch.cat([boxes, scores, classes.float(),
                                box_uncertainty_al, box_uncertainty_ep,
                                cls_uncertainty_al, cls_uncertainty_ep], dim=1)
    else:
        detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if num_det < max_det_per_image:
        detections = torch.cat([
            detections,
            torch.zeros((max_det_per_image - num_det, detections.shape[1]), device=detections.device, dtype=detections.dtype)
        ], dim=0)
    return detections


def get_feat_sizes(image_size: Tuple[int, int], max_level: int):
    """Get feat widths and heights for all levels.
    Args:
      image_size: a tuple (H, W)
      max_level: maximum feature level.
    Returns:
      feat_sizes: a list of tuples (height, width) for each level.
    """
    feat_size = image_size
    feat_sizes = [feat_size]
    for _ in range(1, max_level + 1):
        feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
        feat_sizes.append(feat_size)
    return feat_sizes


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size: Tuple[int, int]):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: Sequence specifying input image size of model (H, W).
                The image_size should be divided by the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        if isinstance(anchor_scale, Sequence):
            assert len(anchor_scale) == max_level - min_level + 1
            self.anchor_scales = anchor_scale
        else:
            self.anchor_scales = [anchor_scale] * (max_level - min_level + 1)

        assert isinstance(image_size, Sequence) and len(image_size) == 2
        # FIXME this restriction can likely be relaxed with some additional changes
        assert image_size[0] % 2 ** max_level == 0, 'Image size must be divisible by 2 ** max_level (128)'
        assert image_size[1] % 2 ** max_level == 0, 'Image size must be divisible by 2 ** max_level (128)'
        self.image_size = tuple(image_size)
        self.feat_sizes = get_feat_sizes(image_size, max_level)
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    @classmethod
    def from_config(cls, config):
        return cls(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        anchor_configs = {}
        feat_sizes = self.feat_sizes
        for level in range(self.min_level, self.max_level + 1):
            anchor_configs[level] = []
            for scale_octave in range(self.num_scales):
                for aspect in self.aspect_ratios:
                    anchor_configs[level].append(
                        ((feat_sizes[0][0] // feat_sizes[level][0],
                          feat_sizes[0][1] // feat_sizes[level][1]),
                         scale_octave / float(self.num_scales), aspect,
                         self.anchor_scales[level - self.min_level]))
        return anchor_configs

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes_all = []
        for _, configs in self.config.items():
            boxes_level = []
            for config in configs:
                stride, octave_scale, aspect, anchor_scale = config
                base_anchor_size_x = anchor_scale * stride[1] * 2 ** octave_scale
                base_anchor_size_y = anchor_scale * stride[0] * 2 ** octave_scale
                if isinstance(aspect, Sequence):
                    aspect_x = aspect[0]
                    aspect_y = aspect[1]
                else:
                    aspect_x = np.sqrt(aspect)
                    aspect_y = 1.0 / aspect_x
                anchor_size_x_2 = base_anchor_size_x * aspect_x / 2.0
                anchor_size_y_2 = base_anchor_size_y * aspect_y / 2.0

                x = np.arange(stride[1] / 2, self.image_size[1], stride[1])
                y = np.arange(stride[0] / 2, self.image_size[0], stride[0])
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).float()
        return anchor_boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes.
    """

    def __init__(self, anchors, num_classes: int, match_threshold: float = 0.5):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            anchors: an instance of class Anchors.

            num_classes: integer number representing number of classes in the dataset.

            match_threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        similarity_calc = IouSimilarity()
        matcher = ArgMaxMatcher(
            match_threshold,
            unmatched_threshold=match_threshold,
            negatives_lower_than_unmatched=True,
            force_match_for_each_row=True)
        box_coder = FasterRcnnBoxCoder()

        self.target_assigner = TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.match_threshold = match_threshold
        self.num_classes = num_classes
        self.indices_cache = {}

    def label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        """Labels anchors with ground truth inputs.

        Args:
            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
                For each row, it stores [y0, x0, y1, x1] for four corners of a box.

            gt_classes: A integer tensor with shape [N, 1] representing groundtruth classes.

            filter_valid: Filter out any boxes w/ gt class <= -1 before assigning

        Returns:
            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.

            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at l-th level.

            num_positives: scalar tensor storing number of positives in an image.
        """
        cls_targets_out = []
        box_targets_out = []

        if filter_valid:
            valid_idx = gt_classes > -1  # filter gt targets w/ label <= -1
            gt_boxes = gt_boxes[valid_idx]
            gt_classes = gt_classes[valid_idx]

        cls_targets, box_targets, matches = self.target_assigner.assign(
            BoxList(self.anchors.boxes), BoxList(gt_boxes), gt_classes)

        # class labels start from 1 and the background class = -1
        cls_targets = (cls_targets - 1).long()

        # Unpack labels.
        """Unpacks an array of cls/box into multiple scales."""
        count = 0
        for level in range(self.anchors.min_level, self.anchors.max_level + 1):
            feat_size = self.anchors.feat_sizes[level]
            steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
            cls_targets_out.append(cls_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            box_targets_out.append(box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
            count += steps

        num_positives = (matches.match_results > -1).float().sum()

        return cls_targets_out, box_targets_out, num_positives

    def batch_label_anchors(self, gt_boxes, gt_classes, filter_valid=True):
        batch_size = len(gt_boxes)
        assert batch_size == len(gt_classes)
        num_levels = self.anchors.max_level - self.anchors.min_level + 1
        cls_targets_out = [[] for _ in range(num_levels)]
        box_targets_out = [[] for _ in range(num_levels)]
        num_positives_out = []

        anchor_box_list = BoxList(self.anchors.boxes)
        for i in range(batch_size):
            last_sample = i == batch_size - 1

            if filter_valid:
                valid_idx = gt_classes[i] > -1  # filter gt targets w/ label <= -1
                gt_box_list = BoxList(gt_boxes[i][valid_idx])
                gt_class_i = gt_classes[i][valid_idx]
            else:
                gt_box_list = BoxList(gt_boxes[i])
                gt_class_i = gt_classes[i]
            cls_targets, box_targets, matches = self.target_assigner.assign(anchor_box_list, gt_box_list, gt_class_i)

            # class labels start from 1 and the background class = -1
            cls_targets = (cls_targets - 1).long()

            # Unpack labels.
            """Unpacks an array of cls/box into multiple scales."""
            count = 0
            for level in range(self.anchors.min_level, self.anchors.max_level + 1):
                level_idx = level - self.anchors.min_level
                feat_size = self.anchors.feat_sizes[level]
                steps = feat_size[0] * feat_size[1] * self.anchors.get_anchors_per_location()
                cls_targets_out[level_idx].append(
                    cls_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
                box_targets_out[level_idx].append(
                    box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
                count += steps
                if last_sample:
                    cls_targets_out[level_idx] = torch.stack(cls_targets_out[level_idx])
                    box_targets_out[level_idx] = torch.stack(box_targets_out[level_idx])

            num_positives_out.append((matches.match_results > -1).float().sum())
            if last_sample:
                num_positives_out = torch.stack(num_positives_out)

        return cls_targets_out, box_targets_out, num_positives_out

