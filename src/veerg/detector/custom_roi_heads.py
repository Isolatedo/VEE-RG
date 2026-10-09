from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops


def _select_top1(scores):
    """Select one proposal for each anatomical region."""
    region_scores = scores[:, 1:]
    classes = torch.argmax(region_scores, dim=1)
    class_mask = F.one_hot(classes, num_classes=region_scores.shape[1])
    top_scores, indices = torch.max(region_scores * class_mask, dim=0)
    detected = class_mask.sum(dim=0) > 0
    return top_scores, indices, detected


def _select_class_specific_top1(scores):
    """Select the best proposal for every region class, ignoring assignment."""
    region_scores = scores[:, 1:]
    return torch.max(region_scores, dim=0)


def _gather_class_boxes(boxes, indices):
    if indices.ndim != 1:
        raise ValueError("indices must have shape [R]")
    classes = torch.arange(
        indices.shape[0], dtype=torch.int64, device=indices.device
    )
    return boxes[indices, classes]


class CustomRoIHeads(RoIHeads):
    def __init__(
        self,
        return_feature_vectors,
        feature_map_output_size,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )
        # return_feature_vectors == True if we train/evaluate the object detector as part of the full model
        self.return_feature_vectors = return_feature_vectors

        # set kernel_size = feature_map_output_size, such that we average over the whole feature maps
        self.avg_pool = nn.AvgPool2d(kernel_size=feature_map_output_size)
        self.dim_reduction = nn.Linear(2048, 1024)

    def get_top_region_features_detections_class_detected(
        self,
        box_features,
        box_regression,
        class_logits,
        proposals,
        image_shapes
    ):
        """Collect the original top-1 ROI and class-specific box for every region."""
        # apply softmax on background class as well
        # (such that if the background class has a high score, all other classes will have a low score)
        pred_scores = F.softmax(class_logits, -1)

        # get number of proposals/boxes per image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        num_images = len(boxes_per_image)

        # split pred_scores (which is a tensor with scores for all RoIs of all images in the batch)
        # into the tuple pred_scores_per_img (where 1 pred_score tensor has scores for all RoIs of 1 image)
        pred_scores_per_img = torch.split(pred_scores, boxes_per_image, dim=0)

        # if we train/evaluate the full model, we need the top region/box features
        if self.return_feature_vectors:
            # split region_features the same way as pred_scores
            region_features_per_img = torch.split(box_features, boxes_per_image, dim=0)
        else:
            region_features_per_img = [None] * num_images  # dummy list such that we can still zip everything up

        # if we evaluate the object detector, we need the detections
        if not self.training:
            pred_region_boxes = self.box_coder.decode(box_regression, proposals)
            pred_region_boxes_per_img = torch.split(pred_region_boxes, boxes_per_image, dim=0)
        else:
            pred_region_boxes_per_img = [None] * num_images  # dummy list such that we can still zip everything up

        output = {}
        output["class_detected"] = []  # list collects the bool arrays of shape [29] that specify if a class was detected (True) for each image
        output["top_region_features"] = []  # list collects the tensors of shape [29 x 2048] of the top region features for each image

        output["detections"] = {
            "top_region_boxes": [],
            "top_scores": [],
            "pseudo_region_boxes": [],
            "pseudo_scores": [],
        }

        for pred_scores_img, pred_region_boxes_img, region_features_img, img_shape in zip(pred_scores_per_img, pred_region_boxes_per_img, region_features_per_img, image_shapes):
            top_scores, indices_with_top_scores, class_detected = _select_top1(
                pred_scores_img
            )
            pseudo_scores, pseudo_indices = _select_class_specific_top1(pred_scores_img)

            output["class_detected"].append(class_detected)

            if self.return_feature_vectors:
                top_region_features = region_features_img[indices_with_top_scores]
                output["top_region_features"].append(top_region_features)

            if not self.training:
                # pred_region_boxes_img is of shape [num_boxes_in_image x 30 x 4]

                # clip boxes so that they lie inside an image of size "img_shape"
                pred_region_boxes_img = box_ops.clip_boxes_to_image(pred_region_boxes_img, img_shape)

                # remove predictions with the background label
                # pred_region_boxes_img is now of shape [num_boxes_in_image x 29 x 4]
                pred_region_boxes_img = pred_region_boxes_img[:, 1:]

                top_region_boxes = _gather_class_boxes(
                    pred_region_boxes_img, indices_with_top_scores
                )
                pseudo_region_boxes = _gather_class_boxes(
                    pred_region_boxes_img, pseudo_indices
                )

                output["detections"]["top_region_boxes"].append(top_region_boxes)
                output["detections"]["top_scores"].append(top_scores)
                output["detections"]["pseudo_region_boxes"].append(pseudo_region_boxes)
                output["detections"]["pseudo_scores"].append(pseudo_scores)

        # convert lists into batched tensors
        output["class_detected"] = torch.stack(output["class_detected"], dim=0)  # of shape [batch_size x 29]

        if self.return_feature_vectors:
            output["top_region_features"] = torch.stack(output["top_region_features"], dim=0)  # of shape [batch_size x 29 x 2048]

        if not self.training:
            output["detections"]["top_region_boxes"] = torch.stack(output["detections"]["top_region_boxes"], dim=0)  # of shape [batch_size x 29 x 4]
            output["detections"]["top_scores"] = torch.stack(output["detections"]["top_scores"], dim=0)  # of shape [batch_size x 29]
            output["detections"]["pseudo_region_boxes"] = torch.stack(output["detections"]["pseudo_region_boxes"], dim=0)
            output["detections"]["pseudo_scores"] = torch.stack(output["detections"]["pseudo_scores"], dim=0)

        return output

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError("target labels must of int64 type, instead got {t['labels'].dtype}")

        if targets is not None:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # box_roi_pool_feature_maps has shape [overall_num_proposals_for_all_images x 2048 x 8 x 8]
        box_roi_pool_feature_maps = self.box_roi_pool(features, proposals, image_shapes)

        # box_feature_vectors has shape [overall_num_proposals_for_all_images x 1024]
        box_feature_vectors = self.box_head(box_roi_pool_feature_maps)
        class_logits, box_regression = self.box_predictor(box_feature_vectors)

        detector_losses = {}

        if labels and regression_targets:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        # we always return the detector_losses (even if it's an empty dict, which is the case for targets==None (i.e. during inference))
        roi_heads_output = {}
        roi_heads_output["detector_losses"] = detector_losses

        # if we train the full model (i.e. self.return_feature_vectors == True), we need the "top_region_features"
        # if we evaluate the object detector (in isolation or as part of the full model), we need the "detections"
        # if we do either of them, we always need "class_detected" (see doc_string of method for details)
        if self.return_feature_vectors or not self.training:
            # average over the spatial dimensions, i.e. transform roi pooling features maps from [num_proposals, 2048, 8, 8] to [num_proposals, 2048, 1, 1]
            box_features = self.avg_pool(box_roi_pool_feature_maps)

            box_features = torch.flatten(box_features, start_dim=1)

            output = self.get_top_region_features_detections_class_detected(box_features, box_regression, class_logits, proposals, image_shapes)

            roi_heads_output["class_detected"] = output["class_detected"]

            if self.return_feature_vectors:
                # transform top_region_features from [batch_size x 29 x 2048] to [batch_size x 29 x 1024]
                roi_heads_output["top_region_features"] = self.dim_reduction(output["top_region_features"])

            if not self.training:
                roi_heads_output["detections"] = output["detections"]

        return roi_heads_output
