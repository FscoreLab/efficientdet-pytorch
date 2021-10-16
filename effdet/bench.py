""" PyTorch EfficientDet support benches

Hacked together by Ross Wightman
"""
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn

from .anchors import AnchorLabeler, Anchors, generate_detections
from .loss import DetectionLoss
from .loss import _sample_outputs as sample_loss


def _sample_outputs(
    outputs: List[torch.Tensor], num_gmm: int, predict_uncertainties: bool
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    uncertainties_aleatoric = []
    uncertainties_epistemic = []
    for level in range(len(outputs)):
        mean, var, weights = torch.tensor_split(outputs[level].permute(0, 2, 3, 1), 3, dim=-1)
        var = torch.sigmoid(var)
        weights = einops.rearrange(weights, "b h w (c k) -> b h w c k", k=num_gmm)
        weights = torch.softmax(weights, dim=-1)
        weights = einops.rearrange(weights, "b h w c k -> b h w (c k)", k=num_gmm)
        weighted_mean = einops.reduce(weights * mean, "b h w (c k) -> b h w c", "sum", k=num_gmm)
        outputs[level] = weighted_mean

        if predict_uncertainties:
            uncertainty_aleatoric = einops.reduce(weights * var, "b h w (c k) -> b h w c", "sum", k=num_gmm)
            uncertainties_aleatoric.append(uncertainty_aleatoric)

            weighted_mean_repeated = einops.repeat(weighted_mean, "b h w c -> b h w (c k)", k=num_gmm)
            mean_diff = (mean - weighted_mean_repeated).square()
            uncertainty_epistemic = einops.reduce(weights * mean_diff, "b h w (c k) -> b h w c", "sum", k=num_gmm)
            uncertainties_epistemic.append(uncertainty_epistemic)
        else:
            uncertainties_aleatoric.append(None)
            uncertainties_epistemic.append(None)

    return outputs, uncertainties_aleatoric, uncertainties_epistemic


def _cat_outputs(outputs: List[torch.Tensor], batch_size: int, last_dim_size: int):
    return torch.cat([output_level.reshape([batch_size, -1, last_dim_size]) for output_level in outputs], 1)


def _post_process_uncertainties(
    uncertainties: List[Optional[torch.Tensor]],
    batch_size: int,
    last_dim_size: int,
    gather_func: Callable[[torch.Tensor], torch.Tensor],
) -> List[Optional[torch.Tensor]]:
    if uncertainties[0] is None:
        return [None] * batch_size
    uncertainties_all = _cat_outputs(uncertainties, batch_size, last_dim_size)
    uncertainties_all_after_topk = gather_func(uncertainties_all)
    uncertainties_reduced = einops.reduce(uncertainties_all_after_topk, "b n k -> b n 1", "max")
    return uncertainties_reduced


def _post_process(
    cls_outputs: List[torch.Tensor],
    box_outputs: List[torch.Tensor],
    num_levels: int,
    num_classes: int,
    num_gmm: int,
    predict_uncertainties: bool,
    max_detection_points: int = 5000,
):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].

        num_levels (int): number of feature levels

        num_classes (int): number of output classes
    """

    def _gather_box_outputs(outputs: torch.Tensor, indices_all: torch.Tensor):
        return torch.gather(outputs, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    def _gather_cls_outputs(
        outputs: torch.Tensor, indices_all: torch.Tensor, classes_all: torch.Tensor, num_classes: int
    ):
        outputs_after_topk = torch.gather(outputs, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes))
        return torch.gather(outputs_after_topk, 2, classes_all.unsqueeze(2))

    batch_size = cls_outputs[0].shape[0]

    cls_outputs, cls_uncertainties_aleatoric, cls_uncertainties_epistemic = _sample_outputs(
        cls_outputs, num_gmm, predict_uncertainties
    )
    cls_outputs_all = _cat_outputs(cls_outputs, batch_size, num_classes)

    box_outputs, box_uncertainties_aleatoric, box_uncertainties_epistemic = _sample_outputs(
        box_outputs, num_gmm, predict_uncertainties
    )
    box_outputs_all = _cat_outputs(box_outputs, batch_size, 4)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points)
    indices_all = cls_topk_indices_all // num_classes
    classes_all = cls_topk_indices_all % num_classes

    box_outputs_all_after_topk = _gather_box_outputs(box_outputs_all, indices_all)
    box_uncertainties_aleatoric_all_after_topk = _post_process_uncertainties(
        box_uncertainties_aleatoric, batch_size, 4, partial(_gather_box_outputs, indices_all=indices_all)
    )
    box_uncertainties_epistemic_all_after_topk = _post_process_uncertainties(
        box_uncertainties_epistemic, batch_size, 4, partial(_gather_box_outputs, indices_all=indices_all)
    )

    cls_outputs_all_after_topk = _gather_cls_outputs(cls_outputs_all, indices_all, classes_all, num_classes)
    cls_uncertainties_aleatoric_all_after_topk = _post_process_uncertainties(
        cls_uncertainties_aleatoric,
        batch_size,
        num_classes,
        partial(_gather_cls_outputs, indices_all=indices_all, classes_all=classes_all, num_classes=num_classes),
    )
    cls_uncertainties_epistemic_all_after_topk = _post_process_uncertainties(
        cls_uncertainties_epistemic,
        batch_size,
        num_classes,
        partial(_gather_cls_outputs, indices_all=indices_all, classes_all=classes_all, num_classes=num_classes),
    )

    return (
        cls_outputs_all_after_topk,
        cls_uncertainties_aleatoric_all_after_topk,
        cls_uncertainties_epistemic_all_after_topk,
        box_outputs_all_after_topk,
        box_uncertainties_aleatoric_all_after_topk,
        box_uncertainties_epistemic_all_after_topk,
        indices_all,
        classes_all,
    )


@torch.jit.script
def _batch_detection(
    batch_size: int,
    class_out,
    class_uncertainties_aleatoric: List[Optional[torch.Tensor]],
    class_uncertainties_epistemic: List[Optional[torch.Tensor]],
    box_out,
    box_uncertainties_aleatoric: List[Optional[torch.Tensor]],
    box_uncertainties_epistemic: List[Optional[torch.Tensor]],
    anchor_boxes,
    indices,
    classes,
    img_scale: Optional[torch.Tensor] = None,
    img_size: Optional[torch.Tensor] = None,
    max_det_per_image: int = 100,
    soft_nms: bool = False,
):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        img_scale_i = None if img_scale is None else img_scale[i]
        img_size_i = None if img_size is None else img_size[i]
        detections = generate_detections(
            class_out[i],
            class_uncertainties_aleatoric[i],
            class_uncertainties_epistemic[i],
            box_out[i],
            box_uncertainties_aleatoric[i],
            box_uncertainties_epistemic[i],
            anchor_boxes,
            indices[i],
            classes[i],
            img_scale_i,
            img_size_i,
            max_det_per_image=max_det_per_image,
            soft_nms=soft_nms,
        )
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


class DetBenchPredict(nn.Module):
    def __init__(self, model, predict_uncertainties=False):
        super(DetBenchPredict, self).__init__()
        self.model = model
        self.config = model.config  # FIXME remove this when we can use @property (torchscript limitation)
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.num_gmm = model.config.gaussian_count
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms
        self.predict_uncertainties = predict_uncertainties

    def forward(self, x, img_info: Optional[Dict[str, torch.Tensor]] = None):
        class_out, box_out = self.model(x)
        class_out, cls_un_al, cls_un_ep, box_out, box_un_al, box_un_ep, indices, classes = _post_process(
            class_out,
            box_out,
            num_levels=self.num_levels,
            num_classes=self.num_classes,
            max_detection_points=self.max_detection_points,
            num_gmm=self.num_gmm,
            predict_uncertainties=self.predict_uncertainties,
        )
        if img_info is None:
            img_scale, img_size = None, None
        else:
            img_scale, img_size = img_info["img_scale"], img_info["img_size"]
        return _batch_detection(
            x.shape[0],
            class_out,
            cls_un_al,
            cls_un_ep,
            box_out,
            box_un_al,
            box_un_ep,
            self.anchors.boxes,
            indices,
            classes,
            img_scale,
            img_size,
            max_det_per_image=self.max_det_per_image,
            soft_nms=self.soft_nms,
        )


class DetBenchTrain(nn.Module):
    def __init__(self, model, create_labeler=True, predict_uncertainties=False, use_pred_boxes=True, class_weight=0.1):
        super(DetBenchTrain, self).__init__()
        self.model = model
        self.config = model.config  # FIXME remove this when we can use @property (torchscript limitation)
        self.num_levels = model.config.num_levels
        self.num_classes = model.config.num_classes
        self.num_gmm = model.config.gaussian_count
        self.anchors = Anchors.from_config(model.config)
        self.max_detection_points = model.config.max_detection_points
        self.max_det_per_image = model.config.max_det_per_image
        self.soft_nms = model.config.soft_nms
        self.anchor_labeler = None
        if create_labeler:
            self.anchor_labeler = AnchorLabeler(
                self.anchors,
                self.num_classes,
                match_threshold=None,
                use_pred_boxes=use_pred_boxes,
                class_weight=class_weight,
                congig=model.config,
            )
        self.loss_fn = DetectionLoss(model.config)
        self.predict_uncertainties = predict_uncertainties

    def forward(self, x, target: Dict[str, torch.Tensor]):
        class_out, box_out = self.model(x)

        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert "label_num_positives" in target
            cls_targets = [target[f"label_cls_{l}"] for l in range(self.num_levels)]
            box_targets = [target[f"label_bbox_{l}"] for l in range(self.num_levels)]
            num_positives = target["label_num_positives"]
        else:
            cls_output = [
                sample_loss(einops.rearrange(layer.detach(), "b c h w -> b h w c"), self.num_gmm, std_weight=0)
                for layer in class_out
            ]
            box_output = [
                sample_loss(einops.rearrange(layer.detach(), "b c h w -> b h w c"), self.num_gmm, std_weight=0)
                for layer in box_out
            ]
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target["bbox"], target["cls"], cls_output, box_output
            )

        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
        output = {"loss": loss, "class_loss": class_loss, "box_loss": box_loss}
        if not self.training:
            # if eval mode, output detections for evaluation
            class_out_pp, cls_un_al, cls_un_ep, box_out_pp, box_un_al, box_un_ep, indices, classes = _post_process(
                class_out,
                box_out,
                num_levels=self.num_levels,
                num_classes=self.num_classes,
                max_detection_points=self.max_detection_points,
                num_gmm=self.num_gmm,
                predict_uncertainties=self.predict_uncertainties,
            )
            output["detections"] = _batch_detection(
                x.shape[0],
                class_out_pp,
                cls_un_al,
                cls_un_ep,
                box_out_pp,
                box_un_al,
                box_un_ep,
                self.anchors.boxes,
                indices,
                classes,
                target["img_scale"],
                target["img_size"],
                max_det_per_image=self.max_det_per_image,
                soft_nms=self.soft_nms,
            )
        return output


def unwrap_bench(model):
    # Unwrap a model in support bench so that various other fns can access the weights and attribs of the
    # underlying model directly
    if hasattr(model, "module"):  # unwrap DDP or EMA
        return unwrap_bench(model.module)
    elif hasattr(model, "model"):  # unwrap Bench -> model
        return unwrap_bench(model.model)
    else:
        return model
