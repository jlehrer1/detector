from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch 
from pytorch_lightning import LightningModule
import torch.nn as nn 
import torchvision
from utils import SetCriterion, HungarianMatcher
from torch.optim import AdamW
from torch import Tensor 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import numpy as np
import tempfile
import os
from utils import box_cxcywh_to_xyxy
import uuid

class DETR(LightningModule):
    LR = 1e-4
    LR_BACKBONE = 1e-5
    BATCH_SIZE = 2
    WEIGHT_DECAY = 1e-4
    EPOCHS = 300
    LR_DROP = 200
    CLIP_MAX_NORM = 0.1

    # Model parameters
    FROZEN_WEIGHTS = None

    # Backbone
    BACKBONE = 'resnet50'
    DILATION = True
    POSITION_EMBEDDING = 'sine'

    # Transformer
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    HIDDEN_DIM = 256
    DROPOUT = 0.1
    NHEADS = 8
    NUM_QUERIES = 100
    PRE_NORM = True

    # Segmentation
    MASKS = True

    # Loss
    AUX_LOSS = True

    # Matcher
    SET_COST_CLASS = 1
    SET_COST_BBOX = 5
    SET_COST_GIOU = 2

    # Loss coefficients
    MASK_LOSS_COEF = 1
    DICE_LOSS_COEF = 1
    BBOX_LOSS_COEF = 5
    GIOU_LOSS_COEF = 2
    EOS_COEF = 0.1

    def __init__(self, num_classes: int):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
        # self.base_model.class_embed = nn.Linear(in_features=self.base_model.class_embed.in_features, out_features=num_classes + 1) # for no class

        matcher = HungarianMatcher(cost_class=self.SET_COST_CLASS, cost_bbox=self.SET_COST_BBOX, cost_giou=self.SET_COST_GIOU)
        self.loss = SetCriterion(
            num_classes=self.base_model.class_embed.out_features - 1, 
            matcher=matcher, 
            weight_dict={'loss_ce': 1, 'loss_bbox': self.BBOX_LOSS_COEF, 'loss_giou': self.GIOU_LOSS_COEF}, 
            eos_coef=self.EOS_COEF, 
            losses=['labels', 'boxes', 'cardinality']
        )

    def forward(self, x):
        return self.base_model(x)

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.LR_DROP)

        return [optimizer], [lr_scheduler]
    
    def _step(self, batch):
        images, targets = batch

        outputs = self(images)
        loss_dict = self.loss(outputs, targets)

        weight_dict = self.loss.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        print("Loss result is", losses)
        return {"loss": losses, "outputs": outputs, "targets": targets}

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self._step(batch)
        self.logger.experiment.log({"train_loss": outputs["loss"]})

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self._step(batch)
        self.log("val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        box_results = self.postprocess(outputs["outputs"], outputs["targets"])

        # log image with ground truth box and prediction box to wandb
        # randomly sample 1 image from the batch
        idx = np.random.randint(0, len(outputs["targets"]))

        print(f"Softmax probabilities for rand image are\n", outputs["outputs"]["pred_logits"][idx].softmax(-1).max(-1).values)
        self.plot_and_log_bounding_boxes(
            image=outputs["targets"][idx]["original_image"],
            predicted_bboxes=box_results["pred_boxes"][idx],
            predicted_probabilities=outputs["outputs"]["pred_logits"][idx].softmax(-1).max(-1).values,
            ground_truth_bbox=outputs["targets"][idx]["boxes"],
        )
        return outputs["loss"]

    @torch.no_grad()
    def postprocess(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]):
        """Takes the output of the model and the targets."""
        logits = outputs["pred_logits"]
        bboxes = outputs["pred_boxes"]

        target_sizes = torch.stack([x["size"] for x in targets])
        assert len(logits) == len(target_sizes)

        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)

        # convert predicted boxes and ground truth boxes back to x1y1x2y2 format
        bboxes = box_cxcywh_to_xyxy(bboxes)

        # resize to original image size
        img_w, img_h = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = bboxes * scale_fct[:, None, :]

        return {"pred_boxes": boxes, "pred_labels": labels, "pred_probabilities": scores}

    def plot_and_log_bounding_boxes(self, image, predicted_bboxes, predicted_probabilities, ground_truth_bbox):
        keep = predicted_probabilities > 0.9
        predicted_bboxes = predicted_bboxes[keep]
        predicted_probabilities = predicted_probabilities[keep]

        w, h = image.size
        ground_truth_bbox = box_cxcywh_to_xyxy(ground_truth_bbox) * torch.as_tensor([w, h, w, h], dtype=torch.float32)

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.imshow(image)

        if len(predicted_bboxes) > 0:  # with all low confidence scores we'll have no BB
            for pred_box in predicted_bboxes:
                print("predicted boxes are")
                print(pred_box)
                pred_rect = patches.Rectangle(
                    (pred_box[0], pred_box[1]),
                    pred_box[2] - pred_box[0],
                    pred_box[3] - pred_box[1],
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                )
                ax.add_patch(pred_rect)

        # Create a green rectangle for the ground truth bounding box
        for gt_box in ground_truth_bbox:
            gt_rect = patches.Rectangle(
                (gt_box[0], gt_box[1]),
                gt_box[2] - gt_box[0],
                gt_box[3] - gt_box[1],
                linewidth=2,
                edgecolor='g',
                facecolor='none',
            )

            ax.add_patch(gt_rect)
        
        plt.title('Bounding Box Comparison')
        plt.savefig(f"comparison_{str(uuid.uuid4())[0:5]}.png", format="png")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format="png")
        
        wandb.log({"rand_val_image": wandb.Image(tmpfile.name)})
        os.remove(tmpfile.name)
