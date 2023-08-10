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

        return outputs["loss"]

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self._step(batch)
        self.log("val_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        processed = self.postprocess(outputs["outputs"], outputs["targets"])

        print(f"Outputs shapes are {outputs['outputs']['pred_logits'].shape = } {outputs['outputs']['pred_boxes'].shape = }")

        # log image with ground truth box and prediction box to wandb
        # randomly sample 1 image from the batch

        self.log_random_batch_example(
            pbboxes=processed["pred_boxes"],
            scores=processed["pred_softmax"],
            targets=outputs["targets"],
        )
        return outputs["loss"]

    @torch.no_grad()
    def postprocess(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], nms_threshold=0.5):
        """Takes the output of the model and the targets."""
        logits = outputs["pred_logits"]
        bboxes = outputs["pred_boxes"]

        target_sizes = torch.stack([x["size"] for x in targets]) # (batch_size, 2)
        assert len(logits) == len(target_sizes)

        probs = logits[..., :-1].softmax(-1) # (batch_size, num_queries, num_classes)
        scores, labels = probs.max(-1) # (batch_size, num_queries), (batch_size, num_queries)

        bboxes = box_cxcywh_to_xyxy(bboxes) # (batch_size, num_queries, 4)

        # resize to original image size
        img_w, img_h = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = bboxes * scale_fct[:, None, :]

        return {"pred_boxes": boxes, "pred_labels": labels, "pred_softmax": probs, "pred_scores": scores, "pred_labels": labels}

    def log_random_batch_example(self, pbboxes, scores, targets):
        classes = self.trainer.train_dataloader.dataset.categories

        # take the first image in the batch
        scores = scores[0]
        pbboxes = pbboxes[0]
        image = targets[0]["original_image"]

        # keep only predictions with a score above 0.9
        keep = scores.max(-1).values > 0.9
        pbboxes = pbboxes[keep]
        scores = scores[keep]

        # plot the image with the predicted boxes and labels
        plt.figure(figsize=(16,10))
        plt.imshow(image)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax) in zip(scores, pbboxes.tolist()):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='r', linewidth=3))
            cl = p.argmax()
            text = f'{classes[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig(f"{uuid.uuid4()}.png")
