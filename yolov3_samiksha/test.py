import argparse
from pathlib import Path
import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import *
from models import Darknet
from utils import utils
from dataset_Pascal import PascalVOC, collate_fn
from transforms import DEFAULT_TRANSFORMS
from loss import compute_loss


def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres):
    """Calculate metrics across the dataset"""
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):

        if targets is None:
            continue

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets = utils.xywh_to_xyxy(targets).cpu()
        img_size = imgs.shape[1]
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.detach().cpu()
            outputs = utils.non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += utils.get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class
