import numpy as np
import torch
import wandb
from tqdm import tqdm

from utils import utils
import logger

def evaluate_metrics(model, dataloader, device, iou_thres, conf_thres, nms_thres, mode="Test"):
    """Calculate metrics across the dataset"""
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        imgs = imgs.to(device)
        if targets is None:
            continue

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets = utils.xywh_to_xyxy(targets).cpu()
        _, _, height, width = imgs.shape  # height and width will be the same
        targets[:, 2:] *= height

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

    wandb.log({f"{mode}/precision": precision.mean().item()}, commit=False)
    wandb.log({f"{mode}/recall": recall.mean().item()}, commit=False)
    wandb.log({f"{mode}/mAP": AP.mean().item()}, commit=False)
    wandb.log({f"{mode}/f1": f1.mean().item()}, commit=False)

    # print("Average Precisions:")
    # class_dict = dataloader.dataset.dataset.class_dict_reverse
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_dict[c]}) - AP: {AP[i]}")
    print(f"{mode}/mAP: {AP.mean().item()}")

    return precision, recall, AP, f1, ap_class


def log_bbox_predictions(model, dataloader, device, conf_thres, nms_thres, mode="Test", max_images_to_upload=16):
    model.eval()
    for batch_i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        if targets is None:
            continue

        if batch_i > 0:
            break

        # Rescale target
        targets = utils.xywh_to_xyxy(targets).cpu()
        _, _, height, width = imgs.shape  # height and width will be the same
        targets[:, 2:] *= height

        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.detach().cpu()
            outputs = utils.non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        class_dict = dataloader.dataset.dataset.class_dict_reverse
        logger.log_bboxes(imgs, targets, outputs, class_dict,
                          wandb_heading=f"{mode}/Predictions", max_images_to_upload=max_images_to_upload)
