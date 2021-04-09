import numpy as np
from tqdm import tqdm

from utils import utils
import logger

def evaluate(model, dataloader, device, iou_thres, conf_thres, nms_thres, mode="Train"):
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
        img_size = imgs.shape[1]
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.detach().cpu()
            outputs = utils.non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += utils.get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        # Upload images to wandb
        if batch_i % 25 == 0:
            logger.log_bboxes(imgs, targets, outputs, f"{mode}/Predictions", 8)

    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    wandb.log({f"{mode}/precision": precision.mean().item()}, commit=False)
    wandb.log({f"{mode}/recall": recall.mean().item()}, commit=False)
    wandb.log({f"{mode}/AP": AP.mean().item()}, commit=False)
    wandb.log({f"{mode}/f1": f1.mean().item()}, commit=False)

    return precision, recall, AP, f1, ap_class
