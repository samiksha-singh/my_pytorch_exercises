import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset_Pascal import PascalVOC, collate_fn
from models import Darknet
from transforms import DEFAULT_TRANSFORMS
from utils import utils


def draw_bbox(img, pred_bboxes, targets_bboxes, img_id):
    """
    Draw the predictions and labels bboxes on the image

    Args:
        img (np.ndarray): RGB image
        pred_bboxes (torch.Tensor): Predictions. Shape=[N, 6] (x1, y1, x2, y2, conf, cls_id). Absolute coords.
        targets_bboxes (torch.Tensor): Labels. Shape=[N, 6] (img_id, cls_id, x1, y1, x2, y2). Absolute coords.
        img_id (int): Index of image within batch

    Returns:
        None
    """
    img_opencv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Skip images with no predictions
    if pred_bboxes.shape[0] == 0:
        print(f"No predictions for image {img_id}")
        cv2.imwrite(f"output/{img_id}.jpg", img_opencv)
        return

    pred_bboxes_np = pred_bboxes.numpy()
    for pred_box in pred_bboxes_np:
        bbox = pred_box[:4].astype(np.uint32)
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(img_opencv, start_point, end_point, color, thickness)

    targets_bboxes_np = targets_bboxes.numpy()
    for lbl_box in targets_bboxes_np:
        bbox = lbl_box[2:6].astype(np.uint32)
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        color = (0, 0, 255)
        thickness = 2
        image2 = cv2.rectangle(image, start_point, end_point, color, thickness)

    cv2.imwrite(f"output/{img_id}.jpg", image2)


def evaluate(model, dataloader, device, iou_thres, conf_thres, nms_thres, mode="Train"):
    """Calculate metrics across the dataset"""
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (imgs, targets) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        if batch_i > 0:
            break

        imgs = imgs.to(device)
        if targets is None:
            continue

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets = utils.xywh_to_xyxy(targets).cpu()
        img_size = imgs.shape[2]
        targets[:, 2:] *= img_size

        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.detach().cpu()
            outputs = utils.non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        imgs = imgs.detach().cpu()
        for idx, pred_bboxes in enumerate(outputs):
            # Select corresponding image
            images = imgs[idx]
            images_np = (images.numpy() * 255).astype(np.uint8)
            images_np = images_np.transpose((1, 2, 0))
            # Select corresponding labels
            targets_for_img = utils.select_bbox_from_img_id(targets, idx)
            # Draw bboxes
            draw_bbox(images_np, pred_bboxes, targets_for_img, idx)

        sample_metrics += utils.get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' - AP: {AP[i]}")

    print(f"{mode}/mAP: {AP.mean().item():0.5f}")
    print(f"{mode}/f1: {f1.mean().item():0.5f}")

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--root_test", type=Path, default="/home/samiksha/dataset/voc2007/test",
                        help="root directory for test")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model
    model = Darknet(opt.model_def).to(device)

    # Load weights
    if opt.weights_path is not None:
        if opt.weights_path.endswith('.weights'):
            # Load darknet weights
            print('Loading darknet weights')
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            print('Loading trained checkpoint')
            model.load_state_dict(torch.load(opt.weights_path))

    # dataloader
    root_test = opt.root_test
    batch_size = model.hyperparams['batch']
    dataset_test = PascalVOC(root_test, transform=DEFAULT_TRANSFORMS)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                             collate_fn=collate_fn, num_workers=8)

    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    iou_thres = opt.iou_thres
    evaluate(model, testloader, device, iou_thres=iou_thres, conf_thres=conf_thres, nms_thres=nms_thres, mode="Train")
