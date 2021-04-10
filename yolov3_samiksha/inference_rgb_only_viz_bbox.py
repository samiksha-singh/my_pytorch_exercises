import argparse
import torch
from tqdm import tqdm
from utils import utils
import numpy as np
from pathlib import Path
import cv2

from models import Darknet

from transforms import get_transforms
from dataset_rgb_only import RGBOnly


def draw_bbox(img, label_tensor_nx6, img_id):
    img_opencv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Skip images with no predictions
    if label_tensor_nx6.shape[0] == 0:
        print(f"No predictions for image {img_id}")
        cv2.imwrite(f"output/{img_id}.jpg", img_opencv)
        return

    label_np = label_tensor_nx6.numpy()
    for single_label in label_np:
        bbox = single_label[:4].astype(np.uint32)
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        color = (255, 0, 0)
        thickness = 2
        image = cv2.rectangle(img_opencv, start_point, end_point, color, thickness)

    cv2.imwrite(f"output/{img_id}.jpg", image)


def inference_visualize(model, dataloader, device, conf_thres, nms_thres):
    """Runs inference on a set of images and saves the images with bboxes drawn on them"""
    model.eval()

    for batch_i, (imgs, dummy_label) in enumerate(tqdm(dataloader, desc="Detecting objects")):
        if batch_i >= 1:
            break

        imgs = imgs.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = outputs.detach().cpu()
            outputs = utils.non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        imgs = imgs.detach().cpu()
        for idx, bbox_tensor in enumerate(outputs):
            # get image corresponding to label
            image = imgs[idx]
            image = (image.numpy() * 255).astype(np.uint8)
            image = image.transpose((1, 2, 0))

            draw_bbox(image, bbox_tensor, idx)

            print(f"Image {idx}")
            for x1, y1, x2, y2, cls_conf, cls_pred in bbox_tensor:
                print(f"\t+ Label: {int(cls_pred)}, Conf: {cls_conf.item():.5f}")

    return

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--root_test", type=Path, default="sample/", help="root directory containing rgb images")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup model
    model = Darknet(opt.model_def, img_size=416).to(device)

    # Load weights
    if opt.weights_path is not None:
        if opt.weights_path.endswith('.weights'):
            # Load original author's darknet weights
            print('Loading darknet weights')
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load our training checkpoint weights
            print('Loading trained checkpoint')
            model.load_state_dict(torch.load(opt.weights_path))

    # dataloader
    root_test = opt.root_test
    dataset_test = RGBOnly(root_test, transform=get_transforms(img_size=416))

    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    batch_size = opt.batch_size
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    inference_visualize(model, testloader, device, conf_thres=conf_thres, nms_thres=nms_thres)
