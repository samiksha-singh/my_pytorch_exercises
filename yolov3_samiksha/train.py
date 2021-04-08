import argparse
from pathlib import Path
import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from utils import utils

from utils.logger import *
from models import Darknet
from utils import utils
from dataset_Pascal import PascalVOC, collate_fn
from transforms import DEFAULT_TRANSFORMS
from loss import compute_loss

def xywh_to_xyxy(label_xywh):
    x = label_xywh[:, 2]
    y = label_xywh[:, 3]
    w = label_xywh[:, 4]
    h = label_xywh[:, 5]

    label_xyxy = torch.zeros_like(label_xywh)
    label_xyxy[:, 2] = (x - (w / 2))
    label_xyxy[:, 3] = (y - (h / 2))
    label_xyxy[:, 4] = (x + (w / 2))
    label_xyxy[:, 5] = (y + (h / 2))
    label_xyxy[:, 0] = label_xywh[:, 0]
    label_xyxy[:, 1] = label_xywh[:, 1]

    return label_xyxy

def log_bboxes(imgs, targets, max_imgs_to_log=3):
    """Log predicted bboxes to wandb"""
    for idx, img in enumerate(imgs[:max_imgs_to_log]):
        # select bboxes belonging to image using image id
        matching_bboxes = []
        for lbl in targets:
            if lbl[0] == idx:
                matching_bboxes.append(lbl)
        label_xywh = torch.stack(matching_bboxes, 0)

        # convert x,y,w,h to xmin,ymin,xmax,ymax
        label_xyxy = xywh_to_xyxy(label_xywh)
        # convert to absolute values
        # label_xyxy[:, 2:] *= opt.img_size

        # upload to wandb
        pass

    return

def train_loop (dataloader, model, compute_loss, optimizer, device):
    sample_metrics = []
    labels = []
    for batch_idx, (imgs, targets) in enumerate(dataloader):

        model.train()

        #now we do the prediction
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)

        outputs = model(imgs)

        #calculate the loss
        loss, loss_components = compute_loss(outputs, targets, model)
        #backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_, current = loss.item(), batch_idx * len(imgs)
        wandb.log({"Train/loss": loss_})


        # Calculate metrics
        outputs = utils.non_max_suppression(outputs, conf_thres=opt.conf_thres, iou_thres=opt.nms_thres)
        label_xyxy = xywh_to_xyxy(targets)
        img_size = imgs.shape[1]
        label_xyxy[:, 2:] *= img_size
        sample_metrics += utils.get_batch_statistics(outputs, label_xyxy, iou_threshold=opt.iou_thres)
        # Extract labels for calculating full epoch stats
        labels += targets[:, 1].tolist()

        if batch_idx % 100 == 0:
            print(f"loss : {loss_:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]")
            img_cat_list = bbox_wandb(imgs, targets)
            wandb.log({"Train/Prediction": img_cat_list})

            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = utils.ap_per_class(true_positives, pred_scores, pred_labels, labels)
            wandb.log({"Train/precision": precision.mean()})
            wandb.log({"Train/recall": recall.mean()})
            wandb.log({"Train/AP": AP.mean()})
            wandb.log({"Train/F1": f1.mean()})

def test_loop(dataloader, model, compute_loss, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            test_loss += loss

    test_loss /= size
    #wandb.log({"Test/loss": test_loss.cpu().item()})

    print(f"Avg loss: {test_loss.cpu().item():>8f} \n")
    for images, labels in targets:
        img_cat_list = bbox_wandb(images, labels)
        wandb.log({"Test/Prediction": img_cat_list})
# Draw bbox on the images using wandb

def bbox_wandb(img_tensor, label_tensor):
    img_cat_list = []
    for idx, img in enumerate(img_tensor):
        # select bboxes belonging to image using image id
        a = []
        for lbl in label_tensor:
            if lbl[0] == idx:
                a.append(lbl)
        b = torch.stack(a, 0)

        # convert x,y,w,h to xmin,ymin,xmax,ymax
        x = b[:,2]
        y = b[:,3]
        w = b[:,4]
        h = b[:,5]
        label_xyxy = torch.zeros_like(b)
        label_xyxy[:,2] = (x - (w/2))
        label_xyxy[:,3] = (y - (h/2))
        label_xyxy[:,4] = (x + (w/2))
        label_xyxy[:,5] = (y + (h/2))
        label_xyxy[:,0] = b[:,0]
        label_xyxy[:,1] = b[:,1]

        # Convert bbox tensor to wandb Image for logging
        prediction_list = []
        for tensor_ele in label_xyxy:
            bbox_data = {
                "position" : {
                    "minX" : tensor_ele[2].item(),
                    "maxX" : tensor_ele[4].item(),
                    "minY" : tensor_ele[3].item(),
                    "maxY" : tensor_ele[5].item(),
                },
                "class_id": int(tensor_ele[1].item()),
            }
            prediction_list.append(bbox_data)

        image = wandb.Image(
            img,
            boxes={"predictions": {"box_data": prediction_list}}
        )
        img_cat_list.append(image)
    return img_cat_list


def main(opt):
    #wandb.init(project="training_loop_tutorial", entity='samiksha')

    logger = Logger(opt.logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(utils.weights_init_normal)

    # dataloader
    root_train = opt.root_train
    root_test = opt.root_test
    dataset_train = PascalVOC(root_train, transform=DEFAULT_TRANSFORMS)
    dataset_test = PascalVOC(root_test, transform=DEFAULT_TRANSFORMS)

    batch_size = model.hyperparams['batch']
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                              collate_fn=collate_fn, num_workers=8)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                             collate_fn=collate_fn, num_workers=8)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay'],
    )

    epochs = opt.epochs

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(trainloader, model, compute_loss, optimizer, device)
        test_loop(testloader, model, compute_loss, device)




    # img = cv2.imread(path_to_img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = wandb.Image(img, boxes={
    #     "predictions": {
    #         "box_data": [{
    #             "position":{
    #                 "minX": 0.2,
    #                 "maxX": 0.1,
    #                 "minY": 0.1,
    #                 "maxY": 0.4,
    #             },
    #             "class_id": 2,
    #             "box_caption": "minMax(pixel)",
    #         },
    #         ],
    #     },
    #
    # })
    # wandb.log({"Train/Prediction": image})

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--verbose", "-v", default=False, action='store_true',
                        help="Makes the training more verbose")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Defines the directory where the training log files are stored")
    parser.add_argument("--root_train", type=Path, default="/home/samiksha/dataset/voc2007/train",
                        help="root directory for train")
    parser.add_argument("--root_test", type=Path, default="/home/samiksha/dataset/voc2007/test",
                        help="root directory for test")
    opt = parser.parse_args()
    main(opt)



