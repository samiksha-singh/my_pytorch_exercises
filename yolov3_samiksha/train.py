import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Darknet
from utils import utils
from dataset_Pascal import PascalVOC, collate_fn
from transforms import get_transforms
from loss import compute_loss
from test import evaluate_metrics, log_bbox_predictions


def train_loop (dataloader, model, optimizer, device):
    for batch_idx, (imgs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        model.train()

        # Forward Pass
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)
        outputs = model(imgs)

        # Calculate the loss
        loss, loss_components = compute_loss(outputs, targets, model)

        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        loss_, current = loss.item(), batch_idx * len(imgs)
        wandb.log({"Train/loss": loss_})


def main(opt):
    wandb.init(project="training_loop_tutorial", entity='samiksha')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(utils.weights_init_normal)

    pretrained_weights = opt.pretrained_weights
    if pretrained_weights is not None:
        print(f'\nLoading weights: {pretrained_weights}\n')
        if pretrained_weights.endswith(".pth"):
            # Load our pytorch training's checkpoint
            checkpoint = torch.load(pretrained_weights)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Load original author's darknet weights (trained on yolo)
            model.load_darknet_weights(pretrained_weights)

    # dataloader
    root_train = opt.root_train
    root_test = opt.root_test
    img_size = opt.img_size
    dataset_train = PascalVOC(root_train, transform=get_transforms(img_size=img_size))
    dataset_test = PascalVOC(root_test, transform=get_transforms(img_size=img_size))

    # Take subset of dataset for faster testing
    debug_mode = opt.debug_mode
    if debug_mode:
        num_images_train = 100
        num_images_test = 100
        print(f'Warning: Debugging mode, only {num_images_train} images from datasets will be used.')
    else:
        num_images_train = len(dataset_train)
        num_images_test = len(dataset_test)
    dataset_train = torch.utils.data.Subset(dataset_train, list(range(num_images_train)))
    dataset_test = torch.utils.data.Subset(dataset_test, list(range(num_images_test)))

    batch_size = model.hyperparams['batch']
    n_cpu = opt.n_cpu
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                              collate_fn=collate_fn, num_workers=n_cpu)

    sampler = torch.utils.data.SequentialSampler(dataset_test)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                             collate_fn=collate_fn, num_workers=n_cpu)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model.hyperparams['learning_rate'],
        weight_decay=model.hyperparams['decay'],
    )

    epochs = opt.epochs
    evaluation_interval = opt.evaluation_interval
    checkpoint_interval = opt.checkpoint_interval
    log_image_interval = opt.log_image_interval
    for epoch_idx in range(epochs):
        print(f"Epoch {epoch_idx + 1}\n-------------------------------")
        train_loop(trainloader, model, optimizer, device)

        # Run Evaluation
        if (epoch_idx+1) % evaluation_interval == 0:
            evaluate_metrics(model, testloader, device, iou_thres=0.5, conf_thres=0.1, nms_thres=0.5, mode="Test")
        if (epoch_idx+1) % log_image_interval == 0:
            log_bbox_predictions(model, testloader, device, conf_thres=0.1, nms_thres=0.5, mode="Test",
                                 max_images_to_upload=16)

        # Save checkpoint
        if (epoch_idx+1) % checkpoint_interval == 0:
            run_id = wandb.run.id
            save_dir = Path(f"checkpoints/{run_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(save_dir / f"yolov3_ckpt_{epoch_idx}.pth")

            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str,
                        help="Whether to load any weights. Can be original darknet weights or prev pytorch saved checkpoints")
    parser.add_argument("--root_train", type=Path, default="/home/samiksha/dataset/voc2007/train",
                        help="root directory for train")
    parser.add_argument("--root_test", type=Path, default="/home/samiksha/dataset/voc2007/test",
                        help="root directory for test")
    parser.add_argument("--debug_mode", action="store_true",
                        help="Run in debug mode. Only small subset of data will be used")

    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--log_image_interval", type=int, default=1, help="interval evaluations on validation set")
    opt = parser.parse_args()
    main(opt)
