import argparse
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from utils.logger import *
from models import Darknet
from utils import utils
from dataset_Pascal import PascalVOC, collate_fn
from transforms import DEFAULT_TRANSFORMS
from loss import compute_loss



def train_loop (dataloader, model, compute_loss, optimizer,device):
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

        if batch_idx % 100 == 0:
            print(f"loss : {loss_:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]")


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
    wandb.log({"Test/loss": test_loss.cpu().item()})

    print(f"Avg loss: {test_loss.cpu().item():>8f} \n")


def main(opt):
    wandb.init(project="training_loop_tutorial", entity='samiksha')

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
        # train_loop(trainloader, model, compute_loss, optimizer, device)
        test_loop(testloader, model, compute_loss, device)

    # path_to_img = "/home/samiksha/my_pycharm_projects/machine_learning_pycharm/Dataset_customized/images/000005.jpg"
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



