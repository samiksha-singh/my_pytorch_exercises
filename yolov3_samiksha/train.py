import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import wandb
import cv2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.sequence = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.sequence(x)
        return logits

def train_loop (dataloader, model, loss_fn, optimizer):
    for batch_idx, (X, y) in enumerate(dataloader):
        size = len(dataloader.dataset)
        #now we do the prediction
        pred = model(X)
        #calculate the loss
        loss = loss_fn(pred, y)

        #backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_, current = loss.item(), batch_idx * len(X)
        wandb.log({"Train/loss": loss_})

        if batch_idx % 100 == 0:
            print(f"loss : {loss_:>7f} [{current:>5d}/{size:>5d}]")


def test_loop (dataloader, model , loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).float().sum().item()


    test_loss /= size
    correct /= size
    wandb.log({"Test/loss": test_loss})

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    wandb.init(project="training_loop_tutorial", entity='samiksha')

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=ToTensor())
    trainset = torch.utils.data.Subset(trainset, range(1000))
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=ToTensor())
    testloader = DataLoader(testset, batch_size=8, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    model = NeuralNetwork()

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

    epochs = 5

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(trainloader, model , loss_fn, optimizer)
        test_loop(testloader, model, loss_fn)

    path_to_img = "/home/samiksha/my_pycharm_projects/machine_learning_pycharm/Dataset_customized/images/000005.jpg"
    img = cv2.imread(path_to_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = wandb.Image(img, boxes={
        "predictions": {
            "box_data": [{
                "position":{
                    "minX": 0.2,
                    "maxX": 0.1,
                    "minY": 0.1,
                    "maxY": 0.4,
                },
                "class_id": 2,
                "box_caption": "minMax(pixel)",
            },
            ],
        },

    })
    wandb.log({"Train/Prediction": image})


    print("Done")

if __name__ == "__main__":
    main()

