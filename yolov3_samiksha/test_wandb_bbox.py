import torch
import numpy
from train import PascalVOC, collate_fn
from transforms import DEFAULT_TRANSFORMS

root_train = "/home/samiksha/dataset/voc2007/train/"
dataset_train = PascalVOC(root_train, transform=DEFAULT_TRANSFORMS)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True,
                                          collate_fn=collate_fn, num_workers=8)

for idx, batch in enumerate(trainloader):
    img_tensor, label_tensor = batch
    for idx, img in enumerate(img_tensor):
        a = []
        for lbl in label_tensor:
            if lbl[0] == idx:
                a.append(lbl)
        b = torch.stack(a, 0)



label = torch.zeros((3,6))
label[0,0] = 1
label[1,0] = 2
label[2,0] = 2
label[:,1] = 10
print(label)
print("")

print(b)

