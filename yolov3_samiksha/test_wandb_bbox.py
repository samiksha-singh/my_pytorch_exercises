import torch
import numpy as np
import numpy
import wandb
from train import PascalVOC, collate_fn
from transforms import DEFAULT_TRANSFORMS

root_train = "/home/samiksha/dataset/voc2007/train/"
dataset_train = PascalVOC(root_train, transform=DEFAULT_TRANSFORMS)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True,
                                          collate_fn=collate_fn, num_workers=8)
wandb.init(project="test_wandb_bbox", entity='samiksha')
for batch_idx, batch in enumerate(trainloader):
    img_tensor, label_tensor = batch
    def bbox_wandb (img_tensor, label_tensor):
        img_cat_list = []
        for idx, img in enumerate(img_tensor):
            img_numpy = np.array(img)
            a = []
            for lbl in label_tensor:
                if lbl[0] == idx:
                    a.append(lbl)
            b = torch.stack(a, 0)
            #convert x,y,w,h to xmin,ymin,xmax,ymax
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
            prediction_list = []
            #img_cat_list = []
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
                image = wandb.Image(img, boxes=
                {
                    "predictions": {
                        "box_data": prediction_list
                    },

                })
                img_cat_list.append(image)
            wandb.log({"Train/Prediction": img_cat_list})

    if batch_idx > 1:
        break




label = torch.zeros((3,6))
label[0,0] = 1
label[1,0] = 2
label[2,0] = 2
label[:,1] = 10
#print(label)
#print("")
#print(b[:,2])

