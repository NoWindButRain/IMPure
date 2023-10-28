import os
import time
import json
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

import target_model
from dataset import ImageNet

torch.backends.cudnn.benchmark=True

device = "cuda:2" if torch.cuda.is_available() else "cpu"

model_name = 'inceptionresnetv2_v1'
model = getattr(target_model, model_name)()
model.model = model.model.to(device)
loss_fun = torch.nn.CrossEntropyLoss(reduction='none')

filter_size = -1
root_path = '/data/object_class/ILSVRC2012'
dataset_split = 'val' # 'train' 'val'
batch_size = 128
dataset = ImageNet(
    root_path, transform=model.preprocess, split=dataset_split
)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True
)

if __name__ == '__main__':
    print(model)
    print(dataset)
    total_size = len(dataset)
    correct = 0
    true_data = [[] for _ in range(1000)]
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model.model(images)
            loss = loss_fun(outputs, labels).cpu()
            preds = outputs.argmax(1)
            correct += (preds == labels).type(torch.float).sum().item()
            imgs = dataset.imgs[i*batch_size: i*batch_size+len(images)]
            for j in np.arange(len(imgs))[(preds == labels).cpu().numpy()]:
                img = imgs[j]
                img = [img[0].replace(root_path, '.'), img[1], loss[j].item()]
                true_data[img[1]].append(img)
    print(f"Accuracy: {(100*correct/total_size):>0.2f}%")
    filter_data = []
    for item in true_data:
        if filter_size > 0 and len(item) >= filter_size:
            filter_data.extend([item[i] for i in np.random.choice(len(item), filter_size, replace=False)])
        else:
            # print(len(item))
            filter_data.extend(item)

    _date = time.strftime('%Y%m%d%H%M%S', time.localtime())
    save_path = '../results/filter_data_fix/{}_{}_{}_{}_{}.json'.format(
        dataset.name, dataset.split, model.name, filter_size, _date)
    with open(save_path, 'w') as file:
        json.dump(filter_data, file)