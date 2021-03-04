#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils as utils
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes

def get_dataset(split='train'):
    return Cityscapes('./cityscapes',
                    mode='fine',
                    split=split,
                    target_type='semantic',
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]),
                    target_transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))

def explore(train_set):
    train_set = get_dataset('train')
    train_loader = utils.data.DataLoader(train_set, batch_size=20)
    images, targets = next(iter(train_loader))
    print(images.shape, targets.shape)
    images = images.permute(0, 2, 3, 1)
    targets = targets.permute(0, 2, 3, 1)
    print(images.shape, targets.shape)
    fig, ax = plt.subplots(20, 2, figsize=(10, 20))
    for i in range(20):
        ax[i][0].imshow(images[i])
        ax[i][1].imshow(targets[i])
    plt.show()

def main():
    test_set = get_dataset('test')

    model = deeplabv3_resnet50(pretrained=True)
    print(model.classifier)
    image, target = next(iter(test_set))

    output = model(image)

    fig, ax = plt.subplots(1, 3)
    for i in range(1):
        ax[i][0].imshow(image)
        ax[i][1].imshow(output)
        ax[i][2].imshow(target)


if __name__ == '__main__':
    main()