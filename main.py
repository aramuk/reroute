#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils as utils
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes

def main():
    train_set = Cityscapes('./cityscapes',
                        mode='fine',
                        split='train',
                        target_type='semantic',
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]),
                        target_transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))
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

if __name__ == '__main__':
    main()