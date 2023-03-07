import os
import torch
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import barts2019loader
import unet3d_test as model



def train(net, device, data_root, epochs=40, batch_size=4, lr=1e-5):
    data = barts2019loader.dataset(data_root)
    train_loader = torch.utils.data.Dataloader(dataset=data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion=nn.BCEWithLogitsLoss()
    max_loss = float('inf')

    for epoch in range(epochs):
        print('EPOCH: ' + epoch)
        net.train()

        for image, label in train_loader:
            optimizer.zero_grad()
            image.reshape(1,0,2,3,4)
            image = torch.Tensor(image)
            label = torch.Tensor(label)
            # image = image.to(device=device, dtype=torch.float32)
            # label = label.to(device=device, dtype=torch.float32)

            pred = net(image)

            loss = criterion(pred,label)

            print('Loss/train', loss.item())

            if loss < max_loss:
                max_loss = loss
                loss.backward()
                optimizer.step()

if __name__ == "__main__": 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPOCH = 70
    data_root = './joey/examples/3d_unet/data'
    batch_size=4

    net = model(in_channel = 4, filter = 16)
    train(net, device, data_root, MAX_EPOCH, batch_size)
