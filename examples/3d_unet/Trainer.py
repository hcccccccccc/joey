import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import barts2019loader
import diceloss
import unet3d_test

class Trainer:
    def __init__(self, model, train_dataset, criterion, batch_size, lr, num_epochs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = criterion
        self.num_epochs = num_epochs
        
    def train(self):
        total_loss = []
        ET_loss = []
        TC_loss = []
        WT_loss = []
        for epoch in range(self.num_epochs):
            train_loss = 0
            train_acc = 0
            tc_dice = 0
            wt_dice = 0
            et_dice = 0
            print('EPOCH: {}'.format(epoch+1))
            self.model.train()
            train_loss = 0
            train_correct = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                train_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                loss.backward()
                self.optimizer.step()
            train_loss /= len(self.train_loader.dataset)
            train_accuracy = train_correct / len(self.train_loader.dataset)
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    val_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_loss /= len(self.val_loader.dataset)
            val_accuracy = val_correct / len(self.val_loader.dataset)
            
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}'
                  .format(epoch+1, self.num_epochs, train_loss, train_accuracy, val_loss, val_accuracy))


if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPOCH = 200
    data_root = './datasets/HGG'
    batch_size= 2
    barts2019 = barts2019loader.BratsDataset(data_root)
    criterion=diceloss.WeightedMulticlassDiceLoss(num_classes=3, class_weights=[0.5,0.3,0.2])
    net = unet3d_test.test_pytorch(in_channel = 4, filter = 16)
    trainer = Trainer(net, barts2019, criterion, batch_size, lr=5e-4, num_epochs=MAX_EPOCH)