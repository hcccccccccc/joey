import torch
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import time
import os
import matplotlib.pyplot as plt
import barts2019loader
import diceloss
import unet3d_test
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, num_epochs, optimizer, criterion):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(self):
        # print("memory used: {}".format(torch.cuda.memory_allocated(0)))
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)
        total_train_loss = []
        total_val_loss = []
        total_val_acc = []
        total_time = 0
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O0")

        for epoch in range(self.num_epochs):
            start = time.time()
            print("EPOCH: {}".format(epoch+1))
            self.model.train()
            train_loss = 0.0

            for batch, (inputs, targets) in enumerate(train_loader):
                print('\rprocess {:.2%}'.format(batch/len(train_loader)), end='')
                inputs = torch.stack(inputs, dim=1)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, et, tc, wt = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()
            
            time_finished = time.time() - start

            train_loss /= len(self.train_dataset)            
            val_loss, val_acc = self.validate(val_loader)

            print(f'Epoch: {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
            print("Time spend: {:.0f}m {:.0f}s".format(time_finished // 60, time_finished % 60))
            total_time += time_finished
            total_train_loss.append(train_loss)
            total_val_loss.append(val_loss)
            total_val_acc.append(val_acc)
            plt.clf()
            plt.plot(range(0, epoch+1),total_train_loss, label='train loss')
            plt.plot(range(0, epoch+1),total_val_loss, label='val loss')
            # plt.plot(range(0, epoch+1),total_val_acc, label='val acc')
            # plt.plot(range(0,epoch+1), ET_loss, label='ET loss')
            # plt.plot(range(0,epoch+1), TC_loss, label='TC loss')
            # plt.plot(range(0,epoch+1), WT_loss, label='WT loss')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig('test_loss.png')

        print("Total time: {:.0f}m {:.0f}s | Average time cost: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60, (total_time // self.num_epochs) // 60, (total_time // self.num_epochs )% 60))

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = torch.stack(inputs, dim=1)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss[0].item() * inputs.size(0)
                # preds = torch.argmax(outputs, axis=1)
                # val_acc += accuracy_score(targets.flatten().cpu().numpy(), outputs.flatten().cpu().numpy()) * inputs.size(0)
                
        val_loss /= len(self.val_dataset)
        # val_acc /= len(self.val_dataset)
        
        return val_loss, val_acc

if __name__ == "__main__": 
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 5
    data_root = '/run/datasets/MICCAI_BraTS_2019_Data_Training' 
    batch_size = 2
    data_size = 128

    # dataset, dataloader
    barts2019 = barts2019loader.BratsDataset(data_root, data_size, 'pytorch')
    train_idx, val_idx = train_test_split(list(range(len(barts2019))), test_size=0.203, random_state=42)
    train_dataset = Subset(barts2019, train_idx)
    val_dataset = Subset(barts2019, val_idx)

    # model
    net = unet3d_test.test_pytorch(in_channel = 4, filter = 16)
    criterion=diceloss.WeightedMulticlassDiceLoss(num_classes=3, class_weights=[0.5,0.3,0.2])
    optimizer=torch.optim.Adam(net.parameters(), lr=5e-4)
    # summary(net, input_size=(4,128,128,128))

    #train
    trainer = Trainer(net, train_dataset, val_dataset, batch_size, num_epochs, optimizer, criterion)
    trainer.train()