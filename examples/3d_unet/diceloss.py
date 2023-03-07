import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        num = pred.size(0)
        
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(num, -1)
        target_flat = target.view(num, -1)

        intersection = pred_flat*target_flat
        union = pred_flat.sum(1) + target_flat.sum(1)

        loss = (2 * (intersection.sum(1) + self.epsilon) / (union + self.epsilon))
        loss = 1 - loss.sum()/num

        return loss


class MultiDiceLoss(nn.Module):
    def __init__(self, weight=None) -> None:
        super(MultiDiceLoss, self).__init__()
        self.weight = weight
    
    def forward(self, input, target):
        channel = input.shape[1]
        dice = DiceLoss()
        totalLoss = 0

        for i in range(channel):

            diceloss =  dice(input[:,i], target)
            if(self.weight is not None):
                diceLoss *= self.weight[i]
            totalLoss += diceloss
        return totalLoss / channel
