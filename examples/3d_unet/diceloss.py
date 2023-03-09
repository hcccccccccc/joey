import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1.):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        nclasses = pred.shape(1)
        
        # flatten predictions and targets
        pred_flat = pred.view(pred.size(0), nclasses, -1)
        target_flat = target.view(target.size(0), nclasses, -1)
    
        # compute intersection
        intersection = (pred_flat * target_flat).sum(-1)
    
        # compute cardinality
        cardinality = pred_flat.sum(-1) + target_flat.sum(-1)
    
        # compute dice coefficient
        dice_coef = (2. * intersection + self.epsilon) / (cardinality + self.epsilon)
    
        # compute dice loss
        loss = 1. - dice_coef.mean()

        return loss


class MultiDiceLoss(nn.Module):
    def __init__(self, epsilon=1., nclasses=3) -> None:
        super(MultiDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.nclasses = nclasses
    
    def forward(self, pred, target):
        # channel = input.shape[1]
        # dice = DiceLoss()
        # totalLoss = 0

        # for i in range(channel):

        #     diceloss =  dice(input[:,i], target)
        #     if(self.weight is not None):
        #         diceLoss *= self.weight[i]
        #     totalLoss += diceloss
        # return totalLoss / channel
        pred = torch.sigmoid(pred)

        loss = 0.
        for c in range(self.nclasses):
            # get one-hot encoding of target for current class
            target_c = (target == c).float()
            # get one-hot encoding of pred for current class
            pred_c = pred[:, c]
            # calculate intersection between pred and target
            intersection = (pred_c * target_c).sum()
            # calculate sum of pred and target
            total = pred_c.sum() + target_c.sum()
            # calculate dice score
            dice = (2. * intersection + self.epsilon) / (total + self.epsilon)
            # add dice score to loss
            loss += 1. - dice
        return loss / self.nclasses
    
        return multi_dice_loss
