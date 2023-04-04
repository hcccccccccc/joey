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


class WeightedMulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(WeightedMulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.epsilion = 1e-5
        
    def forward(self, input, target):
        # calculate class weights
        if self.class_weights is None:
            class_weights = torch.ones(self.num_classes).to(target.device)
        else:
            class_weights = torch.tensor(self.class_weights).to(target.device)

        # calculate soft probabilities
        # input = F.softmax(input, dim=1)

        # calculate one-hot target
        target_one_hot = target.float()

        # # calculate per channel dice
        # per_channel_dice = torch.zeros(self.num_classes).to(target.device)
        # for i in range(self.num_classes):
        #     class_mask = target_one_hot[:, i, :, :, :]
        #     if class_mask.sum() > 0:
        #         intersection = torch.sum(input_soft[:, i, :, :, :] * class_mask, dim=(1, 2, 3))
        #         union = torch.sum(input_soft[:, i, :, :, :], dim=(1, 2, 3)) + torch.sum(class_mask, dim=(1, 2, 3))
        #         per_channel_dice[i] = (2 * intersection / union).mean()

        # # calculate weighted loss
        # weights = class_weights / (per_channel_dice ** 2 + 1e-6)
        # weights /= weights.sum()
        # weighted_per_channel_dice = per_channel_dice * weights
        # loss = 1 - weighted_per_channel_dice.sum()

        # calculate ET, TC, WT loss
        et_mask = target_one_hot[:, 0, :, :, :]
        tc_mask = target_one_hot[:, 1, :, :, :] + et_mask
        wt_mask = target_one_hot[:, 2, :, :, :] + tc_mask

        et_pred = input[:, 0, :, :, :]
        tc_pred = input[:, 1, :, :, :] + et_pred
        wt_pred = input[:, 2, :, :, :] + tc_pred

        et_dice = self.dice_loss(et_pred, et_mask)
        tc_dice = self.dice_loss(tc_pred, tc_mask)
        wt_dice = self.dice_loss(wt_pred, wt_mask)

        loss = (et_dice*class_weights[0] + tc_dice*class_weights[1] + wt_dice*class_weights[2])

        return loss, et_dice, tc_dice, wt_dice
    
    def dice_loss(self, prediction, target):
        intersection = (prediction * target).sum(dim=(1, 2, 3))
        union = (prediction + target).sum(dim=(1, 2, 3))
        dice = 2 * (intersection + self.epsilion) / (union + self.epsilion)
        loss = 1 - dice.mean()
        return loss