import torch
import torch.nn as nn

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

        # calculate one-hot target
        target_one_hot = target.float()

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