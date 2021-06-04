import torch as t
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
    
    def _one_hot_encoder(self, in_tensor):
        tensor_ls = []
        for i in range(self.n_classes):
            temp_prob = in_tensor == i
            tensor_ls.append(temp_prob.unsqueeze(1))
        output_tensor = t.cat(tensor_ls, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = t.sum(score * target)
        y_sum = t.sum(target*target)
        z_sum = t.sum(score*score)
        loss = (2*intersect + smooth)/(z_sum + y_sum + smooth)
        loss = 1-loss
        return loss
    
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = t.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []

        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
