import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#####################################################################################
"""
		Loss functions
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################


class DiceLoss(nn.Module):
    """
    The Dice Loss function
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, labels):
        total_dice_loss = 0.0
        num_classes = probs.size(1)
        dice_loss_label = np.zeros(num_classes)

        for j in range(num_classes):
            probs_square_sum = torch.sum(probs[:, j, :, :])
            target_square_sum = torch.sum(labels[:, j, :, :])
            intersect = torch.sum(probs[:, j, :, :] * (labels[:, j, :, :]))
            dice = (2 * intersect + self.smooth) / (probs_square_sum + target_square_sum + intersect + self.smooth)
            dice_loss_per_class = 1 - dice
            total_dice_loss += dice_loss_per_class
            dice_loss_label[j] = dice_loss_per_class

        total_dice_loss /= num_classes

        return dice_loss_label, total_dice_loss


class BoundaryLoss(nn.Module):
    """
    The Boundary Loss function
    """
    def __init__(self, smooth=1e-6):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth
        a = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]).astype(np.float32)
        b = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]]).astype(np.float32)

        self.conv_grad_x=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_grad_x.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.conv_grad_y=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_grad_y.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, probs, labels):
        total_boundary_loss = 0.0
        global G_result_norm
        global G_gt_norm
        num_classes = probs.size(1)
        for i in range(1,num_classes):
            input_y = probs[:,i,:,:].unsqueeze(1)
            input_x = labels[:,i,:,:].unsqueeze(1)

            G_x_result = self.conv_grad_x(input_x)
            G_y_result = self.conv_grad_y(input_x)
            G_result=torch.sqrt(torch.pow(G_x_result,2)+ torch.pow(G_y_result,2)+self.smooth)

            G_x_gt = self.conv_grad_x(input_y)
            G_y_gt = self.conv_grad_y(input_y)
            G_gt=torch.sqrt(torch.pow(G_x_gt,2)+ torch.pow(G_y_gt,2)+self.smooth)

            '''
            if((torch.max(G_result) != 0) and (torch.min(G_result) != 0)):
                G_result_norm = ((G_result - torch.min(G_result)) / (torch.max(G_result) - torch.min(G_result)))
            else:
                G_result_norm = G_result

            if((torch.min(G_gt) != 0) and (torch.max(G_gt) != 0)):
                G_gt_norm = ((G_gt - torch.min(G_gt)) / (torch.max(G_gt) - torch.min(G_gt)))
            else:
                G_gt_norm = G_gt
            '''

#            boundary_loss = F.mse_loss(G_result, G_gt, reduction='mean')
            boundary_loss = self.mse_loss(G_result, G_gt)
            total_boundary_loss = total_boundary_loss + boundary_loss
        #total_boundary_loss /= num_classes

        return total_boundary_loss


class DistanceTransformLoss(nn.Module):
    def __init__(self, dt_conv):
        super(DistanceTransformLoss, self).__init__()
        self.dt_conv = dt_conv
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, probs, labels):
        dt_loss = 0.0

        ## Distance transform for prob and label maps
        #probs = probs.half()
        probs_dt = self.dt_conv(probs)
        labels_dt = self.dt_conv(labels)

        dt_loss += self.mse_loss(probs_dt, labels_dt)
        dt_loss = dt_loss.float()

        return dt_loss


class CombinedLoss(nn.Module):
    def __init__(self, dt_conv, alpha, if_auxloss=1):
        super(CombinedLoss, self).__init__()
        self.auxloss = if_auxloss
        self.alpha = alpha
        self.dt_conv = dt_conv
        self.dt_loss_fun = DistanceTransformLoss(dt_conv)
        self.boundary_loss_fun = BoundaryLoss()
        self.dice_loss_fun = DiceLoss()

    def forward(self, result, label):
        labels = label.float()
        probs = F.softmax(result,1)

        boundary_loss = self.boundary_loss_fun(probs, labels)
        dt_loss = self.dt_loss_fun(probs, labels)
        dice_loss_labels, dice_loss = self.dice_loss_fun(probs, labels)
        total_loss = (1-self.alpha)*dice_loss + self.alpha*dt_loss

        if self.auxloss == 1:
            return total_loss, dice_loss_labels, dice_loss, dt_loss
        elif self.auxloss == 0:
            return dice_loss_labels, dice_loss
