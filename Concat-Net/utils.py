import os
import torch
import numpy as np
import torch.nn as nn

#####################################################################################
"""
		Containing utility functions
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

def setgpu(gpus):
    '''
    Set GPUs
    '''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))

def get_lr(epoch, lr = 0.025):
    '''
    Decay learning rate for each epoch
    Args:
        epoch: current epoch number
        learning rate: current learning rate
    '''
    if epoch > 0:
        lr = lr * 0.95
    return lr

## Converting  a tensor to numpy array
def tensor_to_numpy(tensor):
    t_numpy = tensor.detach().cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy

# Trying to implement He weight initialization for function ReLu
def weight_init(m):
    '''
    Initializing weights for the neural network
    '''
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)


## The evaluation score for  validation set computed on entire CT volume
def eval_dice(result, target, num_classes=5):
    """
    Pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    epsilon = 1e-6
    loss_label =  np.zeros(5)
    Loss = 0.0

    for j in range(0, num_classes):
        result_square_sum = torch.sum(result[:, j, :, :])
        target_square_sum = torch.sum((target[:, j, :, :]).float())
        intersect = torch.sum(result[:, j, :, :] * (target[:, j, :, :]).float())
        dice = (2 * intersect + epsilon) / (result_square_sum + target_square_sum + epsilon)
        Loss += (1 - dice)
        loss_label[j] = (1-dice).detach().cpu().numpy()

    Loss = Loss.detach().cpu().numpy()
    return loss_label, Loss

