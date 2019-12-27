import os
import torch
import numpy as np
import torch.nn as nn

def setgpu(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))

def get_lr(epoch, lr = 0.085):
    if epoch > 0:
        lr = lr * 0.95
    return lr

def tensor_to_numpy(tensor):
    t_numpy = tensor.detach().cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy


# Trying to implement He weight initialization for function ReLu
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)