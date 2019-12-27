import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary

#####################################################################################
"""
		Fixed convolutional layer computing distance transform 
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

class FixedConv2d(nn.Module):
    def __init__(self, size, prec):
        super(FixedConv2d, self).__init__()
        self.prec = prec
        A, B = np.mgrid[0:size[0]+1, 0:size[1]+1].astype(np.float64)
        A = A - size[0]/2
        B = B - size[1]/2
        f = np.sqrt(A*A + B*B)
        f = np.exp(-f/self.prec)
        f = f.reshape(1, 1, f.shape[0], f.shape[1])
        f = torch.from_numpy(f)
        self.f = nn.Parameter(data=torch.DoubleTensor(f), requires_grad=False)
        padding = math.floor(self.f.size(2)/2)
        self.const_padding = nn.ConstantPad2d((padding, padding, padding, padding), 1)

    def forward(self, x):
#        x = x[:,0,:,:].double()
        x = x[:,0,:,:].double()
        x = x.unsqueeze_(1)
        input_x = self.const_padding(x)
        out = F.conv2d(input_x, self.f)
        out = -self.prec*torch.log(out)
        return out
