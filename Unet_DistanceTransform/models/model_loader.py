import torch
from torch import nn
from models.model_unet import UNet
from models.loss import CombinedLoss
from models.distance_transform_layer import FixedConv2d

#####################################################################################
"""
		Load approprate model and loss function
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

def get_model(model_name = 'early_concat', loss_name = 'CombinedLoss', alpha=None, if_auxloss=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_conv = FixedConv2d((128, 128), prec=0.35).to(device)

    if loss_name == 'CombinedLoss':
        loss = CombinedLoss(dt_conv, alpha=alpha, if_auxloss=if_auxloss)

    if model_name == 'unet_mtl':
        net = UNet()
    elif model_name == 'vanilla_unet':
        net = UNet()
    return net, loss
