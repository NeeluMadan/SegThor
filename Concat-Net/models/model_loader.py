import torch
from torch import nn
from models.loss import CombinedLoss, DiceLoss, DiceLossWithImprovement, BoundaryLoss

#####################################################################################
"""
		Load approprate model and loss function
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################


def get_model(model_name='early_concat', with_improvement = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if with_improvement == 1:
        loss = DiceLossWithImprovement()
    elif with_improvement == 0:
        loss = DiceLoss()

    if model_name == 'early_concat':
        from models.model_early import UNetEarly
        net = UNetEarly()
    elif model_name == 'late_concat':
        from models.model_late import UNetLate
        net = UNetLate()
    elif model_name == 'aux_concat':
        from models.model_aux import UNetAux
        net = UNetAux()
    elif model_name == 'aux_skip_concat':
        from models.model_aux_skip import UNetAuxSkip
        net = UNetAuxSkip()
    elif model_name == 'vanilla_unet':
        from models.model_unet import UNet
        net = UNet()
    return net, loss
