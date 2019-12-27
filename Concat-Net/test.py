import os
import torch
from pylab import *
import numpy as np
from numpy import zeros
from tqdm import tqdm
import torch.nn as nn
from models.model_unet import UNet
from scipy import ndimage
import SimpleITK as sitk
from numpy import ndarray
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from data_processing.data_loader import SegThorDataset
from data_processing.transformations import Rescale, Normalize, ToTensor, ToTensor2, RandomFlip, RandomRotation

#####################################################################################
"""
		Testing the model and making predictions
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

def tensor_to_numpy(tensor):
    t_numpy = tensor.cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy


def test():
    test_path = "../../data/test"
    for patient in tqdm(os.listdir(test_path)):
        count = 0
        area = 0

        file = patient
        x = file.split(".")
        filename = x[0] + '.' + x[1]
        patient_name = x[0]

        ## Checking result after each epoch
        SegThorValTrans = transforms.Compose([Rescale(1.0, labeled=False), Normalize(labeled=False), ToTensor2(labeled=False)])
        test_set = SegThorDataset("../../../data/test", patient = patient_name, phase='test', transform=SegThorValTrans)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

        seg_vol_2d = zeros([len(test_set), 512, 512])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = torch.load("models/model4.pt")
        model.eval()
        model.to(device)

        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                images, cs = sample['image'].to(device, dtype=torch.float), sample['coarse_segmentation'].to(device, dtype=torch.float)
                outputs = model(images, cs)

                images = tensor_to_numpy(images)
                max_idx = torch.argmax(outputs, 1, keepdim=True)
                max_idx = tensor_to_numpy(max_idx)

                slice_v = max_idx[:, :]
                seg_vol_2d[count, :, :] = slice_v
                count = count + 1

            segmentation = sitk.GetImageFromArray(seg_vol_2d, isVector=False)
            print(segmentation.GetSize())
            sitk.WriteImage(sitk.Cast(segmentation, sitk.sitkUInt8), filename, True)


if __name__ == "__main__":
    test()
