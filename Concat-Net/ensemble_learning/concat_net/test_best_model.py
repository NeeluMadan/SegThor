import os
import torch
from pylab import *
import numpy as np
from numpy import zeros
from tqdm import tqdm
import torch.nn as nn
from model import UNet
from scipy import ndimage
import SimpleITK as sitk
from numpy import ndarray
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from data import SegThorDataset
from utils import Rescale, Normalize, ToTensor, ToTensor2, RandomFlip, RandomRotation

os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
        test_set = SegThorDataset("../../data/test", patient = patient_name, phase='test',
                                       transform=transforms.Compose([
                                           Rescale(1.0, labeled=False),
                                           Normalize(labeled=False),
                                           ToTensor(labeled=False)
                                       ]))

        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                 batch_size=1,
                                                 shuffle=False)

        seg_vol_2d = zeros([len(test_set),  512, 512])
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
                          
              #  for k in range(outputs.size(0)): 
              #  print(max_idx.shape)
                slice_v = max_idx[:,:]   
                #slice_v = slice_v.astype(float32)
                #slice_v = ndimage.interpolation.zoom(slice_v, zoom=2, order=0, mode='nearest', prefilter=True)
                seg_vol_2d[count,:,:] = slice_v
                count = count + 1
               
            segmentation = sitk.GetImageFromArray(seg_vol_2d, isVector=False)
            print(segmentation.GetSize())
            sitk.WriteImage(sitk.Cast( segmentation, sitk.sitkUInt8 ), filename, True) 

            
if __name__ == "__main__":
    test()
