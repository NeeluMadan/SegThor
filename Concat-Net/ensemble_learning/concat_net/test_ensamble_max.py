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
from utils import Rescale, Normalize, ToTensor
from scipy.ndimage import label

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def getLargestCC(segmentation_mask, num_labels=5):

    # Let us create a binary mask.
    # It is 0 everywhere `segmentation_mask` is 0 and 1 everywhere else.
    binary_mask = segmentation_mask.copy()
    binary_mask[binary_mask != 0] = 1

    # Now, we perform region labelling. This way, every connected component
    # will have their own colour value.
    labelled_mask, num_labels = label(binary_mask)

    # Let us now remove all the too small regions.
    refined_mask = segmentation_mask.copy()
    minimum_cc_sum = 2000
    for curr_label in range(num_labels):
        if np.sum(refined_mask[labelled_mask == curr_label]) < minimum_cc_sum:
            refined_mask[labelled_mask == curr_label] = 0

    return refined_mask


def tensor_to_numpy(tensor):
    t_numpy = tensor.cpu().numpy()
    t_numpy = np.transpose(t_numpy, [0, 2, 3, 1])
    t_numpy = np.squeeze(t_numpy)

    return t_numpy


def test():
    test_path = "../../data/test_train"
    for patient in tqdm(os.listdir(test_path)): 
        count = 0
        area = 0
        
        file = patient
        x = file.split(".")
        filename = x[0] + '.' + x[1]
        patient_name = x[0]

        ## Checking result after each epoch
        test_set = SegThorDataset("../../data/test_train", patient = patient_name, phase='test',
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

        model1 = torch.load("models/model1.pt")
        model1.eval()
        model1.to(device)
        
        model2 = torch.load("models/model2.pt")
        model2.eval()
        model2.to(device)
        
        model3 = torch.load("models/model3.pt")
        model3.eval()
        model3.to(device)

        model4 = torch.load("models/model4.pt")
        model4.eval()
        model4.to(device)

        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):     
                images, cs = sample['image'].to(device, dtype=torch.float), sample['coarse_segmentation'].to(device, dtype=torch.float)
                output_1 = model1(images, cs)
                output_2 = model2(images, cs)
                output_3 = model3(images, cs)
                output_4 = model4(images, cs)

                output_ensamble = torch.zeros(1,5,512,512)
                output_ensamble[0,0,:,:] = torch.max(output_1[0,0,:,:],torch.max(output_2[0,0,:,:],torch.max(output_3[0,0,:,:],output_4[0,0,:,:])))
                output_ensamble[0,1,:,:] = torch.max(output_1[0,1,:,:],torch.max(output_2[0,1,:,:],torch.max(output_3[0,1,:,:],output_4[0,1,:,:])))
                output_ensamble[0,2,:,:] = torch.max(output_1[0,2,:,:],torch.max(output_2[0,2,:,:],torch.max(output_3[0,2,:,:],output_4[0,2,:,:])))
                output_ensamble[0,3,:,:] = torch.max(output_1[0,3,:,:],torch.max(output_2[0,3,:,:],torch.max(output_3[0,3,:,:],output_4[0,3,:,:])))
                output_ensamble[0,4,:,:] = torch.max(output_1[0,4,:,:],torch.max(output_2[0,4,:,:],torch.max(output_3[0,4,:,:],output_4[0,4,:,:])))

                images = tensor_to_numpy(images)            
                max_idx = torch.argmax(output_ensamble, 1, keepdim=True)
                max_idx = tensor_to_numpy(max_idx)
                          
                slice_v = max_idx[:,:]   
                seg_vol_2d[count,:,:] = slice_v
                count = count + 1

            segmentation_output = getLargestCC(seg_vol_2d)
            segmentation = sitk.GetImageFromArray(segmentation_output, isVector=False)
            sitk.WriteImage(sitk.Cast( segmentation, sitk.sitkUInt8 ), filename, True) 

            
if __name__ == "__main__":
    test()
