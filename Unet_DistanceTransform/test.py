import os
import torch
from pylab import *
import numpy as np
from numpy import zeros
from tqdm import tqdm
import torch.nn as nn
from models.model import UNet
from scipy import ndimage
import SimpleITK as sitk
from numpy import ndarray
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from data_processing.data_loader import SegThorDataset
from data_processing.transformations import JointTransform2D, Rescale, ToTensor, Normalize
from scipy.ndimage import label

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def getLargestCC(segmentation_mask, num_labels=5):

    # Let us create a binary mask.
    # It is 0 everywhere `segmentation_mask` is 0 and 1 everywhere else.
    binary_mask = segmentation_mask.copy()
    binary_mask[binary_mask != 0] = 1
    print("unique numpy values: ", np.unique(binary_mask))
    
    # Now, we perform region labelling. This way, every connected component
    # will have their own colour value.
    labelled_mask, num_labels = label(binary_mask)

    # Let us now remove all the too small regions.
    refined_mask = segmentation_mask.copy()
    minimum_cc_sum = 1000
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
    test_path = '../../../data/test'
    for patient in tqdm(os.listdir(test_path)): 
        count = 0
        area = 0
        
        file = patient
        x = file.split(".")
        filename = x[0] + '.' + x[1]
        patient_id = x[0]

        print("Saving test data for patient = ", patient)
        test_set = SegThorDataset(test_path,
                                  patient=patient, 
                                  phase='test',
                                  transform=transforms.Compose([
                                         Rescale(0.5, labeled=False),
                                         Normalize(labeled=False),
                                         ToTensor(labeled=False)
                                  ]))

        test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                                  batch_size=1, 
                                                  shuffle=False)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = torch.load("models/model.pt")
        model.eval()
        model.to(device)

        segment_map =  os.path.join(test_path, "../coarse_segmentation", patient_id+'.npy')
        seg_map = np.load(segment_map).astype(np.float32)
        seg_map = np.argmax(seg_map, axis=3)
        
        seg_without_esophagus = np.where(seg_map==1, 0, seg_map)
        seg_vol_2d = zeros([len(test_set),  512, 512])

        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):     
                images = sample['image'].to(device, dtype=torch.float)        

                ## predicted output as tensor
                output_esophagus = model(images)

                ## Tensor to numpy conversion and assigining class to the max prediction
                images = tensor_to_numpy(images)            
                max_idx_esophagus = torch.argmax(output_esophagus, 1, keepdim=True)
                max_idx_esophagus = tensor_to_numpy(max_idx_esophagus)
                slice_v = max_idx_esophagus[:,:]   

                slice_v = ndimage.interpolation.zoom(slice_v, zoom=2, order=0, mode='nearest', prefilter=True)
                seg_vol_2d[count,:,:] = slice_v
                count = count + 1

            ##  Copying dummy labels from classes other that esophagus for online evaluation on segthor website
            Segmentation_output = np.where(seg_vol_2d==0, seg_without_esophagus, seg_vol_2d)
            ## Connected component analysis
            segmentation_largestCC = getLargestCC(Segmentation_output)

            ## Converting numpy array to nifti image
            segmentation = sitk.GetImageFromArray(segmentation_largestCC, isVector=False)
            sitk.WriteImage(sitk.Cast( segmentation, sitk.sitkUInt8 ), filename, True) 
            
if __name__ == "__main__":
    test()
