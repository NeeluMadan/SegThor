import cv2
import os
import torch
import skimage
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from data_processing.transformations import JointTransform2D, Rescale, ToTensor, Normalize

#####################################################################################
"""
		Preparing dataset
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

class SegThorDataset(Dataset):
    def __init__(self, datapath, phase, patient=None, transform=None, file_list=None):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.datapath = datapath
        self.transform = transform
        self.file_list = file_list
        self.patient = patient
        folder = datapath
        self.images = []
        self.masks = []
        self.num_slice = 0

        print("phase = ", self.phase)
        if self.phase == 'test':
            raw_img = os.path.join(folder, self.patient)   # Reading nifti image
            raw_itk = sitk.ReadImage(raw_img)
            raw_volume_array = sitk.GetArrayFromImage(raw_itk)
            raw_volume_array = truncated_range(raw_volume_array)

            for s in range(0, raw_volume_array.shape[0]):
                raw_slice_array = raw_volume_array[s,:,:]
                self.images.append(raw_slice_array)

        else: 
            for patient in tqdm(os.listdir(folder)):
                if not patient in self.file_list:
                    continue

                raw_img = os.path.join(folder, patient, patient+'.nii.gz')   # Reading nifti image
                raw_itk = sitk.ReadImage(raw_img)
                raw_volume_array = sitk.GetArrayFromImage(raw_itk)
                raw_volume_array = truncated_range(raw_volume_array)
    
                label_img = os.path.join(folder, patient, 'GT.nii.gz')       # Reading Ground Truth labels
                label_itk = sitk.ReadImage(label_img)
                label_volume_array = sitk.GetArrayFromImage(label_itk)                
                label_volume_array[np.where(label_volume_array > 1)] = 0
                self.num_slice = raw_volume_array.shape[0]

                # Appending input and GTi images into an array
                for s in range(0, raw_volume_array.shape[0]):
                    raw_slice_array = raw_volume_array[s,:,:]
                    self.images.append(raw_slice_array)
    
                    label_slice_array = label_volume_array[s,:,:]                    
                    self.masks.append(label_slice_array)
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self,item):

        if self.phase == 'test':
            image = self.images[item]
            sample = {'image': image}
    
            if self.transform:
                sample = self.transform(sample)
    
            return sample

        else:
            image, labels, num_slice = self.images[item], self.masks[item], self.num_slice
            sample = {'image': image, 'label': labels}

            if self.transform:
                sample = self.transform(sample)
    
            if self.phase == 'val':
                sample['num_slice'] = num_slice

            return sample

def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255.

if __name__ == "__main__":

    '''
    ## Loading data for testing phase
    segthor_dataset = SegThorDataset(datapath="/home/WIN-UNI-DUE/smnemada/Master_Thesis/SegThor/data/test",
                                     patient='Patient_58.nii.gz',
                                     phase='test',
                                     transform=transforms.Compose([
                                         Rescale(1.0, labeled=False),
                                         Normalize(labeled=False),
    #                                     ToTensor(labeled=False)
                                    ]))
    
    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]
        
    #    print(i, sample['image'].size())
        plt.imshow(sample['image'])
        plt.show()
        if i == 50:
            break

    '''
#    '''
    ## Loading data for training phase
    files = ['Patient_01', 'Patient_02', 'Patient_03', 'Patient_04', 'Patient_05', 'Patient_06', 'Patient_07', 'Patient_08', 'Patient_09', 'Patient_10', 'Patient_11', 'Patient_12', 'Patient_13', 'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17', 'Patient_18', 'Patient_19', 'Patient_20', 'Patient_21', 'Patient_27', 'Patient_28', 'Patient_29', 'Patient_30', 'Patient_31', 'Patient_32', 'Patient_33', 'Patient_34', 'Patient_35', 'Patient_36', 'Patient_37', 'Patient_38', 'Patient_39', 'Patient_40', 'Patient_22', 'Patient_23', 'Patient_24', 'Patient_25', 'Patient_26']

    segthor_dataset = SegThorDataset(datapath="../../../data/train",
                                     phase='train',
                                     transform=transforms.Compose([
                                         Rescale(1.0, labeled=True),
                                         Normalize(labeled=True),
                                         JointTransform2D(),
                                         ToTensor(labeled=True)
                                    ]), file_list=files)
    
    print(len(segthor_dataset))
    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]
        
        print(i, sample['image'].size(), sample['label'].size())
        if i == 5:
            break
#    '''
