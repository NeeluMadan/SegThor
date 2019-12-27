import numpy as np
import cv2
import os
import torch
import ntpath
import os.path as osp
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from torch.utils.data import Dataset
from torchvision import transforms

from data_processing.transformations import Rescale, Normalize, ToTensor, ToTensor2, RandomFlip, RandomRotation

class SegThorDataset(Dataset):
    def __init__(self, datapath, phase,  patient=None, transform=None, file_list=None):

        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.datapath = datapath
        self.datapath_list = []
        self.images = []
        self.masks = []
        self.cs = []
        self.patient = patient
        self.transform = transform
        self.file_list = file_list
        folder = datapath
        num_slice = []
        patient_list = []
        idx_list = []
        self.num_slice = 0

        if not self._check_exists():
            raise RuntimeError("dataset not found")

        print("phase = ", self.phase)
        if self.phase == 'test':
            raw_img = os.path.join(folder, self.patient+'.nii.gz')   # Reading nifti image
            raw_itk = sitk.ReadImage(raw_img)
            raw_volume_array = sitk.GetArrayFromImage(raw_itk)
            raw_volume_array = truncated_range(raw_volume_array)

            segment_map =  os.path.join(folder, "../coarse_segmentation", self.patient+'.npy')
            seg_map = np.load(segment_map).astype(np.float32)
            seg_map = np.true_divide(seg_map, 8)

            self.num_slice = raw_itk.GetDepth()
            for s in range(0, raw_volume_array.shape[0]):
                self.images.append(raw_volume_array[s,:,:])
                self.cs.append(seg_map[s,:,:,:])

        else:
            for f in tqdm(os.listdir(datapath)):
                if not f in self.file_list:
                    continue

                patient = os.path.join(folder, f)
                patient_list.append(patient)
                raw_img = os.path.join(folder, f, f+'.nii.gz')
                nifti_image = sitk.ReadImage(raw_img)
                raw_volume_array = sitk.GetArrayFromImage(nifti_image)
                raw_volume_array = truncated_range(raw_volume_array)

                label_img = os.path.join(folder, f, 'GT.nii.gz')
                label_itk = sitk.ReadImage(label_img)
                label_volume_array = sitk.GetArrayFromImage(label_itk)

                segment_map =  os.path.join(folder, "../coarse_segmentation", f+'.npy')
                seg_map = np.load(segment_map).astype(np.float32)
                seg_map = np.true_divide(seg_map, 8)

                self.num_slice = nifti_image.GetDepth()
                for s in range(0, raw_volume_array.shape[0]):
                    self.images.append(raw_volume_array[s,:,:])
                    self.masks.append(label_volume_array[s,:,:])
                    self.cs.append(seg_map[s,:,:,:])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,item):

        if self.phase == 'test':
            image, cs, num_slice = self.images[item], self.cs[item], self.num_slice
            sample = {'image': image, 'coarse_segmentation': cs}

            torch.manual_seed(1)
            if self.transform:
                sample = self.transform(sample)
            sample['num_slice'] = num_slice

#            image, mask, coarse_segmentation = sample['image'].astype(np.float32), sample['coarse_segmentation'].astype(np.float32)
#            sample = {'image': image, 'coarse_segmentation': cs, 'num_slice': num_slice}

            return sample

        else:
            image, labels, cs,  num_slice = self.images[item], self.masks[item], self.cs[item], self.num_slice
            sample = {'image': image, 'label': labels, 'coarse_segmentation': cs}

            torch.manual_seed(1)
            if self.transform:
                sample = self.transform(sample)
            sample['num_slice'] = num_slice

#            image, mask, coarse_segmentation = sample['image'].astype(np.float32), sample['label'].astype(np.uint8), sample['coarse_segmentation'].astype(np.float32)
#            sample = {'image': image, 'label': labels, 'coarse_segmentation': cs, 'num_slice': num_slice}

            return sample

    def _check_exists(self):
        return osp.exists(osp.join(self.datapath))

def truncated_range(img):
    max_hu = 384
    min_hu = -384
    img[np.where(img > max_hu)] = max_hu
    img[np.where(img < min_hu)] = min_hu
    return (img - min_hu) / (max_hu - min_hu) * 255.

if __name__ == "__main__":

#    files = ['Patient_01', 'Patient_02', 'Patient_03', 'Patient_04', 'Patient_05', 'Patient_06', 'Patient_07', 'Patient_08', 'Patient_09', 'Patient_10', 'Patient_11', 'Patient_12', 'Patient_13', 'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17', 'Patient_18', 'Patient_19', 'Patient_20', 'Patient_21', 'Patient_22', 'Patient_23', 'Patient_24', 'Patient_25', 'Patient_26', 'Patient_27', 'Patient_28', 'Patient_29', 'Patient_30', 'Patient_31', 'Patient_32', 'Patient_33', 'Patient_34', 'Patient_35', 'Patient_36', 'Patient_37', 'Patient_38', 'Patient_39', 'Patient_40']
    files = ['Patient_01']
    segthor_dataset = SegThorDataset(datapath="../../data/train",
                                     phase='train',
                                     transform=transforms.Compose([Rescale(1.0, labeled=True),
                                         Normalize(labeled=True),
                                         RandomFlip(),
                                         RandomRotation(),
                                         ToTensor2(labeled=True)
                                         ]), file_list=files)

    for i in range(len(segthor_dataset)):
        sample = segthor_dataset[i]

        print(i, sample['image'].size(), sample['label'].size(), sample['coarse_segmentation'].size())
        print( sample['image'].type(),  sample['label'].type(),  sample['coarse_segmentation'].type() )
        if i == 5:
            break
