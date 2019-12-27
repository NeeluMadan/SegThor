import os
import torch
import random
import numpy as np
from numpy import zeros

from PIL import Image
from skimage import io
from scipy import ndimage

from skimage import transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms as T
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torchvision.transforms import functional as F


def to_one_hot(mask, n_class):
    """
    Transform a mask to one hot
    Args:
        mask:
        n_class: number of class for segmentation
    Returns:
        y_one_hot: one hot mask
    """
    y_one_hot = torch.zeros((n_class, mask.shape[1], mask.shape[2]))
    y_one_hot = y_one_hot.scatter(0, mask, 1).long()
    return y_one_hot


class Rescale:
    def __init__(self, output_size, model='early_concat', labeled=True):
        self.output_size = output_size
        self.model = model
        self.labeled = labeled

    def __call__(self, sample):
        if self.model == 'late_concat':
            cs_size = self.output_size/4
        else:
            cs_size = self.output_size

        img = ndimage.interpolation.zoom(sample['image'], zoom=self.output_size, order=1, mode='constant')

        if self.labeled:
            mask = ndimage.interpolation.zoom(sample['label'], zoom=self.output_size, order=0, mode='nearest')

        coarse_seg = np.zeros((int(sample['coarse_segmentation'].shape[0]*cs_size), int(sample['coarse_segmentation'].shape[1]*cs_size), 5))
        for c in range(5):
            cs = ndimage.interpolation.zoom(sample['coarse_segmentation'][:,:,c], zoom=cs_size, order=1, mode='constant')
            coarse_seg[:,:,c] = cs

        if self.labeled:
            return {'image': img, 'label':mask, 'coarse_segmentation': coarse_seg}
        else:
            return {'image': img, 'coarse_segmentation': coarse_seg}


class Normalize:
    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        if self.labeled:
            mask = sample['label']

        coarse_segmentation = sample['coarse_segmentation']
        image = sample['image'].astype(np.float32)
        image = 2.*(image - np.min(image))/np.ptp(image)-1

        if self.labeled:
            return {'image': image, 'label': mask, 'coarse_segmentation': coarse_segmentation}
        else:
            return {'image': image, 'coarse_segmentation': coarse_segmentation}


class ToTensor:
    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        if len(sample['image'].shape) == 2:
            cs = sample['coarse_segmentation'].transpose((2, 0, 1))
            img = np.expand_dims(sample['image'], axis=0)
            if self.labeled:
                mask = np.expand_dims(sample['label'], axis=0)
        elif len(sample['image'].shape) == 3:
            img = sample['image'].transpose((2, 0, 1))
            cs = sample['coarse_segmentation'].transpose((2, 0, 1))
            if self.labeled:
                mask = sample['label'].transpose((2, 0, 1))
        else:
            print("Unsupported shape!")

        img = torch.from_numpy(img)
        cs = torch.from_numpy(cs)
        if self.labeled:
            mask = torch.from_numpy(mask)

        #concat_img = torch.cat((img,cs), dim=1)
        if self.labeled:
            return {'image': img, 'label':mask, 'coarse_segmentation': cs}
        else:
            return {'image': img, 'coarse_segmentation': cs}


class ToTensor2:
    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample, n_class=5):
        if len(sample['image'].shape) == 2:
            cs = sample['coarse_segmentation'].transpose((2, 0, 1))
            img = np.expand_dims(sample['image'], axis=0)
            if self.labeled:
                mask = np.expand_dims(sample['label'], axis=0)
        elif len(sample['image'].shape) == 3:
            cs = sample['coarse_segmentation'].transpose((2, 0, 1))
            img = np.expand_dims(sample['image'], axis=0)
            if self.labeled:
                mask = sample['label'].transpose((2, 0, 1))
        else:
            print("Unsupported shape!")

        cs = torch.from_numpy(cs)
        img = torch.from_numpy(img)
        if self.labeled:
            mask = torch.from_numpy(mask)
            mask = mask.type(torch.LongTensor)
            mask = to_one_hot(mask, n_class)

        if self.labeled:
            return {'image': img, 'label':mask, 'coarse_segmentation': cs}
        else:
            return {'image': img, 'coarse_segmentation': cs}



class RandomFlip(object):
    """ random horizontal or vertical flip """
    def __init__(self, prob=0.8):
        self.prob = prob

    def __call__(self, sample):
        image, mask, coarse_segmentation = sample['image'].astype(np.float32), sample['label'].astype(np.uint8), sample['coarse_segmentation'].astype(np.float32)
        x = random.uniform(0, 1)
        if x < self.prob:
            phase = random.randint(0, 1)
            image = np.flip(image, phase).copy()
            mask = np.flip(mask, phase).copy()
            for i in range(5):
                coarse_segmentation[:,:,i] = np.flip(coarse_segmentation[:,:,i], phase).copy()

        return {'image': image.astype(np.float32), 'label': mask, 'coarse_segmentation': coarse_segmentation}


class RandomRotation(object):
    """ random rotation (angle is randomly set as a multiplier of given angle) """
    def __init__(self, angle=90, prob=0.8):
        self.angle = angle
        self.prob = prob

    def __call__(self, sample):
        #image, mask, coarse_segmentation = sample['image'], sample['label'], sample['coarse_segmentation']
        image, mask, coarse_segmentation = sample['image'].astype(np.float32), sample['label'].astype(np.uint8), sample['coarse_segmentation'].astype(np.float32)
        rand_angle = random.randrange(0, 360, self.angle)
        x = random.uniform(0, 1)
        if x <= self.prob:
            image = transform.rotate(image, rand_angle, mode='reflect', preserve_range=True)
            mask = transform.rotate(mask, rand_angle, mode='reflect', preserve_range=True, order=0)
            for i in range(5):
                coarse_segmentation[:,:,i] = transform.rotate(coarse_segmentation[:,:,i], rand_angle, mode='reflect', preserve_range=True)


        '''
        fig=plt.figure()
        fig.add_subplot(2,2,1)
        plt.imshow(image)
        fig.add_subplot(2,2,2)
        plt.imshow(mask)
        fig.add_subplot(2,2,3)
        plt.imshow(coarse_segmentation[:,:,4])
        fig.add_subplot(2,2,4)
        plt.imshow(coarse_segmentation[:,:,2])
        plt.show()
        '''

        return {'image': image.astype(np.float32), 'label': mask, 'coarse_segmentation': coarse_segmentation}

class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """
    def __init__(self, p_deform=0.7, model='early_concat'):
        self.p_deform = p_deform
        self.model = model

    def __call__(self, sample):
        image, mask, coarse_segmentation = sample['image'].astype(np.float32), sample['label'].astype(np.uint8), sample['coarse_segmentation'].astype(np.float32)
        sigma = image.shape[0] * 0.05
        alpha = image.shape[0] * 0.25

        if np.random.rand() < self.p_deform:

            shape = image.shape
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode = 'constant', cval = 0) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode = 'constant', cval = 0) * alpha

            x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
            indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]

            image = map_coordinates(image, indices, order=2).reshape(shape)
            mask = map_coordinates(mask, indices, order=0).reshape(shape)
    
            '''
            if self.model == 'late_concat':
                coarse_segment = zeros([128, 128, 5])
                for c in range(5):
                    cs = ndimage.interpolation.zoom(sample['coarse_segmentation'][:,:,c], zoom=0.25, order=1, mode='constant')
                    coarse_segment[:,:,c] = cs
                coarse_segmentation = coarse_segment
            print("cs shape", coarse_segmentation.shape)
            '''

            shape_cs = coarse_segmentation.shape
            for i in range(coarse_segmentation.shape[2]):
                coarse_segmentation[:,:,i] = map_coordinates(coarse_segmentation[:,:,i], indices, order=0).reshape(shape_cs)

        sample = {'image': image, 'label': mask, 'coarse_segmentation': coarse_segmentation}

        return sample
