import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary

#####################################################################################
"""
		Implementation of vanilla Unet in 2D 
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

class UNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(UNet, self).__init__()
        #conv_depths = [64, 128, 256, 512, 1024]
        conv_depths = [32, 64, 128, 256, 512]
        self.conv1_1 = nn.Conv2d(1, conv_depths[0], kernel_size=kernel_size, padding=padding) # Now the input contain 5 slices i.e. 5 for eac class segmentation and 1 is for input
        self.conv1_1_bn = nn.BatchNorm2d(conv_depths[0])
        self.conv1_2 = nn.Conv2d(conv_depths[0], conv_depths[0], kernel_size=kernel_size, padding=padding)
        self.conv1_2_bn = nn.BatchNorm2d(conv_depths[0])
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(conv_depths[0], conv_depths[1], kernel_size=kernel_size, padding=padding)
        self.conv2_1_bn = nn.BatchNorm2d(conv_depths[1])
        self.conv2_2 = nn.Conv2d(conv_depths[1], conv_depths[1], kernel_size=kernel_size, padding=padding)
        self.conv2_2_bn = nn.BatchNorm2d(conv_depths[1])
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(conv_depths[1], conv_depths[2], kernel_size=kernel_size, padding=padding)
        self.conv3_1_bn = nn.BatchNorm2d(conv_depths[2])
        self.conv3_2 = nn.Conv2d(conv_depths[2], conv_depths[2], kernel_size=kernel_size, padding=padding)
        self.conv3_2_bn = nn.BatchNorm2d(conv_depths[2])
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(conv_depths[2], conv_depths[3], kernel_size=kernel_size, padding=padding)
        self.conv4_1_bn = nn.BatchNorm2d(conv_depths[3])
        self.conv4_2 = nn.Conv2d(conv_depths[3], conv_depths[3], kernel_size=kernel_size, padding=padding)
        self.conv4_2_bn = nn.BatchNorm2d(conv_depths[3])
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(conv_depths[3], conv_depths[4], kernel_size=kernel_size, padding=padding)
        self.conv5_1_bn = nn.BatchNorm2d(conv_depths[4])
        self.conv5_2 = nn.Conv2d(conv_depths[4], conv_depths[4], kernel_size=kernel_size, padding=padding)
        self.conv5_2_bn = nn.BatchNorm2d(conv_depths[4])

        #To reduce the size of feature map to half for both coarse_segmentation and input
        self.conv5_t = nn.ConvTranspose2d(conv_depths[4], conv_depths[3], 2, stride=2)

        self.conv6_1 = nn.Conv2d(conv_depths[4], conv_depths[3], kernel_size=kernel_size, padding=padding)
        self.conv6_1_bn = nn.BatchNorm2d(conv_depths[3])
        self.conv6_2 = nn.Conv2d(conv_depths[3], conv_depths[3], kernel_size=kernel_size, padding=padding)
        self.conv6_2_bn = nn.BatchNorm2d(conv_depths[3])
        self.conv6_t = nn.ConvTranspose2d(conv_depths[3], conv_depths[2], 2, stride=2)

        self.conv7_1 = nn.Conv2d(conv_depths[3], conv_depths[2], kernel_size=kernel_size, padding=padding)
        self.conv7_1_bn = nn.BatchNorm2d(conv_depths[2])
        self.conv7_2 = nn.Conv2d(conv_depths[2], conv_depths[2], kernel_size=kernel_size, padding=padding)
        self.conv7_2_bn = nn.BatchNorm2d(conv_depths[2])
        self.conv7_t = nn.ConvTranspose2d(conv_depths[2], conv_depths[1], 2, stride=2)

        self.conv8_1 = nn.Conv2d(conv_depths[2], conv_depths[1], kernel_size=kernel_size, padding=padding)
        self.conv8_1_bn = nn.BatchNorm2d(conv_depths[1])
        self.conv8_2 = nn.Conv2d(conv_depths[1], conv_depths[1], kernel_size=kernel_size, padding=padding)
        self.conv8_2_bn = nn.BatchNorm2d(conv_depths[1])
        self.conv8_t = nn.ConvTranspose2d(conv_depths[1], conv_depths[0], 2, stride=2)

        self.conv9_1 = nn.Conv2d(conv_depths[1], conv_depths[0], kernel_size=kernel_size, padding=padding)
        self.conv9_1_bn = nn.BatchNorm2d(conv_depths[0])
        self.conv9_2 = nn.Conv2d(conv_depths[0], conv_depths[0], kernel_size=kernel_size, padding=padding)
        self.conv9_2_bn = nn.BatchNorm2d(conv_depths[0])

        self.conv10 = nn.Conv2d(conv_depths[0], 5, kernel_size=1)


    def forward(self, x):
        conv1 = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        conv1 = F.relu(self.conv1_2_bn(self.conv1_2(conv1)))
        pool1 = self.maxpool1(conv1)

        conv2 = F.relu(self.conv2_1_bn(self.conv2_1(pool1)))
        conv2 = F.relu(self.conv2_2_bn(self.conv2_2(conv2)))
        pool2 = self.maxpool2(conv2)

        conv3 = F.relu(self.conv3_1_bn(self.conv3_1(pool2)))
        conv3 = F.relu(self.conv3_2_bn(self.conv3_2(conv3)))
        pool3 = self.maxpool3(conv3)

        conv4 = F.relu(self.conv4_1_bn(self.conv4_1(pool3)))
        conv4 = F.relu(self.conv4_2_bn(self.conv4_2(conv4)))
        pool4 = self.maxpool4(conv4)

        conv5 = F.relu(self.conv5_1_bn(self.conv5_1(pool4)))
        conv5 = F.relu(self.conv5_2_bn(self.conv5_2(conv5)))

        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1_bn(self.conv6_1(up6)))
        conv6 = F.relu(self.conv6_2_bn(self.conv6_2(conv6)))

        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1_bn(self.conv7_1(up7)))
        conv7 = F.relu(self.conv7_2_bn(self.conv7_2(conv7)))

        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1_bn(self.conv8_1(up8)))
        conv8 = F.relu(self.conv8_2_bn(self.conv8_2(conv8)))

        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1_bn(self.conv9_1(up9)))
        conv9 = F.relu(self.conv9_2_bn(self.conv9_2(conv9)))

        return F.softmax(self.conv10(conv9), 1)
