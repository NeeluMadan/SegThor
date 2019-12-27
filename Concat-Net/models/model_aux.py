import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#####################################################################################
"""
		Concat coarse information using auxiliary branch without skip connections
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

class UNetAux(nn.Module):
    def __init__(self, kernel_size=5, padding=2):
        super(UNetAux, self).__init__()
        print("RUNNING aux")
        conv_depths = [64, 128, 256, 512, 1024]
        conv_depths_cs = [8, 16, 32, 64, 128]

        ## Extracting features from just input image
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
        
        ## Extracting features from CS+input
        self.conv1_1_cs = nn.Conv2d(6, conv_depths_cs[0], kernel_size=kernel_size, padding=padding)
        self.conv1_1_cs_bn = nn.BatchNorm2d(conv_depths_cs[0])
        self.conv1_2_cs = nn.Conv2d(conv_depths_cs[0], conv_depths_cs[0], kernel_size=kernel_size, padding=padding)
        self.conv1_2_cs_bn = nn.BatchNorm2d(conv_depths_cs[0])
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1_cs = nn.Conv2d(conv_depths_cs[0], conv_depths_cs[1], kernel_size=kernel_size, padding=padding)
        self.conv2_1_cs_bn = nn.BatchNorm2d(conv_depths_cs[1])
        self.conv2_2_cs = nn.Conv2d(conv_depths_cs[1], conv_depths_cs[1], kernel_size=kernel_size, padding=padding)
        self.conv2_2_cs_bn = nn.BatchNorm2d(conv_depths_cs[1])
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1_cs = nn.Conv2d(conv_depths_cs[1], conv_depths_cs[2], kernel_size=kernel_size, padding=padding)
        self.conv3_1_cs_bn = nn.BatchNorm2d(conv_depths_cs[2])
        self.conv3_2_cs = nn.Conv2d(conv_depths_cs[2], conv_depths_cs[2], kernel_size=kernel_size, padding=padding)
        self.conv3_2_cs_bn = nn.BatchNorm2d(conv_depths_cs[2])
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1_cs = nn.Conv2d(conv_depths_cs[2], conv_depths_cs[3], kernel_size=kernel_size, padding=padding)
        self.conv4_1_cs_bn = nn.BatchNorm2d(conv_depths_cs[3])
        self.conv4_2_cs = nn.Conv2d(conv_depths_cs[3], conv_depths_cs[3], kernel_size=kernel_size, padding=padding)
        self.conv4_2_cs_bn = nn.BatchNorm2d(conv_depths_cs[3])
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1_cs = nn.Conv2d(conv_depths_cs[3], conv_depths_cs[4], kernel_size=kernel_size, padding=padding)
        self.conv5_1_cs_bn = nn.BatchNorm2d(conv_depths_cs[4])
        self.conv5_2_cs = nn.Conv2d(conv_depths_cs[4], conv_depths_cs[4], kernel_size=kernel_size, padding=padding)
        self.conv5_2_cs_bn = nn.BatchNorm2d(conv_depths_cs[4])

        #To reduce the size of feature map to half for both coarse_segmentation and input
        self.conv_reduced = nn.Conv2d(conv_depths[4]+conv_depths_cs[4], conv_depths[4], kernel_size=1)
        self.conv_reduced_bn = nn.BatchNorm2d(conv_depths[4])
        self.conv5_2 = nn.Conv2d(conv_depths[4], conv_depths[4], kernel_size=kernel_size, padding=padding)
        self.conv5_2_bn = nn.BatchNorm2d(conv_depths[4])

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
        

    def forward(self, x, seg):
        ## Concatenate the input and coarse segmentation  accross the last axis
        # Extracting features from cs segmentation map
        combined_in = torch.cat((seg, x), dim=1)
        conv1_seg = F.relu(self.conv1_1_cs_bn(self.conv1_1_cs(combined_in)))
        conv1_seg = F.relu(self.conv1_2_cs_bn(self.conv1_2_cs(conv1_seg)))
        pool1_seg = self.maxpool1(conv1_seg)

        conv2_seg = F.relu(self.conv2_1_cs_bn(self.conv2_1_cs(pool1_seg)))
        conv2_seg = F.relu(self.conv2_2_cs_bn(self.conv2_2_cs(conv2_seg)))
        pool2_seg = self.maxpool2(conv2_seg)

        conv3_seg = F.relu(self.conv3_1_cs_bn(self.conv3_1_cs(pool2_seg)))
        conv3_seg = F.relu(self.conv3_2_cs_bn(self.conv3_2_cs(conv3_seg)))
        pool3_seg = self.maxpool3(conv3_seg)

        conv4_seg = F.relu(self.conv4_1_cs_bn(self.conv4_1_cs(pool3_seg)))
        conv4_seg = F.relu(self.conv4_2_cs_bn(self.conv4_2_cs(conv4_seg)))
        pool4_seg = self.maxpool4(conv4_seg)

        conv5_seg = F.relu(self.conv5_1_cs_bn(self.conv5_1_cs(pool4_seg)))
        conv5_seg = F.relu(self.conv5_2_cs_bn(self.conv5_2_cs(conv5_seg)))


        # Extracting feature from input image
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

        ## Fusing coarse segmentation and the input information
        combined_feature_maps = torch.cat((conv5, conv5_seg), dim=1)
        combine_features_reduced = F.relu(self.conv_reduced_bn(self.conv_reduced(combined_feature_maps)))
        combine_features = F.relu(self.conv5_2_bn(self.conv5_2(combine_features_reduced)))

        ## Decoder path
        up6 = torch.cat((self.conv5_t(combine_features), conv4), dim=1)
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

        return F.softmax(self.conv10(conv9),1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = UNetAux().to(device)
    summary(model, [(1, 512, 512), (5, 512, 512)])
