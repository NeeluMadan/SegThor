import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

#####################################################################################
"""
		Concat coarse information using auxiliary branch with skip connections
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

class UNetAuxSkip(nn.Module):
    def __init__(self, kernel_size=5, padding=2):
        super(UNetAuxSkip, self).__init__()
        print("RUNNING aux with skip")
        conv_depths = [64, 128, 256, 512, 1024]
        conv_depths_cs = [8, 16, 32, 64, 128]

        ## Extracting features from just input image
        self.conv1_1 = nn.Conv2d(1, conv_depths[0], kernel_size=kernel_size, padding=padding) # Now the input contain 5 slices i.e. 5 for eac class segmentation and 1 is for input
        self.conv1_1_bn = nn.BatchNorm2d(conv_depths[0])
        #== fusion ==
        self.conv_1_fusion = nn.Conv2d(conv_depths_cs[0]+conv_depths[0], conv_depths[0], kernel_size=1)
        self.conv_1_fusion_bn = nn.BatchNorm2d(conv_depths[0])
        #== fusion ==
        self.conv1_2 = nn.Conv2d(conv_depths[0], conv_depths[0], kernel_size=kernel_size, padding=padding)
        self.conv1_2_bn = nn.BatchNorm2d(conv_depths[0])
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(conv_depths[0], conv_depths[1], kernel_size=kernel_size, padding=padding)
        self.conv2_1_bn = nn.BatchNorm2d(conv_depths[1])
        #== fusion ==
        self.conv_2_fusion = nn.Conv2d(conv_depths_cs[1]+conv_depths[1], conv_depths[1], kernel_size=1)
        self.conv_2_fusion_bn = nn.BatchNorm2d(conv_depths[1])
        #== fusion ==
        self.conv2_2 = nn.Conv2d(conv_depths[1], conv_depths[1], kernel_size=kernel_size, padding=padding)
        self.conv2_2_bn = nn.BatchNorm2d(conv_depths[1])
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(conv_depths[1], conv_depths[2], kernel_size=kernel_size, padding=padding)
        self.conv3_1_bn = nn.BatchNorm2d(conv_depths[2])
        #== fusion ==
        self.conv_3_fusion = nn.Conv2d(conv_depths_cs[2]+conv_depths[2], conv_depths[2], kernel_size=1)
        self.conv_3_fusion_bn = nn.BatchNorm2d(conv_depths[2])
        #== fusion ==
        self.conv3_2 = nn.Conv2d(conv_depths[2], conv_depths[2], kernel_size=kernel_size, padding=padding)
        self.conv3_2_bn = nn.BatchNorm2d(conv_depths[2])
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(conv_depths[2], conv_depths[3], kernel_size=kernel_size, padding=padding)
        self.conv4_1_bn = nn.BatchNorm2d(conv_depths[3])
        #== fusion ==
        self.conv_4_fusion = nn.Conv2d(conv_depths_cs[3]+conv_depths[3], conv_depths[3], kernel_size=1)
        self.conv_4_fusion_bn = nn.BatchNorm2d(conv_depths[3])
        #== fusion ==
        self.conv4_2 = nn.Conv2d(conv_depths[3], conv_depths[3], kernel_size=kernel_size, padding=padding)
        self.conv4_2_bn = nn.BatchNorm2d(conv_depths[3])
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(conv_depths[3], conv_depths[4], kernel_size=kernel_size, padding=padding)
        self.conv5_1_bn = nn.BatchNorm2d(conv_depths[4])
        #== fusion ==
        self.conv_5_fusion = nn.Conv2d(conv_depths_cs[4]+conv_depths[4], conv_depths[4], kernel_size=1)
        self.conv_5_fusion_bn = nn.BatchNorm2d(conv_depths[4])
        #== fusion ==
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
        ## fusion
        conv1_skip = torch.cat((conv1, conv1_seg), dim=1)
        conv1_fused = F.relu(self.conv_1_fusion_bn(self.conv_1_fusion(conv1_skip)))
        ## fusion
        conv1 = F.relu(self.conv1_2_bn(self.conv1_2(conv1_fused)))
        pool1 = self.maxpool1(conv1)

        conv2 = F.relu(self.conv2_1_bn(self.conv2_1(pool1)))
        ## fusion
        conv2_skip = torch.cat((conv2, conv2_seg), dim=1)
        conv2_fused = F.relu(self.conv_2_fusion_bn(self.conv_2_fusion(conv2_skip)))
        ## fusion
        conv2 = F.relu(self.conv2_2_bn(self.conv2_2(conv2_fused)))
        pool2 = self.maxpool2(conv2)

        conv3 = F.relu(self.conv3_1_bn(self.conv3_1(pool2)))
        ## fusion
        conv3_skip = torch.cat((conv3, conv3_seg), dim=1)
        conv3_fused = F.relu(self.conv_3_fusion_bn(self.conv_3_fusion(conv3_skip)))
        ## fusion
        conv3 = F.relu(self.conv3_2_bn(self.conv3_2(conv3_fused)))
        pool3 = self.maxpool3(conv3)

        conv4 = F.relu(self.conv4_1_bn(self.conv4_1(pool3)))
        ## fusion
        conv4_skip = torch.cat((conv4, conv4_seg), dim=1)
        conv4_fused = F.relu(self.conv_4_fusion_bn(self.conv_4_fusion(conv4_skip)))
        ## fusion
        conv4 = F.relu(self.conv4_2_bn(self.conv4_2(conv4_fused)))
        pool4 = self.maxpool4(conv4)

        conv5 = F.relu(self.conv5_1_bn(self.conv5_1(pool4)))
        ## fusion
        conv5_skip = torch.cat((conv5, conv5_seg), dim=1)
        conv5_fused = F.relu(self.conv_5_fusion_bn(self.conv_5_fusion(conv5_skip)))
        ## fusion
        conv5 = F.relu(self.conv5_2_bn(self.conv5_2(conv5_fused)))

        ## Decoder path
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

        return F.softmax(self.conv10(conv9),1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = UNetAuxSkip().to(device)
    summary(model, [(1, 512, 512), (5, 512, 512)])
