import torch
import datetime
import glob, os
import argparse
from pylab import *
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
from numpy import ndarray
from scipy import ndimage
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import transforms
from tensorboardX import SummaryWriter
from data_processing.data_loader import SegThorDataset
#from models import UNet, FixedConv2d
from models.model_loader import get_model
from utils import setgpu, weight_init, tensor_to_numpy, get_lr
from data_processing.transformations import JointTransform2D, Rescale, ToTensor, Normalize, ElasticTransform

#####################################################################################
"""
		Main function 
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="1"
logdir = os.makedirs("models/logs", exist_ok=True)
writer = SummaryWriter(log_dir = logdir)

parser = argparse.ArgumentParser(description='SegTHOR Segmentation')
parser.add_argument(
    '--model_name',
    '-m_name',
    metavar='MODEL',
    default='early_fusion',
    help='model_name')

parser.add_argument(
    '--epochs',
    default=32,
    type=int,
    metavar='N',
    help='number of max epochs')

parser.add_argument(
    '-b',
    '--batch-size',
    default=4,
    type=int,
    metavar='N',
    help='mini-batch size (default: 4)')

parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='starting learning rate')

parser.add_argument(
    '--momentum', 
    default=0.9, 
    type=float, 
    metavar='M', 
    help='momentum term')

parser.add_argument(
    '--alpha',
    default=0.3,
    type=float,
    metavar='A',
    help='weight of regression loss')

parser.add_argument(
    '--weight-decay',
    '--wd',
    default=0.00001,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-5)')

parser.add_argument(
    '--save_dir',
    default='save_path/',
    type=str,
    metavar='SAVE',
    help='directory to save checkpoint (default: save_path)')

parser.add_argument(
    '--gpu', 
    default='all', 
    type=str, 
    metavar='N', 
    help='use gpu')

parser.add_argument(
    '--loss_name',
    default='DiceLoss',
    type=str,
    metavar='N',
    help='defines loss function')

parser.add_argument(
    '--data_path',
    default='../data/',
    type=str,
    metavar='N',
    help='data path')

parser.add_argument(
    '--if_auxloss',
    default=1,
    type=int,
    metavar='1(True) or 0(False)',
    help='if using multi-task learning')
            
def loss_from_distance_transform(results, labels, device, dt_conv):

#    results = result.half()
    labels = labels.float()

    transformed_result = dt_conv(results)
    transformed_label = dt_conv(labels)

    ## If i don't normalize it than it leads to the problem of exploding gradients
#    transformed_result = ((transformed_result - torch.min(transformed_result)) / (torch.max(transformed_result) - torch.min(transformed_result)))
#    transformed_label = ((transformed_label - torch.min(transformed_label)) / (torch.max(transformed_label) - torch.min(transformed_label)))

    loss_mse = F.mse_loss(transformed_result, transformed_label, reduction='mean')
    loss_mse = loss_mse.float()

    return loss_mse


def dice_loss(result, target, total_classes = 5):
    
    """
    Pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    
    """
    epsilon = 1e-6
    total_loss = 0.0     
    dice_per_class = 0.0
    loss_label =  np.zeros(5)
    weight = [0.2, 2, 0.4, 0.9, 0.8]

    for i in range(result.size(0)):
        Loss = []

        for j in range(0, total_classes):
            result_square_sum = torch.sum(result[i, j, :, :])
            target_square_sum = torch.sum((target[i, j, :, :]).float())
            intersect = torch.sum(result[i, j, :, :] * (target[i, j, :, :]).float())
            dice = (2 * intersect + epsilon) / (result_square_sum + target_square_sum + intersect + epsilon)
            dice_per_class = 1 - dice
            total_loss += dice_per_class/total_classes
            loss_label[j] += dice_per_class


    loss_label = np.true_divide(loss_label, result.size(0))
        
    return loss_label, total_loss/result.size(0) 

def dice_loss2(result, target, total_classes = 2):
    
    """
    Pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    
    """
    epsilon = 1e-6
    loss_label =  np.zeros(total_classes)
    total_loss = 0
    for j in range(0, total_classes):
            result_square_sum = torch.sum(result[:, j, :, :])
            target_square_sum = torch.sum((target[:, j, :, :]).float())
            intersect = torch.sum(result[:, j, :, :] * (target[:, j, :, :]).float())
            dice = (2 * intersect + epsilon) / (result_square_sum + target_square_sum + intersect + epsilon)
            dice_loss_per_class = 1 - dice
            total_loss += dice_loss_per_class
            
            loss_label[j] = dice_loss_per_class
            
            
    total_loss /= total_classes
        
    return loss_label, total_loss


def main(args):
    print("save path = ", args.save_dir)
    
    torch.manual_seed(1234)
    setgpu(args.gpu)
    highest_loss = 2.0
    best_label_acc_fold = np.zeros(2)
    output_file = os.path.join(args.save_dir, 'output.log')

    train_list = ['Patient_01', 'Patient_02', 'Patient_03', 'Patient_04', 'Patient_05', 'Patient_06', 'Patient_07', 'Patient_08', 'Patient_09', 'Patient_10', 'Patient_11', 'Patient_12', 'Patient_13', 'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17', 'Patient_18', 'Patient_19', 'Patient_20', 'Patient_21', 'Patient_27', 'Patient_28', 'Patient_29', 'Patient_30', 'Patient_31', 'Patient_32', 'Patient_33', 'Patient_34', 'Patient_35', 'Patient_36', 'Patient_37', 'Patient_38', 'Patient_39', 'Patient_40']
    test_list = ['Patient_22', 'Patient_23', 'Patient_24', 'Patient_25', 'Patient_26']

#    train_list = ['Patient_39', 'Patient_40']
#    test_list = ['Patient_26']

    # Loading Test Data
    SegThorTrainTrans = transforms.Compose([Rescale(0.5), Normalize(), ElasticTransform(), JointTransform2D(crop=(128, 128), p_flip=0.5), ToTensor()])
    train_set = SegThorDataset("../../../data/train", phase='train', transform=SegThorTrainTrans, file_list=train_list)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    # Loading validation data
    SegThorValTrans = transforms.Compose([Rescale(0.5), Normalize(), ToTensor()])
    val_set = SegThorDataset("../../../data/train", phase='val', transform=SegThorValTrans, file_list=test_list)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

    ## Checking result after each epoch
    SegThorTestTrans = transforms.Compose([Rescale(0.5, labeled=False), Normalize(labeled=False), ToTensor(labeled=False)])
    test_set = SegThorDataset("../../../data/test", patient = 'Patient_44', phase='test', transform=SegThorTestTrans)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, closs = get_model(model_name = 'unet_mtl', loss_name = 'CombinedLoss', alpha=args.alpha, if_auxloss=args.if_auxloss)

    net = net.to(device)
    closs = closs.to(device)
    if len(args.gpu.split(',')) > 1:
        net = DataParallel(net)
    net.apply(weight_init)

    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

    for epoch in range(args.epochs):
        f = open(output_file, 'a')
        f.write('Epoch {}/{}\n'.format(epoch + 1, args.epochs))
        f.write('-' * 10)
        lr = get_lr(epoch, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss, train_loss_label = train(train_loader, net, optimizer, epoch, device, args.batch_size, closs, args.if_auxloss, f)
        val_loss, val_loss_label = validation(val_loader, net, epoch, device, args.batch_size, val_set, closs, args.if_auxloss, f)

        writer.add_scalar('Train/train_loss', train_loss, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)
        f.close()

        if val_loss < highest_loss:
            highest_loss = val_loss

            for j in range(2):
                best_label_acc_fold[j] = val_loss_label[j]

            model_file = os.path.join(args.save_dir, '../model/model.pt')
            torch.save(net, model_file)
            state_dict = os.path.join(args.save_dir, '../state_dict')
            for i in glob.glob(os.path.join(state_dict,'*')):
                os.remove(i)

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                       },os.path.join(args.save_dir, '../state_dict', '%d.ckpt' % epoch))


        ## Visualize test results after every 8th epoch
        if epoch%8 == 0 and os.path.isfile(model_file):
            evaluate_model(net, test_loader, test_set, epoch, device, model_file)


    f = open(output_file, 'a')
    f.write("BEST RESULT \n")
    f.write("Eusophagus(Dice Score) = {:.4f} \n".format(best_label_acc_fold[1]))
    f.close()
    writer.close()


def train(train_loader, model, optimizer, epoch, device, batch_size, closs, auxloss, f):
    model.train()
    running_loss = 0.0
    running_dl = 0.0
    running_mse = 0.0
    running_loss_label = np.zeros(2)
    for batch_idx, sample in enumerate(train_loader):
        train_data, labels = sample['image'].to(device, dtype=torch.float), sample['label'].to(device, dtype=torch.uint8)

        optimizer.zero_grad()
        output = model(train_data)

        if auxloss==1:
            loss, loss_label, loss_dice, dt_loss = closs(output, labels)
        elif auxloss==0:
            loss_label, loss_dice = closs(output, labels)
            loss = loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        for i in range(2):
            running_loss_label[i] += loss_label[i]

    epoch_loss = running_loss / len(train_loader)
    epoch_loss_class = np.true_divide(running_loss_label, len(train_loader))
    f.write("\n Train Loss: Background = {:.4f} Eusophagus = {:.4f}\n".format(epoch_loss_class[0], epoch_loss_class[1]))

    return epoch_loss, epoch_loss_class

def validation(val_loader, model, epoch, device, batch_size, test_list, closs, auxloss, f):
    running_loss = 0.0
    running_loss_label = np.zeros(5)
    with torch.no_grad():
        model.eval()
        for batch_idx, sample in enumerate(val_loader):
            val_data, labels = sample['image'].to(device, dtype=torch.float32), sample['label'].to(device, dtype=torch.uint8)
            output = model(val_data)

            if auxloss==1:
                loss, loss_label, loss_dice, dt_loss = closs(output, labels)
            elif auxloss==0:
                loss_label, loss_dice = closs(output, labels)

            running_loss += loss_dice.item()
            for i in range(2):
                running_loss_label[i] += loss_label[i]

    epoch_loss = running_loss / len(val_loader)
    epoch_loss_class = np.true_divide(running_loss_label, len(val_loader))
    f.write("Validation Loss: Background = {:.4f} Eusophagus = {:.4f}\n \n".format(epoch_loss_class[0], epoch_loss_class[1]))

    return epoch_loss, epoch_loss_class


def evaluate_model(model, val_loader, val_set, epoch, device, model_file):
    print("Model file", model_file)
    model = torch.load(model_file)
    model.eval()

    count = 0
    seg_vol = zeros([len(val_set),  512, 512])
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            val_data = sample['image'].to(device, dtype=torch.float)

            output = model(val_data)

            max_idx = torch.argmax(output, 1, keepdim=True)
            max_idx = tensor_to_numpy(max_idx)

            slice_v = max_idx[:,:]
            slice_v = slice_v.astype(float32)
            slice_v = ndimage.interpolation.zoom(slice_v, zoom=2, order=0, mode='nearest', prefilter=True)
            seg_vol[count,:,:] = slice_v
            count = count + 1

        os.makedirs("validation_result", exist_ok=True)
        filename = os.path.join('validation_result', 'Patient_44_'+str(epoch)+'.nii')
        segmentation = sitk.GetImageFromArray(seg_vol, isVector=False)
        print("Saving segmented volume of size: ",segmentation.GetSize())
        sitk.WriteImage(sitk.Cast( segmentation, sitk.sitkUInt8 ), filename, True)


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, 'model')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, '../state_dict')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, '../log_dir')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)
#    main(epochs=40, batch_size=32, learning_rate=0.001)
