import os
import glob
import torch
import argparse
from pylab import *
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import SimpleITK as sitk
from numpy import ndarray
from scipy import ndimage
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from cross_validation import k_folds
from models.model_loader import get_model
from data_processing.data_loader import SegThorDataset
from utils import setgpu, get_lr, weight_init, tensor_to_numpy, eval_dice
from data_processing.transformations import Rescale, Normalize, ToTensor, ToTensor2, RandomFlip, RandomRotation, ElasticTransform

#####################################################################################
"""
		Main function 
		Python 3
		Pytorch 1.1.0
		Author: Neelu Madan
        Institute: University of Duisburg-Essen
"""
#####################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

parser.add_argument(
    '--with_improvement',
    default=0,
    type=int,
    metavar='1(True) or 0(False)',
    help='train early_fusion with improvement term')

def main(args):

    cudnn.benchmark = True
    highest_loss = 4.0
    setgpu(args.gpu)
    best_label_acc_fold = np.zeros(5)
    output_file = os.path.join(args.save_dir, 'output.log')
    train_path = os.path.join(args.data_path, 'train')

    train_list = ['Patient_01', 'Patient_02', 'Patient_03', 'Patient_04', 'Patient_05', 'Patient_06', 'Patient_07', 'Patient_08', 'Patient_09', 'Patient_10', 'Patient_11', 'Patient_12', 'Patient_13', 'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17', 'Patient_18', 'Patient_19', 'Patient_20', 'Patient_21', 'Patient_22', 'Patient_23', 'Patient_24', 'Patient_25', 'Patient_26', 'Patient_27', 'Patient_28', 'Patient_29', 'Patient_30', 'Patient_36', 'Patient_37', 'Patient_38', 'Patient_39', 'Patient_40']
    test_list = ['Patient_31', 'Patient_32', 'Patient_33', 'Patient_34', 'Patient_35']

    # Loading Test Data
    SegThorTrainTrans = transforms.Compose([Rescale(1.0, model=args.model_name), Normalize(), RandomFlip(), RandomRotation(), ToTensor2()])
    train_set = SegThorDataset(train_path, phase='train', transform=SegThorTrainTrans, file_list=train_list)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

    # Loading validation data
    SegThorValTrans = transforms.Compose([Rescale(1.0, model=args.model_name), Normalize(), ToTensor2()])
    val_set = SegThorDataset(train_path, phase='val', transform=SegThorValTrans, file_list=test_list)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    print("Loading: ", args.model_name)
    net, loss = get_model(model_name=args.model_name,  with_improvement=args.with_improvement)

    net = net.to(device)
    loss = loss.to(device)
    if len(args.gpu.split(',')) > 1:
        net = DataParallel(net)
    net.apply(weight_init)

    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0001)

    for epoch in range(args.epochs):
        f = open(output_file, 'a')
        f.write('Epoch {}/{} \n'.format(epoch + 1, args.epochs))
        f.write('-' * 10)

        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss, train_loss_label = train(train_loader, net, optimizer, epoch, device, args.batch_size, loss, f)
        val_loss, val_loss_label = validation(val_loader, net, epoch, device, args.batch_size, val_set, f)

        writer.add_scalar('Train/train_loss', train_loss, epoch)
        writer.add_scalar('Val/val_loss', val_loss, epoch)
        f.close()
        if val_loss < highest_loss:
            highest_loss = val_loss

            for j in range(5):
                best_label_acc_fold[j] = val_loss_label[j]

            model_file = os.path.join(args.save_dir, '../model/model.pt')
            torch.save(net, model_file)
            state_dict = os.path.join(args.save_dir, '../state_dict')
            for i in glob.glob(os.path.join(args.save_dir, 'state_dict','*')):
                os.remove(i)

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                       },os.path.join(args.save_dir, '../state_dict', '%d.ckpt' % epoch))

    f = open(output_file, 'a')
    f.write("RESULT \n")
    f.write(" Eusophagus = {:.4f}  Heart = {:.4f}  Trachea = {:.4f}  Aorta = {:.4f}\n".format(best_label_acc_fold[1], best_label_acc_fold[2], best_label_acc_fold[3], best_label_acc_fold[4]))
    f.close()
    writer.close()


def train(train_loader, model, optimizer, epoch, device, batch_size, diceLoss, f):
    model.train()
    running_loss = 0.0
    running_loss_label = np.zeros(5)
    for batch_idx, sample in enumerate(train_loader):
        train_data, labels, coarse_segmentation = sample['image'].to(device, dtype=torch.float32), sample['label'].to(device, dtype=torch.uint8), sample['coarse_segmentation'].to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(train_data, coarse_segmentation)

        loss_label, loss = diceLoss(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        for i in range(5):
            running_loss_label[i] += loss_label[i]

    epoch_loss = running_loss / len(train_loader)
    epoch_loss_class = np.true_divide(running_loss_label, len(train_loader))
    f.write("\n Train Loss:= Background = {:.4f} Eusophagus = {:.4f}  Heart = {:.4f}  Trachea = {:.4f}  Aorta = {:.4f}\n".format(epoch_loss_class[0], epoch_loss_class[1], epoch_loss_class[2], epoch_loss_class[3], epoch_loss_class[4]))

    return epoch_loss, epoch_loss_class

def validation(val_loader, model, epoch, device, batch_size, test_list, f):
    with torch.no_grad():
        model.eval()
        tbar = tqdm(val_loader, desc='\r')
        total_loss=[]
        running_loss_label = np.zeros(5)
        cur_cube=[]
        cur_label_cube=[]
        next_cube=[]
        counter=0
        end_flag=False
        for batch_idx, sample in enumerate(tbar):
            val_data, val_labels, val_cs, num_slices = sample['image'].to(device, dtype=torch.float), sample['label'].to(device, dtype=torch.uint8), sample['coarse_segmentation'].to(device, dtype=torch.float), sample['num_slice']
            output = model(val_data, val_cs)
            batch_size = output.size(0)
            slice_num = num_slices.numpy()[0]

            counter+=batch_size
            if counter<=slice_num:
                cur_cube.append(output)
                cur_label_cube.append(val_labels)
                if counter==slice_num:
                    end_flag=True
                    counter=0
            else:
                last=batch_size-(counter-slice_num)

                last_o=output[0:last]
                last_l=val_labels[0:last]

                first_o=output[last:]
                first_l=val_labels[last:]

                cur_cube.append(last_o)
                cur_label_cube.append(last_l)
                end_flag=True
                counter=counter-slice_num

            if end_flag:
                end_flag=False
                predict_cube=torch.stack(cur_cube,dim=0)
                label_cube=torch.stack(cur_label_cube,dim=0)
                predict_cube = torch.squeeze(predict_cube, dim=1)
                label_cube= torch.squeeze(label_cube, dim=1)
                cur_cube=[]
                cur_label_cube=[]
                if counter!=0:
                    cur_cube.append(first_o)
                    cur_label_cube.append(first_l)

                assert predict_cube.size()[0]==slice_num
                loss_label, loss = eval_dice(predict_cube, label_cube)

                total_loss.append(loss)
                for p in range(5):
                    running_loss_label[p] += loss_label[p]


        epoch_loss = sum(total_loss) / len(total_loss)
        epoch_loss = epoch_loss/5
        epoch_loss_class = np.true_divide(running_loss_label, len(total_loss))

        f.write("\nValidation Loss:= Background = {:.4f} Eusophagus = {:.4f}  Heart = {:.4f}  Trachea = {:.4f}  Aorta = {:.4f}\n \n".format(epoch_loss_class[0], epoch_loss_class[1], epoch_loss_class[2], epoch_loss_class[3], epoch_loss_class[4]))
        return epoch_loss, epoch_loss_class


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
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
