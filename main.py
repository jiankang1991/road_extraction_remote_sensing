
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from copy import deepcopy

import logger
import data_utils
import augmentation as aug
import metrics
import unet
import ternausnet
import linknet
import albunet_v2
import albunet18
import albunet50
import TernausDense
import TernausXt

import torch
import torch.optim as optim
import time
import argparse
import shutil
import os

from datetime import datetime


model_choices = ['unet_small', 'tnaus', 'tnaus_resnet', 'link34', 'tnaus_resnetv2', 
                 'tnaus_resnet18', 'tnaus_vgg16', 'link50', 'tnaus50', 'tnaus_vgg16_elu', 
                 'tnaus_resnetElu', 'tnaus_dense121', 'tnaus_dense169', 'tnaus_dense121_up',
                 'tnaus_xt']
                 
parser = argparse.ArgumentParser(description='Road Extraction based on unet')

parser.add_argument('--data', metavar='DATA_DIR', 
                        help='path to dataset (parent dir of train and val)')
parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run (default: 75)')
parser.add_argument('--model', default='unet_small', type=str, metavar='M',
                    choices=model_choices,
                        help='choose model for training, choices are: ' \
                         + ' | '.join(model_choices) + ' (default: unet_small)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='epoch to start from (used with resume flag')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-3)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
parser.add_argument('--print-freq', default=30, type=int, metavar='N',
                        help='number of time to log per epoch (default: 30)')
parser.add_argument('--run', default=0, type=int, metavar='N',
                        help='number of run (for tensorboard logging) (default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
parser.add_argument('--crop-sz', default=112, type=int, metavar='SIZE',
                        help='number of cropped pixels from orig image (default: 112)')
# parser.add_argument('--loss-func', default='BCEDice', type=str, metavar='PATH',
#                         help='loss function to be used (BCEDice or Jaccard) (default: BCEDice)')
parser.add_argument('--jt-loss-weight', default=1.0, type=float, metavar='M',
                        help='weight of Dice or Jaccard term of the joint loss (default: 1.0)')
parser.add_argument('--acc-best', dest='acc_best', action='store_true',
                        help='whether store the best model according to validation loss or accuracy (default: Acc)')
parser.add_argument('--lovasz-loss', dest='lovasz_loss', action='store_true',
                        help='whether lovasz loss to be used (BCE_lovasz or BCE_Dice)')
parser.add_argument('--GPU', default=0, type=int, metavar='N',
                        help='which GPU is used for training (0 or 1)')
parser.add_argument('--hard-mining', dest='hard_mining', action='store_true',
                        help='whether use hard negative mining (default: True)')
parser.add_argument('--cycle-start-epoch', default=30, type=int,
                        help='start epoch for using cyclic-lr (default: 30)')

args = parser.parse_args()

#### set which GPU is used for training

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)


def save_checkpoint(state, is_best, name):
    """
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    checkpoint_dir = './checkpoints'
    
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))


def main():
    global args

    since = time.time()
    
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    print('saving file name is ', sv_name)

    if args.model == 'unet_small':
        # get model
        model = unet.UNetSmall()
    elif args.model == 'tnaus':
        model = ternausnet.unet11(pretrained='carvana', model_path='./pre_trained_models/TernausNet.pt')
    elif args.model == 'tnaus_resnet':
        model = ternausnet.AlbuNet(pretrained=True,is_deconv=True)
    elif args.model == 'link34':
        model = linknet.LinkNet34(num_classes=1)
    elif args.model == 'link50':
        model = linknet.LinkNet50(num_classes=1)
    elif args.model == 'tnaus_resnetv2':
        model = albunet_v2.AlbuNet(pretrained=False,is_deconv=True)
    elif args.model == 'tnaus_resnet18':
        model = albunet18.AlbuNet(pretrained=True,is_deconv=True)
    elif args.model == 'tnaus_vgg16':
        model = ternausnet.UNet16(pretrained=True,is_deconv=True)
    elif args.model == 'tnaus50':
        model = albunet50.AlbuNet50(pretrained=True)
    elif args.model == 'tnaus_vgg16_elu':
        model = ternausnet.UNet16_elu(pretrained=True,is_deconv=True)
    elif args.model == 'tnaus_resnetElu':
        model = ternausnet.AlbuNetElu(pretrained=True,is_deconv=True)
    elif args.model == 'tnaus_dense121':
        model = TernausDense.TernausDense121(pretrained=True, is_deconv=True)
    elif args.model == 'tnaus_dense121_up':
        model = TernausDense.TernausDense121(pretrained=True, is_deconv=False)
    elif args.model == 'tnaus_dense169':
        model = TernausDense.TernausDense169(pretrained=True, is_deconv=True)
    elif args.model == 'tnaus_xt':
        model = TernausXt.TernausXt(num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()

    # set up binary cross entropy and dice loss
    #### seems that Dice shows better results
    # if args.loss_func == 'BCEDice':
    #     criterion = metrics.BCEDiceLoss(penalty_weight=args.jt_loss_weight)
    # else:
    #     criterion = metrics.LossBinaryJaccard(jaccard_weight=args.jt_loss_weight)

    if args.lovasz_loss:
        print('using lovasz loss function')
        criterion = metrics.BCELovaszLoss()
    else:
        criterion = metrics.BCEDiceLoss(penalty_weight=args.jt_loss_weight)

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # decay LR

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    # starting params
    best_loss = 999
    best_acc = 0

    start_epoch = args.start_epoch
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)
            
            if checkpoint['epoch'] > args.start_epoch:
                start_epoch = checkpoint['epoch']

            best_loss = checkpoint['best_loss']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    # # normalize according to ImageNet
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # get data
    dataset_train = data_utils.DeepGlobeDataset(args.data, 'train', 
                                                transform=transforms.Compose([aug.RandomCropTarget(output_size=args.crop_sz),
                                                                              aug.RandomFlip(),
                                                                              aug.RandomRotate(),
                                                                            #   aug.RandomHueSaturationValue(),
                                                                              aug.ToTensorTarget()]))
    
    # dataset_train = data_utils.DeepGlobeDataset(args.data, 'train', 
    #                                             transform=transforms.Compose([aug.RandomCropTarget(output_size=args.crop_sz),
    #                                                                           aug.RandomFlip(),
    #                                                                           aug.RandomRotate(),
    #                                                                           aug.RandomBrightnessEnhance(),
    #                                                                           aug.RandomColorEnhance(),
    #                                                                           aug.RandomContrastEnhance(),
    #                                                                           aug.ToTensorTarget()]))
    dataset_val = data_utils.DeepGlobeDataset(args.data, 'val',
                                                transform=transforms.Compose([aug.CenterCropTarget(output_size=args.crop_sz),
                                                                              aug.ToTensorTarget()]))
    
    # creating loaders
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True)


    # loggers
    train_logger = logger.Logger('./logs/run_{}/training'.format(str(args.run)), args.print_freq)
    val_logger = logger.Logger('./logs/run_{}/validation'.format(str(args.run)), args.print_freq)

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # step the learning rate scheduler
        # https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/train.py

        if epoch == args.cycle_start_epoch:
            print("Starting cyclic lr")
            print("initial lr: ", args.lr)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if epoch >= args.cycle_start_epoch:
            lr = cyclic_lr(optimizer, epoch - args.cycle_start_epoch, init_lr=args.lr, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.1)
            print("cycling lr: ", lr)
        else:
            lr_scheduler.step()

        # run training and validation
        train_metrics = train(train_dataloader, model, criterion, optimizer, lr_scheduler, train_logger, epoch)
        valid_metrics = validation(val_dataloader, model, criterion, val_logger, epoch)

        # store best loss according to loss or acc and save a model checkpoint
        
        is_best_loss = valid_metrics['valid_loss'] < best_loss
        is_best_acc = valid_metrics['valid_acc'] > best_acc

        best_loss = min(valid_metrics['valid_loss'], best_loss)
        best_acc = max(valid_metrics['valid_acc'], best_acc)

        if args.acc_best:
            # print('saving loss best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc' : best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best_loss, sv_name)
        else:
            # print('saving accuracy best model')
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'best_acc' : best_acc,
                'optimizer': optimizer.state_dict()
            }, is_best_acc, sv_name)

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def make_train_step(idx, data, model, optimizer, criterion, meters):

    # get the inputs and wrap in Variable
    if torch.cuda.is_available():
        inputs = Variable(data['sat_img'].cuda())
        labels = Variable(data['map_img'].cuda())
    else:
        inputs = Variable(data['sat_img'])
        labels = Variable(data['map_img'])

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    # prob_map = model(inputs) # last activation was a sigmoid
    # outputs = (prob_map > 0.3).float()
    outputs = model(inputs)

    # pay attention to the weighted loss should input logits not probs
    if args.lovasz_loss:
        loss, BCE_loss, DICE_loss = criterion(outputs, labels)
        outputs = torch.nn.functional.sigmoid(outputs)
    else:
        outputs = torch.nn.functional.sigmoid(outputs)
        loss, BCE_loss, DICE_loss = criterion(outputs, labels)

    # backward
    loss.backward()
    # https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/train.py
    # torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
    optimizer.step()


    meters["train_acc"].update(metrics.dice_coeff(outputs, labels), outputs.size(0))
    meters["train_loss"].update(loss.data[0], outputs.size(0))
    meters["train_IoU"].update(metrics.jaccard_index(outputs, labels), outputs.size(0))
    meters["train_BCE"].update(BCE_loss.data[0], outputs.size(0))
    meters["train_DICE"].update(DICE_loss.data[0], outputs.size(0))
    meters["outputs"] = outputs
    return meters

def train(train_loader, model, criterion, optimizer, scheduler, logger, epoch_num):

    
    # logging accuracy and loss
    train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()
    train_IoU = metrics.MetricTracker()
    train_BCE = metrics.MetricTracker()
    train_DICE = metrics.MetricTracker()

    meters = {"train_acc": train_acc, "train_loss": train_loss, 
              "train_IoU": train_IoU, "train_BCE": train_BCE, 
              "train_DICE": train_DICE, "outputs": None}

    log_iter = len(train_loader)//logger.print_freq

    model.train()
    
    scheduler.step()

    cache = None
    cached_loss = 0

    # iterate over data
    for idx, data in enumerate(tqdm(train_loader, desc="training")):

        meters = make_train_step(idx, data, model, optimizer, criterion, meters)

        # hard negative mining
        # https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/train.py
        if args.hard_mining:

            if cache is None or cached_loss < meters["train_loss"].val:
                cached_loss = meters["train_loss"].val
                cache = deepcopy(data)

            if idx % 50 == 0 and cache is not None:
                meters = make_train_step(idx, data, model, optimizer, criterion, meters)
                cache = None
                cached_loss = 0

        # tensorboard logging
        if idx % log_iter == 0:

            step = (epoch_num*logger.print_freq)+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': meters["train_loss"].avg,
                'accuracy': meters["train_acc"].avg,
                'IoU': meters["train_IoU"].avg
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)
            
            # # log weights, biases, and gradients
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(tag, value.data.cpu().numpy(), step)
            #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)
            
            # log the sample images
            log_img = [data_utils.show_tensorboard_image(data['sat_img'], data['map_img'], meters["outputs"], as_numpy=True),]
            logger.image_summary('train_images', log_img, step)

    print('Training Loss: {:.4f} BCE: {:.4f} DICE: {:.4f} Acc: {:.4f} IoU: {:.4f} '.format(
            meters["train_loss"].avg, meters["train_BCE"].avg, meters["train_DICE"].avg, meters["train_acc"].avg, meters["train_IoU"].avg))
    print()
    
    return {'train_loss': meters["train_loss"].avg, 'train_acc': meters["train_acc"].avg, 
            'train_IoU': meters["train_IoU"].avg, 'train_BCE': meters["train_BCE"].avg,
            'train_DICE': meters["train_DICE"].avg}


def validation(valid_loader, model, criterion, logger, epoch_num):
    """

    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:

    Returns:

    """
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()
    valid_IoU = metrics.MetricTracker()
    valid_BCE = metrics.MetricTracker()
    valid_DICE = metrics.MetricTracker()

    log_iter = len(valid_loader)//logger.print_freq

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc='validation')):

        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(data['sat_img'].cuda(), volatile=True)
            labels = Variable(data['map_img'].cuda(), volatile=True)
        else:
            inputs = Variable(data['sat_img'], volatile=True)
            labels = Variable(data['map_img'], volatile=True)

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)

        # pay attention to the weighted loss should input logits not probs
        if args.lovasz_loss:
            loss, BCE_loss, DICE_loss = criterion(outputs, labels)
            outputs = torch.nn.functional.sigmoid(outputs)
        else:
            outputs = torch.nn.functional.sigmoid(outputs)
            loss, BCE_loss, DICE_loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data[0], outputs.size(0))
        valid_IoU.update(metrics.jaccard_index(outputs, labels), outputs.size(0))
        valid_BCE.update(BCE_loss.data[0], outputs.size(0))
        valid_DICE.update(DICE_loss.data[0], outputs.size(0))


        # tensorboard logging
        if idx % log_iter == 0:

            step = (epoch_num*logger.print_freq)+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': valid_loss.avg,
                'accuracy': valid_acc.avg,
                'IoU': valid_IoU.avg
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            # log the sample images
            log_img = [data_utils.show_tensorboard_image(data['sat_img'], data['map_img'], outputs, as_numpy=True),]
            logger.image_summary('valid_images', log_img, step)

    print('Validation Loss: {:.4f} BCE: {:.4f} DICE: {:.4f} Acc: {:.4f} IoU: {:.4f}'.format(
        valid_loss.avg, valid_BCE.avg, valid_DICE.avg, valid_acc.avg, valid_IoU.avg))
    print()

    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg, 
            'valid_IoU': valid_IoU.avg, 'valid_BCE': valid_BCE.avg,
            'valid_DICE': valid_DICE.avg}


def cyclic_lr(optimizer, epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()