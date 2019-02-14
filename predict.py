###https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/discussion/29829
### batch processing for predicting results of large images

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm
import argparse
from PIL import Image

import unet
import numpy as np

import torch

import os
import augmentation as aug
import pickle

import ternausnet
import linknet
import albunet_v2
import albunet18
import albunet50
import TernausDense

model_choices = ['unet_small', 'tnaus', 'tnaus_resnet', 'link34', 'tnaus_resnetv2', 
                 'tnaus_resnet18', 'tnaus_vgg16', 'link50', 'tnaus50', 'tnaus_vgg16_elu', 
                 'tnaus_resnetElu', 'tnaus_dense121', 'tnaus_dense169']

parser = argparse.ArgumentParser(description='Road Extraction based on unet')

parser.add_argument('--data', metavar='DATA_DIR', 
                        help='path to the test dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to the checkpoint file for prediction (default: none)')
parser.add_argument('--save-dir', default='./test_predict', type=str, metavar='PATH',
                        help='path to the predicted results (default: ./test_predict)')
parser.add_argument('--patch-sz', default=112, type=int, metavar='SIZE',
                        help='patch size for image cropped from orig image (default: 112)')
parser.add_argument('--crop-sz', default=80, type=int, metavar='SIZE',
                        help='cropped size from the patch of prediction (default: 80)')
# parser.add_argument('--thres', default=0.5, type=float, metavar='M',
#                         help='threshold for roads (default: 0.5)')
parser.add_argument('--CUDA', default=True, type=str, metavar='M',
                        help='whether use CUDA for prediction (default: True)')
parser.add_argument('--batch-sz', default=10, type=int, metavar='SIZE',
                        help='batch size for prediction process (default: 10)')
parser.add_argument('--model', default='unet_small', type=str, metavar='M',
                        choices=model_choices,
                        help='choose model for training, choices are: ' \
                         + ' | '.join(model_choices) + ' (default: unet_small)')
parser.add_argument('--GPU', default=0, type=int, metavar='N',
                        help='which GPU is used for training (0 or 1)')

args = parser.parse_args()

#### set which GPU is used for predicting

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)


# clear cached memory
# http://forums.fast.ai/t/gpu-memory-not-being-freed-after-training-is-over/10265/3
# torch.cuda.empty_cache()

# https://github.com/pytorch/pytorch/issues/1085
# if there is some memory left on the GPU try to use : (pkill -9 python)

def main():
    global args

    patch_sz = args.patch_sz
    crop_sz = args.crop_sz
    batch_sz = args.batch_sz

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
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
    elif args.model == 'tnaus_dense169':
        model = TernausDense.TernausDense169(pretrained=True, is_deconv=True)
    # model = unet.UNetSmall()

    if args.CUDA:
        model = model.cuda()

    # if torch.cuda.is_available() and args.CUDA:
    #     model = model.cuda()

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

    check_base_name = os.path.basename(args.checkpoint)

    save_subdir = os.path.join(args.save_dir, check_base_name.split('.')[0] + '_' + str(patch_sz) + '_' + str(crop_sz))

    if not os.path.isdir(save_subdir):
        os.mkdir(save_subdir)


    model.eval()

    test_img_names = list(filter(lambda x: x.endswith('_sat.jpg'), os.listdir(args.data)))

    # stride = int(patch_sz / 2)
    stride = int(crop_sz / 2)

    stride_idx = list(range(0, 1024, stride))
    # max_stride_patch_idx = list(range(0, 1024, args.patch_sz))[-1]

    # data_transform = transforms.Compose([aug.ToTensor()])

    predict_results = {}

    # add reflection boundary area
    miro_margin = int((patch_sz-crop_sz)/2)

    batch_num = len(test_img_names) // batch_sz + 1


    # for test_img_nm in tqdm(test_img_names):
    for batch_idx in tqdm(range(0, batch_num)):

        batch_img_name_list = test_img_names[batch_idx*batch_sz:(batch_idx+1)*batch_sz]

        batch_img_path_list = [os.path.join(args.data, name) for name in batch_img_name_list]

        batch_img_list = [np.array(Image.open(path)) for path in batch_img_path_list]
        batch_img_array = np.array(batch_img_list)

        batch_predict_test_maps = np.zeros((len(batch_img_name_list), 1024, 1024))
        # batch_predict_test_masks = np.zeros((len(batch_img_name_list), 1024, 1024))

        # test_img_path = os.path.join(args.data, test_img_nm)
        # test_img = np.array(Image.open(test_img_path))
        # predict_test_map = np.zeros((1024, 1024))
        predict_test_mask = np.zeros((1024, 1024))

        # assert test_img.shape == (1024, 1024, 3)
        # test_img_miro = np.pad(test_img, 
        #     pad_width=[(miro_margin,miro_margin), 
        #                (miro_margin,miro_margin), 
        #                (0,0)], mode='reflect')
        
        test_img_miro_array = np.pad(batch_img_array, 
                                    pad_width=[(0,0),
                                                (miro_margin,miro_margin), 
                                                (miro_margin,miro_margin), 
                                                (0,0)], mode='reflect')

        assert test_img_miro_array.shape[1:] == (1024 + (patch_sz-crop_sz), 
                                                1024 + (patch_sz-crop_sz), 
                                                3)

        for i, strt_row in enumerate(stride_idx):
            for j, strt_col in enumerate(stride_idx):
                
                # refresh temp and temp mask
                # temp_test_map = np.zeros((1024, 1024))
                batch_temp_test_maps = np.zeros((len(batch_img_name_list), 1024, 1024))
                # batch_temp_test_masks = np.zeros((len(batch_img_name_list), 1024, 1024))
                temp_test_mask = np.zeros((1024, 1024))

                # if strt_row + patch_sz > 1024:
                #     strt_row = 1024 - patch_sz
                # if strt_col + patch_sz > 1024:
                #     strt_col = 1024 - patch_sz

                if strt_row + crop_sz > 1024:
                    strt_row = 1024 - crop_sz
                if strt_col + crop_sz > 1024:
                    strt_col = 1024 - crop_sz
                # transform original coordinate into mirror one
                strt_row_miro = strt_row + miro_margin
                strt_col_miro = strt_col + miro_margin

                # crop_test_img = test_img[strt_row:strt_row+patch_sz, strt_col:strt_col+patch_sz, :]
                # crop_test_img = test_img_miro[strt_row_miro-miro_margin:strt_row_miro-miro_margin+patch_sz, 
                #                               strt_col_miro-miro_margin:strt_col_miro-miro_margin+patch_sz,:]
                
                batch_crop_test_imgs = test_img_miro_array[:, strt_row_miro-miro_margin:strt_row_miro-miro_margin+patch_sz, 
                                                            strt_col_miro-miro_margin:strt_col_miro-miro_margin+patch_sz,:]
                
                batch_crop_test_imgs = torch.Tensor(np.transpose(batch_crop_test_imgs, axes=(0, 3, 1, 2)) / 255.0)
                # crop_test_img = (data_transform(crop_test_img)).unsqueeze(0)

                # if torch.cuda.is_available() and args.CUDA:
                #     input_crop_img = Variable(crop_test_img.cuda(), volatile=True)
                # else:
                # if args.CUDA:
                #     input_crop_img = Variable(crop_test_img.cuda(), volatile=True)
                # else:
                #     input_crop_img = Variable(crop_test_img, volatile=True)
                
                if args.CUDA:
                    batch_input_crop_imgs = Variable(batch_crop_test_imgs.cuda(), volatile=True)
                else:
                    batch_input_crop_imgs = Variable(batch_crop_test_imgs, volatile=True)
                
                output_logits = model(batch_input_crop_imgs)

                output_logits = torch.nn.functional.sigmoid(output_logits)

                output_maps = np.squeeze(output_logits.data.cpu().numpy())

                # print(output_maps.shape)
                #
                output_maps_crps = output_maps[:, miro_margin:miro_margin+crop_sz,
                                                  miro_margin:miro_margin+crop_sz]

                # create temp map for the associated patch and mask indicator
                # temp_test_map[strt_row:strt_row+args.patch_sz, strt_col:strt_col+args.patch_sz] = output_map
                # temp_test_mask[strt_row:strt_row+args.patch_sz, strt_col:strt_col+args.patch_sz] = np.ones((args.patch_sz, args.patch_sz))
                
                batch_temp_test_maps[:,strt_row:strt_row+crop_sz, strt_col:strt_col+crop_sz] = output_maps_crps
                # batch_temp_test_masks[:,strt_row:strt_row+crop_sz, strt_col:strt_col+crop_sz] = np.ones((len(batch_img_name_list), crop_sz, crop_sz))
                temp_test_mask[strt_row:strt_row+crop_sz, strt_col:strt_col+crop_sz] = np.ones((crop_sz, crop_sz))

                # calculate predicted map
                # predict_test_map = predict_test_map + temp_test_map
                batch_predict_test_maps = batch_predict_test_maps + batch_temp_test_maps
                
                # check whether there is overlap area
                # overlap_mask = temp_test_mask * predict_test_mask
                
                # if there is, calucate its mean
                # predict_test_map[np.nonzero(overlap_mask)] = predict_test_map[np.nonzero(overlap_mask)] / 2
                
                # update the mask indicator
                predict_test_mask = predict_test_mask + temp_test_mask
                

                # print('predicting {} img with {}th patch'.format(test_img_nm, i*len(stride_idx) + j))
        predict_test_mask = np.expand_dims(predict_test_mask, axis=0)

        # predict_test_map = predict_test_map / predict_test_mask
        batch_predict_test_maps = batch_predict_test_maps / predict_test_mask

        for img_idx, test_img_nm in enumerate(batch_img_name_list):

            save_npy_path = os.path.join(save_subdir, test_img_nm.split('_')[0] + '.npy')
            # print('save img', test_img_nm)
            np.save(save_npy_path, batch_predict_test_maps[img_idx, :])

        # save_npy_path = os.path.join(save_subdir, test_img_nm.split('_')[0] + '.npy')
        
        # np.save(save_npy_path, predict_test_map)

        # predict_results[test_img_nm] = predict_test_map

    # with open(os.path.join(save_subdir, check_base_name.split('.')[0]+'.p'), 'wb') as f:
    #     pickle.dump(predict_results, f)
    # save_npy_path = os.path.join(save_subdir, check_base_name.split('.')[0]+'.npy')
    # np.save(save_npy_path, predict_results)

    # print('saved numpy file of predicted results', save_npy_path)

        # save_test_map = np.zeros((1024, 1024))
        # save_test_map[predict_test_map >= args.thres] = 255

        # save_test_map = np.expand_dims(save_test_map, axis=2)
        # save_test_map = np.repeat(save_test_map, 3, axis=2)

        # sv_predict_mask_nm = test_img_nm.split('_')[0] + '_mask.png'


        # sv_predict_mask_path = os.path.join(save_subdir, sv_predict_mask_nm)

        # Image.fromarray(save_test_map.astype(np.uint8)).save(sv_predict_mask_path)

        # print('saved predicted image {}'.format(test_img_nm))

if __name__ == '__main__':
    main()