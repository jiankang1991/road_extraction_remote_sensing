
# from torchvision.datasets.folder import *
import scipy.misc
from PIL import Image
import os
import numpy as np

import matplotlib
if ("SSH_CONNECTION" in os.environ) or ('SSH_TTY' in os.environ):
    # dont display plot if on remote server
    matplotlib.use('agg')

import matplotlib.pyplot as plt
plt.switch_backend('agg')

class DeepGlobeDataset(object):
    '''deep globe dataset '''

    def __init__(self, root_dir, status = 'train', transform=None):

        self.status = status
        self.root_dir = root_dir
        self.transform = transform
        self.sat_img_names = list(filter(lambda x: '_sat_' in x, os.listdir(os.path.join(self.root_dir, self.status))))
        # self.loader = loader

    def __getitem__(self, index):
        sat_img_nm = self.sat_img_names[index]
        mask_img_nm = self.sat_img_names[index].split('_')[0] + '_mask_' + self.sat_img_names[index].split('_')[2].split('.')[0] + '.png'

        sat_img_path = os.path.join(self.root_dir, self.status, sat_img_nm)
        mask_img_path = os.path.join(self.root_dir, self.status, mask_img_nm)

        sat_img = scipy.misc.imread(sat_img_path)
        # sat_img = Image.open(sat_img_path)
        mask_img = scipy.misc.imread(mask_img_path)

        mask = np.zeros((mask_img.shape[0], mask_img.shape[1]))
        # since it is not exactly 255 for road area, binarize at 128
        # mask[np.where(np.all(mask_img==(255,255,255), axis=-1))] = 1
        mean_mask = np.mean(mask_img, axis=-1)
        mask[mean_mask >= 128] = 1
        
        mask = mask.astype(np.int32)

        sample = {'sat_img': sat_img, 'map_img': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.sat_img_names)





def show_map(sat_img, map_img=None, axis=None):
    """
    Return an image with the shape mask if there is one supplied
    """

    if axis:
        axis.imshow(sat_img)

        if map_img is not None:
            axis.imshow(map_img, alpha=0.5, cmap='gray')

    else:
        plt.imshow(sat_img)

        if map_img is not None:
            plt.imshow(map_img, alpha=0.5, cmap='gray')


# helper function to show a batch
def show_map_batch(sample_batched, img_to_show=3, save_file_path=None, as_numpy=False):
    """
    Show image with map image overlayed for a batch of samples.
    """

    # just select 6 images to show per batch
    sat_img_batch, map_img_batch = sample_batched['sat_img'][:img_to_show, :, :, :],\
                                   sample_batched['map_img'][:img_to_show, :, :, :]
    batch_size = len(sat_img_batch)

    f, ax = plt.subplots(int(np.ceil(batch_size / 3)), 3, figsize=(15, int(np.ceil(batch_size / 3)) * 5))
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    ax = ax.ravel()

    # unorm = UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    for i in range(batch_size):
        ax[i].axis('off')
        show_map(sat_img=sat_img_batch.cpu().numpy()[i, :, :, :].transpose((1, 2, 0)),
                 map_img=map_img_batch.cpu().numpy()[i, 0, :, :], axis=ax[i])

    if save_file_path is not None:
        f.savefig(save_file_path)

    if as_numpy:
        f.canvas.draw()
        width, height = f.get_size_inches() * f.get_dpi()
        mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.cla()
        plt.close(f)

        return mplimage


def show_tensorboard_image(sat_img, map_img, out_img, save_file_path=None, as_numpy=False):
    """
    Show 3 images side by side for verification on tensorboard. Takes in torch tensors.
    """
    # show different image from the batch
    batch_size = sat_img.size(0)
    img_num = np.random.randint(batch_size)

    f, ax = plt.subplots(1, 3, figsize=(12, 5))
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    ax = ax.ravel()

    ax[0].imshow(sat_img[img_num,:,:,:].cpu().numpy().transpose((1,2,0)))
    ax[0].axis('off')
    ax[1].imshow(map_img[img_num,0,:,:].cpu().numpy())
    ax[1].axis('off')
    ax[2].imshow(out_img[img_num,0,:,:].data.cpu().numpy())
    ax[2].axis('off')

    if save_file_path is not None:
        f.savefig(save_file_path)

    if as_numpy:
        f.canvas.draw()
        width, height = f.get_size_inches() * f.get_dpi()
        mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.cla()
        plt.close(f)

        return mplimage




