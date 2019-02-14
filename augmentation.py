
import numpy as np
from torchvision import transforms
import random
import torch
import cv2

from PIL import ImageEnhance
from PIL import Image

class RandomCropTarget(object):
    """
    Crop the image and target randomly in a sample.

    Args:
    output_size (tuple or int): Desired output size. If int, square crop
        is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sat_img, map_img = sample['sat_img'], sample['map_img']

        h, w = sat_img.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sat_img = sat_img[top: top + new_h, left: left + new_w]
        map_img = map_img[top: top + new_h, left: left + new_w]

        return {'sat_img': sat_img, 'map_img': map_img}

class CenterCropTarget(object):
    """
    Crop the image and target in the center in a sample.

    Args:
    output_size (tuple or int): Desired output size. If int, square crop
        is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        sat_img, map_img = sample['sat_img'], sample['map_img']

        h, w = sat_img.shape[:2]
        new_h, new_w = self.output_size

        i = int(round((h - new_h) / 2.))
        j = int(round((w - new_w) / 2.))


        sat_img = sat_img[i: i + new_h, j: j + new_w]
        map_img = map_img[i: i + new_h, j: j + new_w]

        return {'sat_img': sat_img, 'map_img': map_img}

class RandomRotate(object):

    def __call__(self, sample):
        
        rand = random.random()

        if rand < 0.25:
            sat_img = np.rot90(sample['sat_img'], k=1)
            map_img = np.rot90(sample['map_img'], k=1)

        elif 0.25 <= rand and rand < 0.5:
            sat_img = np.rot90(sample['sat_img'], k=2)
            map_img = np.rot90(sample['map_img'], k=2)

        elif 0.5 <= rand and rand < 0.75:
            sat_img = np.rot90(sample['sat_img'], k=3)
            map_img = np.rot90(sample['map_img'], k=3)

        elif 0.75 <= rand and rand < 1:
            sat_img = sample['sat_img']
            map_img = sample['map_img']
            
        return {'sat_img': sat_img.copy(), 'map_img': map_img.copy()}

class RandomFlip(object):

    def __call__(self, sample):
        
        rand = random.random()

        if rand < 1 / 3.0:
            sat_img = np.fliplr(sample['sat_img'])
            map_img = np.fliplr(sample['map_img'])

        elif 1 / 3.0 <= rand and rand < 2 / 3.0:
            sat_img = np.flipud(sample['sat_img'])
            map_img = np.flipud(sample['map_img'])

        elif 2 / 3.0 <= rand and rand < 1:
            sat_img = sample['sat_img']
            map_img = sample['map_img']
            
        return {'sat_img': sat_img.copy(), 'map_img': map_img.copy()}

class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        # print(type(sat_img))
        return {'sat_img': transforms.functional.to_tensor(sat_img),
                'map_img': torch.from_numpy(map_img).float().unsqueeze(0)} # unsqueeze for the channel dimension

class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        for t, m, s in zip(sat_img, self.mean, self.std):
            t.sub_(m).div_(s)

        return {'sat_img': sat_img,
                'map_img': map_img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sat_img):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        # print(type(sat_img))
        return transforms.functional.to_tensor(sat_img)

class RandomHueSaturationValue(object):
    def __init__(self, hue_shift_limit=(-10, 10), sat_shift_limit=(-25, 25), val_shift_limit=(-25, 25), prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, sample):

        sat_img, map_img = sample['sat_img'], sample['map_img']

        if random.random() < self.prob:

            image = cv2.cvtColor(sat_img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            sat_img = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        return {'sat_img': sat_img.copy(),
                'map_img': map_img.copy()} 



class RandomBrightnessEnhance(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        if random.random() < self.prob:
            sat_img = Image.fromarray(sat_img)
            sat_img = np.array(ImageEnhance.Brightness(sat_img).enhance(random.uniform(0.8,1.2)))
        
        return {'sat_img': sat_img.copy(),
                'map_img': map_img.copy()}

class RandomColorEnhance(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        if random.random() < self.prob:
            sat_img = Image.fromarray(sat_img)
            sat_img = np.array(ImageEnhance.Color(sat_img).enhance(random.uniform(0.5,1.5)))
        
        return {'sat_img': sat_img.copy(),
                'map_img': map_img.copy()}


class RandomContrastEnhance(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        if random.random() < self.prob:
            sat_img = Image.fromarray(sat_img)
            sat_img = np.array(ImageEnhance.Contrast(sat_img).enhance(random.uniform(0.5,1.5)))
        
        return {'sat_img': sat_img.copy(),
                'map_img': map_img.copy()}


class RandomSharpness(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        sat_img, map_img = sample['sat_img'], sample['map_img']

        if random.random() < self.prob:
            sat_img = Image.fromarray(sat_img)
            sat_img = np.array(ImageEnhance.Sharpness(sat_img).enhance(random.random()))
        
        return {'sat_img': sat_img.copy(),
                'map_img': map_img.copy()}


