try:
    from scipy import misc
except Exception:  # pragma: no cover
    misc = None
import os
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    if isinstance(size, int):
        size = (size, size)
    cam_img = cv2.resize(cam_img, (size[1], size[0]))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


class VGG19FeatureExtractor(nn.Module):
    """VGG19 network returning intermediate feature maps."""

    def __init__(self, layers=(17, 26)):
        super().__init__()
        try:
            vgg = models.vgg19(pretrained=True)
        except Exception:
            vgg = models.vgg19(pretrained=False)
        self.features = vgg.features
        self.layers = set(layers)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = imagenet_norm(x)
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                feats.append(x)
        return feats


def contextual_loss(x, y, h=0.5, eps=1e-5):
    """Computes Contextual Loss between feature maps x and y."""
    N, C, H, W = x.size()
    x = x.view(N, C, -1)
    y = y.view(N, C, -1)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    x = x.permute(0, 2, 1)
    y = y.permute(0, 2, 1)
    x_norm = (x ** 2).sum(dim=2, keepdim=True)
    y_norm = (y ** 2).sum(dim=2, keepdim=True)
    dist = x_norm + y_norm.transpose(1, 2) - 2 * torch.bmm(x, y.transpose(1, 2))
    dist = torch.sqrt(F.relu(dist) + eps)
    dist_min, _ = dist.min(dim=2, keepdim=True)
    relative_dist = dist / (dist_min + eps)
    w = torch.exp((1 - relative_dist) / h)
    cx = w / (w.sum(dim=2, keepdim=True) + eps)
    cx_i, _ = cx.max(dim=2)
    loss = torch.mean(-torch.log(cx_i + eps))
    return loss


def total_variation_loss(x):
    """Total variation loss for spatial smoothness."""
    loss = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss += torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return loss


class ResizeCenterCrop:
    """Resize keeping aspect ratio and center crop to target size."""

    def __init__(self, size, interpolation=Image.BICUBIC):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (h, w)
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        target_h, target_w = self.size
        w, h = img.size
        scale = max(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), self.interpolation)

        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))
        return img
