import os
import sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as utils
import copy

from Style_Transfer import *

def main():
    image_style = read_image("./images/starry_night_google.jpg")
    image_content = read_image("./images/louvre.jpg")
    show_image(image_style)
    show_image(image_content)

    base_model = models.vgg19(pretrained=True).features.to(device).eval()
    base_model_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    base_model_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    image_init = image_content.clone()
    image_init += torch.randn(image_content.data.size(), device=device)/10
    # print(image_init)
    # plt.figure()
    # show_image(image_init)
    output = proc_transfer(base_model, base_model_normalization_mean, base_model_normalization_std,image_content, image_style, image_init)
    # plt.figure()
    # show_image(output)
    save_image(output)

main()