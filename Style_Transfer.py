# coding: utf-8
# %%
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

if torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

# %%
# define the input and output method
# together with pre-processing
def read_image(filename):
    image = Image.open(filename)
    size=512
    to_tensor = transforms.Compose([transforms.Resize(size),transforms.ToTensor()])
    image = to_tensor(image)
    image = torch.unsqueeze(image, 0)
    image = transforms.functional.crop(image,0,0,size,size)
    # image = torchvision.io.read_image(filename,torchvision.io.image.ImageReadMode.RGB)
    # image = image/256
    return image.to(device)

def show_image(image_tensor):
    image = image_tensor.cpu() 
    image = image.squeeze(0)
    to_pil = transforms.ToPILImage()
    image = to_pil(image)
    plt.imshow(image)

def save_image(image_tensor,filename="Transferred image.jpg"):
    utils.save_image(image_tensor.cpu().clone(),filename)

# %%
def gram_matrix(tensor):
    reshaped_tensor = tensor.view(tensor.size()[0] * tensor.size()[1], tensor.size()[2] * tensor.size()[3]) 
    GT = torch.mm(reshaped_tensor, reshaped_tensor.t())/(tensor.size()[0] * tensor.size()[1] * tensor.size()[2] * tensor.size()[3])
    return GT

class ContentLossLayer(nn.Module):
    def __init__(self, element):
        super(ContentLossLayer, self).__init__()
        self.element = element.detach()

    def forward(self, input):
        self.loss = F.mse_loss(self.element, input)
        return input

class StyleLossLayer(nn.Module):
    def __init__(self, element):
        super(StyleLossLayer, self).__init__()
        self.element = gram_matrix(element).detach()

    def forward(self, input):
        GI = gram_matrix(input)
        self.loss = F.mse_loss(self.element, GI)
        return input

class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, input):
        return (input - self.mean) / self.std

# %%
# construct model
def construct_model(base_model, normalization_mean, normalization_std,image_style, image_content,content_layer_names,style_layer_names):
    model = nn.Sequential()
    # first add norm
    norm_layer = NormalizationLayer(normalization_mean, normalization_std).to(device)
    model.add_module("norm_0",norm_layer)

    i = 0
    for layer in base_model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            layer_name = "conv_"+str(i)
        elif isinstance(layer, nn.ReLU):
            layer_name = "relu_"+str(i)
            layer = nn.ReLU(inplace=False) # must use inplace=False otherwise cannot run
        elif isinstance(layer, nn.MaxPool2d):
            layer_name = "pool_"+str(i)
        # print(layer_name)
        model.add_module(layer_name, layer)

        if layer_name in content_layer_names:
            element = model(image_content).detach()
            content_loss_layer = ContentLossLayer(element)
            model.add_module("cl_"+str(i), content_loss_layer)

        if layer_name in style_layer_names:
            element = model(image_style).detach()
            style_loss_layer = StyleLossLayer(element)
            model.add_module("sl_"+str(i), style_loss_layer)

    # get the correct range of model.
    model = model[0:18]
    # print(model)
    return model

# %%

epoch = 0
def proc_transfer(base_model, normalization_mean, normalization_std,
                       image_content, image_style, image_init, num_iter=200,
                       style_coeff=1000000, content_coeff=1):

    content_layer_names = ["conv_2"]
    style_layer_names = ["conv_1", 
    "conv_2", 
    "conv_3", 
    "conv_4", 
    "conv_5"
    ]

    style_transfer_model = construct_model(base_model, normalization_mean, normalization_std, image_style, image_content,content_layer_names,style_layer_names)
    optimizer = optim.LBFGS([image_init.requires_grad_()])
    content_loss=list()
    style_loss=list()
    for layer in style_transfer_model.children():
        if isinstance(layer,ContentLossLayer):
            content_loss.append(layer)
        elif isinstance(layer,StyleLossLayer):
            style_loss.append(layer)
    
    min = 0
    max = 1
    global epoch
    while epoch <= num_iter:
        def iteration():
            image_init.data=torch.clamp(image_init.data,min,max)

            optimizer.zero_grad()
            style_transfer_model(image_init)

            J_style = style_coeff * sum([a.loss for a in style_loss])
            J_content = content_coeff * sum([a.loss for a in content_loss])
            # print([a.loss for a in style_losses])
            J_total = J_style + J_content
            J_total.backward()

            global epoch
            if epoch % 50 == 0:
                print("--------------------")
                print("Iteration time = "+str(epoch))
                print("Style Loss = {:2f}, Content Loss = {:2f}, Total Loss = {:2f}".format(J_style, J_content,J_total))
                print()
            epoch += 1 

            return J_total

        optimizer.step(iteration)

    epoch = 0
    return image_init
