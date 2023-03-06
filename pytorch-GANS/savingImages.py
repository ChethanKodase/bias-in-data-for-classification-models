import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


import torch
from torchvision.utils import make_grid

import matplotlib
matplotlib.rcdefaults()

import matplotlib.pyplot as plt

device = torch.device('cuda')

import torch.nn as nn


from models import generator

image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

latent_size = 128

xb = torch.randn(batch_size, latent_size, 1, 1) # random latent tensors
#xb = (xb - xb.min()) / (xb.max() - xb.min())
generator.load_state_dict(torch.load('/home/ramana44/pytorch-GANS/saved_models/G.pth', map_location=torch.device(device)))

#generator.load('/home/ramana44/pytorch-GANS/saved_models/G.pth')
fake_images = generator(xb)


print(fake_images.shape)

for i in range(len(fake_images)):
    plt.axis("off")
    plt.imshow(fake_images[i].permute(1,2,0).cpu().detach().numpy())
    plt.savefig('/home/ramana44/pytorch-GANS/generatedImages/gen'+str(i)+'Img.jpg',bbox_inches='tight', pad_inches = 0)
    plt.close()

'''plt.axis("off")
plt.imshow(mean_decoding[0].permute(1,2,0).cpu().detach().numpy())
plt.savefig('/home/ramana44/pytorch-vae/wasserstein_saves/generations/mixOfNode_'+str(i)+'AndTwin'+str(twinIndx)+'.jpg',bbox_inches='tight', pad_inches = 0)
plt.close()
'''