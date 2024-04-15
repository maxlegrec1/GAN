import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
#from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D
from tqdm import tqdm
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

vae = vae
img_dir_list= ["datasets/afhq/train/cat","datasets/afhq/train/dog","datasets/afhq/train/wild"]
out_dir = "datasets/afhq/latents"
img_list = []
for imgdir in img_dir_list:
    for file in os.listdir(imgdir):
        img_list.append(os.path.join(imgdir,file))

for i,img_path in tqdm(enumerate(img_list)):
    img = torchvision.io.read_image(img_path).float()/127.5 - 1
    #grid = torchvision.utils.make_grid(img,normalize=True).permute(1,2,0)

    imgbatch = img.view(1,3,img.shape[1],img.shape[2])
    sample = vae.encoder(imgbatch)
    sample = vae.quant_conv(sample)[:,:4]
    np.savez_compressed(os.path.join(out_dir,f"{i}.npz"),sample.detach().numpy())
    #out = vae.decode(sample).sample
    #outgrid = torchvision.utils.make_grid(out,normalize=True).permute(1,2,0)


    #fig, ax = plt.subplots(nrows=1, ncols=2)
    #ax[0].imshow(grid.cpu())
    #ax[1].imshow(outgrid.cpu())
    #plt.show()