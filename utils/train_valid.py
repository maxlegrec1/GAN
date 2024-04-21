import os
import random
import torchvision
import torch
from ada import AdaptiveDiscriminatorAugmentation
class Dataset():
    def __init__(self,dir,batch_size,scaling_factor) -> None:
        self.dir = dir
        self.scaling = scaling_factor
        self.batch_size = batch_size
        self.images_path = []


        self.aug = AdaptiveDiscriminatorAugmentation(
        xflip=1, 
        rotate90=1,
        xint=1, 
        scale=1, 
        rotate=1, 
        aniso=1,
        xfrac=1, 
        brightness=1, 
        contrast=1, 
        lumaflip=1,
        hue=1, 
        saturation=1,
        ).to("cuda")

        self.aug.set_p(0)
        for subdir in os.listdir(self.dir):
            subdir_path = os.path.join(self.dir,subdir)
            for image in os.listdir(subdir_path) :
                self.images_path.append(os.path.join(subdir_path,image))

        self.size = len(self.images_path)
        random.shuffle(self.images_path)
        self.pointer = 0

    def __next__(self):
        images = []
        for _ in range(self.batch_size):
            img = torchvision.io.read_image(self.images_path[self.pointer])
            img = torchvision.transforms.Resize(self.scaling)(img)
            if random.random()<0.5:
                img = torch.flip(img,[-1])
                
            images.append(img)
            self.pointer+=1
            if self.pointer == self.size:
                self.pointer = 0
        images = (torch.stack(images).to("cuda").float()/127.5) - 1
        images = self.aug(images)        
        return to_many(images)


def create_train_valid(dir,batch_size,scaling_factor):
    
    train_dir = os.path.join(dir,"train")
    valid_dir = os.path.join(dir,"val")


    train,valid = Dataset(train_dir,batch_size,scaling_factor), Dataset(valid_dir,batch_size,scaling_factor)

    return train,valid

def to_many(images):

    img4 = torchvision.transforms.Resize(4)(images)

    img8 = torchvision.transforms.Resize(8)(images)

    img16 = torchvision.transforms.Resize(16)(images)

    img32 = torchvision.transforms.Resize(32)(images)

    img64 = torchvision.transforms.Resize(64)(images)

    img128 = torchvision.transforms.Resize(128)(images)

    return img4,img8,img16,img32,img64,img128,images