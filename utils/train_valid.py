import os
import random
import torchvision
import torch

class Dataset():
    def __init__(self,dir,batch_size,scaling_factor) -> None:
        self.dir = dir
        self.scaling = scaling_factor
        self.batch_size = batch_size
        self.images_path = []

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
        images = torch.stack(images)        
        return (images.to("cuda").float()/127.5) - 1


def create_train_valid(dir,batch_size,scaling_factor):
    
    train_dir = os.path.join(dir,"train")
    valid_dir = os.path.join(dir,"val")


    train,valid = Dataset(train_dir,batch_size,scaling_factor), Dataset(valid_dir,batch_size,scaling_factor)

    return train,valid