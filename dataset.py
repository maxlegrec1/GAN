import os
import random
import torchvision
import torch
import numpy as np

class Dataset():
    def __init__(self,dir,batch_size,scaling_factor) -> None:
        self.dir = dir
        self.scaling = scaling_factor
        self.batch_size = batch_size
        self.images_path = []

        for image in os.listdir(self.dir) :
                self.images_path.append(os.path.join(self.dir,image))

        self.size = len(self.images_path)
        random.shuffle(self.images_path)
        self.pointer = 0

    def __next__(self):
        images = []
        for _ in range(self.batch_size):
            img = np.load(self.images_path[self.pointer])['arr_0']
            img = torch.from_numpy(img)
            img = img.view(img.shape[1],img.shape[2],img.shape[3])
            images.append(img)
            self.pointer+=1
            if self.pointer == self.size:
                self.pointer = 0
        images = torch.stack(images)        
        return (images.to("cuda").float())


def create_train_valid(dir,batch_size,scaling_factor):
    
    train_dir = dir
    valid_dir = dir


    train,valid = Dataset(train_dir,batch_size,scaling_factor), Dataset(valid_dir,1,scaling_factor)

    return train,valid