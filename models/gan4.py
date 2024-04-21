import torch
from torch import nn

def create_gen(batch_size):
    return Generator(batch_size).to("cuda")


def create_disc():
    return Discriminator().to("cuda")



def get_loss(q, k,loss, tau=1):
    """
    This function calculates the negative log likelihood (NLL) kernel, 
    which is a part of the objective function in some learning algorithms.

    Args:
    q: A torch tensor of shape (n, d) where:
        - n is the batch size.
        - d is the dimensionality of each vector.
    k: A torch tensor of shape (n, d) with the same structure as q.
    tau: A scalar tensor representing the temperature parameter.

    Returns:
    A torch tensor of shape (n,) containing the NLL kernel value for each 
    element in the batch.
    """

    # Calculate the exponentiated inner product between q_i and k_j for all i, j in the batch.
    qk_exp = torch.softmax(torch.matmul(q,k.transpose(0,1)),dim = -1)
    identity =torch.arange(qk_exp.shape[-1])
    return loss(qk_exp,identity)


class Discriminator(nn.Module):
    def __init__(self):
        nc = 3
        ndf = 64
        super(Discriminator, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.to_many_2 = ToMany(ndf)
        self.block_2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.to_many_3 = ToMany(ndf*2)
        self.block_3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.to_many_4 = ToMany(ndf*4)
        self.block_4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.to_many_5 = ToMany(ndf*8)
        self.block_5 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.to_many_6 = ToMany(ndf*8)
        self.block_6 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),)
        
        self.to_many_7 = ToMany(ndf*8)
        self.block_7 = nn.Sequential(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

       
        self.contrast_real = torch.nn.Conv2d(ndf*8,ndf*4,1)
        self.contrast_fake = torch.nn.Conv2d(ndf*8,ndf*4,1)
        self.contrast_real_d = torch.nn.Linear(ndf*4,ndf*4,bias=False)
        self.contrast_fake_d = torch.nn.Linear(ndf*4,ndf*4,bias=False)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input):
        rgb4,rgb8,rgb16,rgb32,rgb64,rgb128,rgb256 = input

        disc4 = self.to_many_7(rgb4)
        disc4 = self.block_7(disc4).view(-1,)

        disc8 = self.to_many_6(rgb8)
        disc8 = self.block_6(disc8)
        disc8 = self.block_7(disc8).view(-1,)

        disc16 = self.to_many_5(rgb16)
        disc16 = self.block_5(disc16)
        disc16 = self.block_6(disc16)
        disc16 = self.block_7(disc16).view(-1,)

        disc32 = self.to_many_4(rgb32)
        disc32 = self.block_4(disc32)
        disc32 = self.block_5(disc32)
        disc32 = self.block_6(disc32)
        disc32 = self.block_7(disc32).view(-1,)

        disc64 = self.to_many_3(rgb64)
        disc64 = self.block_3(disc64)
        disc64 = self.block_4(disc64)
        disc64 = self.block_5(disc64)
        disc64 = self.block_6(disc64)
        disc64 = self.block_7(disc64).view(-1,)

        disc128 = self.to_many_2(rgb128)
        disc128 = self.block_2(disc128)
        disc128 = self.block_3(disc128)
        disc128 = self.block_4(disc128)
        disc128 = self.block_5(disc128)
        disc128 = self.block_6(disc128)
        disc128 = self.block_7(disc128).view(-1,)

        disc256 = rgb256
        disc256 = self.block_1(disc256)
        disc256 = self.block_2(disc256)
        disc256 = self.block_3(disc256)
        disc256 = self.block_4(disc256)
        disc256 = self.block_5(disc256)
        disc256 = self.block_6(disc256)
        disc256 = self.block_7(disc256).view(-1,)

        #print(mainout.shape)

        #qk_real = self.contrast_real(out).view(-1,64*4)
        #qk_fake = self.contrast_fake(out).view(-1,64*4)


        #qk_real=self.contrast_real_d(qk_real)
        #qk_fake = self.contrast_fake_d(qk_fake)

        #q_real,k_real = torch.chunk(qk_real,2)
        #q_fake,k_fake = torch.chunk(qk_fake,2)

        #real_loss = get_loss(q_real,k_real,self.loss)

        #fake_loss = get_loss(q_fake,k_fake,self.loss)

        return disc4,disc8,disc16,disc32,disc64,disc128,disc256
    
class Generator(nn.Module):
    def __init__(self,batch_size):
        ngf = 64
        nc = 3
        nz=100
        super(Generator, self).__init__()
        self.batch_size = batch_size

        self.block_1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),)
        #4*4
        self.rgb_1 = ToRGB(ngf*8)

        self.block_2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),)
        #8*8
        self.rgb_2 = ToRGB(ngf*8)

        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),)
        #16*16
        self.rgb_3 = ToRGB(ngf*8)
        
        self.block_4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),)
        #32*32
        self.rgb_4 = ToRGB(ngf*4)

        self.block_5 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),)
        #64*64
        self.rgb_5 = ToRGB(ngf*2)
        
        self.block_6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),)
        #128*128
        self.rgb_6 = ToRGB(ngf)
        
        self.block_7 = nn.Sequential(
            nn.ConvTranspose2d( ngf, ngf, 4, 2, 1, bias=False),)
        #256*256
        self.rgb_7 = ToRGB(ngf)

    def forward(self):
        input = torch.randn(self.batch_size, 100, 1, 1, device="cuda")
        main = self.block_1(input)
        rgb4 = self.rgb_1(main)
        main = self.block_2(main)
        rgb8 = self.rgb_2(main)
        main = self.block_3(main)
        rgb16 = self.rgb_3(main)
        main = self.block_4(main)
        rgb32 = self.rgb_4(main)
        main = self.block_5(main)
        rgb64 = self.rgb_5(main)
        main = self.block_6(main)
        rgb128 = self.rgb_6(main)
        main = self.block_7(main)
        rgb256 = self.rgb_7(main)

        return (rgb4,rgb8,rgb16,rgb32,rgb64,rgb128,rgb256)
    

class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,3,1) 
    
    def forward(self,input):
        out = self.conv(input)
        out = torch.tanh(out)

        return out
    
class ToMany(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(3,out_channels,1) 
    
    def forward(self,input):
        out = self.conv(input)
        return out