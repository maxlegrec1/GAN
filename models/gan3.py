import torch
from torch import nn

def create_gen(batch_size):
    return Generator2(batch_size).to("cuda")


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
    identity =torch.arange(qk_exp.shape[-1]).to("cuda")
    return loss(qk_exp,identity)


class Discriminator(nn.Module):
    def __init__(self):
        nc = 3+1
        ndf = 64
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf, 4, 1, padding="same", bias=False),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf*2, ndf*2, 4, 1, padding="same", bias=False),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf*4, ndf*4, 4, 1, padding="same", bias=False),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, ndf*8, 4, 1, padding="same", bias=False),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, ndf*8, 4, 1, padding="same", bias=False),
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf*8, 4, 1, 0, bias=False),

        )
        self.contrast_real = torch.nn.Conv2d(ndf*8,ndf*4,1)
        self.contrast_fake = torch.nn.Conv2d(ndf*8,ndf*4,1)
        self.contrast_real_d = torch.nn.Linear(ndf*4,ndf*4,bias=False)
        self.contrast_fake_d = torch.nn.Linear(ndf*4,ndf*4,bias=False)
        self.loss = torch.nn.CrossEntropyLoss()
        self.conv_out = torch.nn.Conv2d(ndf*8,1,1)
    def forward(self, input,t):
        b,_,H,W = input.size()
        t = t.view(-1,1,1,1)/500
        ones = torch.ones((b,1,H,W)).to("cuda") * t
        input = torch.cat([input,ones],dim = 1)
        out = self.main(input)
        mainout = self.conv_out(out)
        mainout = torch.sigmoid(mainout)
        #print(mainout.shape)

        qk_real = self.contrast_real(out).view(-1,64*4)
        qk_fake = self.contrast_fake(out).view(-1,64*4)


        #qk_real=self.contrast_real_d(qk_real)
        #qk_fake = self.contrast_fake_d(qk_fake)

        q_real,k_real = torch.chunk(qk_real,2)
        q_fake,k_fake = torch.chunk(qk_fake,2)

        real_loss = get_loss(q_real,k_real,self.loss)

        fake_loss = get_loss(q_fake,k_fake,self.loss)

        return mainout.view(-1,),real_loss,fake_loss
    
class Generator(nn.Module):
    def __init__(self,batch_size):
        ngf = 64
        nc = 3
        nz=100
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            nn.Conv2d(ngf*8,ngf*8, 4, padding="same", bias=False),
                # state size. ``(ngf*8) x 4 x 4``
            
            Block(ngf*8,ngf*8),
            # state size. ``(ngf*4) x 8 x 8``
            Block(ngf*8,ngf*8),
            # state size. ``(ngf*2) x 16 x 16``
            Block(ngf*8,ngf*4),

            Block(ngf*4,ngf*2),

            Block(ngf*2,ngf),

            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self):
        input = torch.randn(self.batch_size, 100, 1, 1, device="cuda")
        out = self.main(input)
        return out
    


class Block(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super().__init__()
        self.in_conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bs = nn.BatchNorm2d(out_channels)
        self.relu= nn.ReLU()
        self.out_conv = nn.Conv2d(out_channels,out_channels, 4, padding="same", bias=False)


    def forward(self,input):
        out1 = self.in_conv(input)
        out = self.bs(out1)
        out = self.relu(out)
        out = self.out_conv(out)+out1
        return out


  
class Generator2(nn.Module):
    def __init__(self,batch_size):
        ngf = 64
        nc = 3
        nz=100
        super(Generator2, self).__init__()
        self.batch_size = batch_size
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
                # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self):
        input = torch.randn(self.batch_size, 100, 1, 1, device="cuda")
        out = self.main(input)
        return out
    
