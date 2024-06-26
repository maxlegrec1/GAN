import pytorch_fid.fid_score
import yaml
import os
import wandb
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import pytorch_fid

from utils.train_valid import create_train_valid
from utils.models import selected_models
from utils.optimizers import selected_opt
from utils.diffusion import Diffusion

def main(config):
    if not os.path.exists(config["run_name"]):
        os.mkdir(config["run_name"])

    if not os.path.exists("fake_ds"):
        os.mkdir("fake_ds")
    
    train,valid = create_train_valid(config["dataset"],config["training"]["batch_size"],config["training"]["scaling_factor"])

    gen, disc = selected_models(config)

    gen_opt,disc_opt = selected_opt(gen,disc,config["training"])

    diffusion = Diffusion()
    loss = torch.nn.BCELoss()
    def loss_fun(x,y):
        disc4,disc8,disc16,disc32,disc64,disc128,disc256 = x
        loss4 = loss(disc4,y)
        loss8 = loss(disc8,y)
        loss16 = loss(disc16,y)
        loss32 = loss(disc32,y)
        loss64 = loss(disc64,y)
        loss128 = loss(disc128,y)
        loss256 = loss(disc256,y)
        return (loss4+loss8+loss16+loss32+loss64+loss128+loss256)/7

    total_train_steps = 0

    ins_weight = 0
    augment_probability = 0

    if config["use_wandb"]:
        wandb.init(project="GAN",config=config,name = config["run_name"])

    real_label = 1.
    fake_label = 0.
    real_label = torch.full((config["training"]["batch_size"],), real_label, dtype=torch.float, device="cuda")
    fake_label = torch.full((config["training"]["batch_size"],), fake_label, dtype=torch.float, device="cuda")

    for epoch in range(config["training"]["epochs"]):
        print("epoch : ",epoch)
        total_train_gen_loss = []
        total_train_disc_loss = []
        disc_real_output = []
        disc_fake_output = []
        total_disc_accuracy = []
        total_valid_disc_loss = []
        total_valid_gen_loss = []
        augment_probability = min(0.5,augment_probability)
        #train.aug.set_p(augment_probability)
        for train_step in range(train.size//train.batch_size):
            total_train_steps+=1
            print(train_step)
            images = next(train)

            #update G
            fake_images = gen()
            #blurr the images 
            #images_b,t_real = diffusion(images)
            #fake_images_b,t_fake = diffusion(fake_images)

            dfo= disc(fake_images)
            g_loss = loss_fun(dfo, real_label) 
            gen_opt.zero_grad()
            g_loss.backward()
            gen_opt.step()

           #update D
            dro = disc(images)
            real_loss = loss_fun(dro, real_label-(torch.rand_like(real_label)<0.1).float()) 
            #real_loss = loss_fun(disc(images,t_real), real_label) 
            dfo = disc((img.detach() for img in fake_images))
            fake_loss = loss_fun(dfo, fake_label) 
            d_loss = (real_loss + fake_loss) / 2
            disc_opt.zero_grad()
            d_loss.backward()
            disc_opt.step()

            disc_accuracy = ((torch.sign(dro[-1] - 0.5)+1).mean() + (torch.sign(0.5-dfo[-1])+1).mean())/4

            print(g_loss.item(),d_loss.item())

            if total_train_steps%config["training"]["plot_gen_images_freq"]== 0 :
                
                fake_images = (fake_images[-1].detach().cpu()+1)*127.5
                images = (images[-1].detach().cpu()+1)*127.5
                grid = vutils.make_grid(images[:8],padding=2, normalize=True)
                vutils.save_image(grid,os.path.join(config["run_name"],f"{total_train_steps}real.png"))
                grid = vutils.make_grid(fake_images[:8],padding=2, normalize=True)
                vutils.save_image(grid,os.path.join(config["run_name"],f"{total_train_steps}fake.png"))

            #update losses
            total_train_disc_loss.append(d_loss.item())
            total_train_gen_loss.append(g_loss.item())
            total_disc_accuracy.append(disc_accuracy.detach().cpu().item())
            disc_real_output.append(dro[-1].detach().cpu().mean().item())
            disc_fake_output.append(dfo[-1].detach().cpu().mean().item())

        #perform validation in no grad
        with torch.no_grad():
            for val_step in range(valid.size//train.batch_size):
                fake_images = gen()[-1].detach().cpu()
                for i in range(train.batch_size):
                    vutils.save_image(vutils.make_grid(fake_images[i],normalize=True),f"fake_ds/{val_step}_{i}.png")

        #calculate loss summaries
        #apply mean to the losses totals
        total_train_gen_loss = sum(total_train_gen_loss)/len(total_train_gen_loss)
        total_train_disc_loss = sum(total_train_disc_loss)/len(total_train_disc_loss)
        disc_fake_output = sum(disc_fake_output)/len(disc_fake_output)
        disc_real_output = sum(disc_real_output)/len(disc_real_output)
        total_disc_accuracy = sum(total_disc_accuracy)/len(total_disc_accuracy)
        
        if total_disc_accuracy>=0.95:
            augment_probability+=0.1

        if total_disc_accuracy<=0.85:
            augment_probability-=0.1

        fid = pytorch_fid.fid_score.calculate_fid_given_paths(paths=["datasets/afhq/ablation512","fake_ds"],batch_size=config["training"]["batch_size"],device="cuda",dims=2048)

        if config["use_wandb"]:
            images = wandb.Image(grid)
            wandb.log({"train/gen_loss" : total_train_gen_loss, "train/disc_loss": total_train_disc_loss, \
                        "disc_real_output": disc_real_output, "disc_fake_output": disc_fake_output, "FID": fid, "gen_output": images, "disc accuracy": total_disc_accuracy})
            #plot the last images :

if __name__ == "__main__":
    config_name = "configs/template.yaml"
    with open(config_name) as f:
        config = yaml.safe_load(f)

    main(config)