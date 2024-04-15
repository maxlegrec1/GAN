import torch

def selected_opt(gen,disc,train_config):
    if train_config["optimizer"] =="nadam":
        gen_opt,disc_opt = torch.optim.NAdam(gen.parameters(),lr = train_config["generator_lr"],betas=(0.9,0.999)),torch.optim.NAdam(disc.parameters(),lr = train_config["discriminator_lr"],betas=(0.5,0.999))

        return gen_opt,disc_opt
    