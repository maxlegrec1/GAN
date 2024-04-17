import os
import torchvision
from tqdm import tqdm
for file in tqdm(os.listdir("datasets/afhq/ablation512")):
    path = os.path.join("datasets/afhq/ablation512",file)
    img = torchvision.io.read_image(path).float()
    img = torchvision.transforms.Resize(256)(img)
    img = torchvision.utils.make_grid(img,normalize=True)
    torchvision.utils.save_image(img,path)