import torch
from mydataset import Image_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from glob import glob
import os
from torchvision.io import read_image,ImageReadMode
import torchvision.transforms.functional as F


class for_normal(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.img_path = glob(os.path.join(data_path,'image/*.png'))
        self.label_path = glob(os.path.join(data_path,'label/*.json'))
        self.mask_path = glob(os.path.join(data_path,'mask/*.png'))
    def __getitem__(self, index):
        img = read_image(self.img_path[index],ImageReadMode.RGB)
        return img, 0
    def __len__(self):
        return len(self.img_path)

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # print(data)
        data = data.float()
        # print(data)
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std



if __name__ == '__main__':
    path = './class_data/Train/*/'
    # trans = transforms.Compose([transforms.Resize((256,256)),transforms.Normalize(0.5557,0.2051)])
    a = for_normal(path)
    img = a[0][0]
    # print(img)
    test = transforms.Normalize((140.3623, 142.0074, 146.5404),(51.8614, 52.1720, 54.9593))
    img = test(img.float())
    # print(img)
    print(torch.unique(img))
    print(F.convert_image_dtype(img,dtype=torch.float32))
    # t = DataLoader(a,batch_size=1)
    # exit()
    # for img, mask, box in t:
    #     print(box)
    # print(a[0][0])
    # mean, std = get_mean_and_std(t)
    # print(mean,std)

    # (140.3623, 142.0074, 146.5404), (51.8614, 52.1720, 54.9593)
    # 0.5592, 0.2034
    # (0.5527,0.5591,0.5766),(0.2022,0.2033,0.2139)
    # 141.7066, 52.2904 no convertion to float
    # 0.5557, 0.2051 after convertion to float