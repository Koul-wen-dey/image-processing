import torch
from mydataset import Image_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        data = data.float()
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std



if __name__ == '__main__':
    path = './class_data/Train/*/'
    trans = transforms.Compose([transforms.Resize((256,256)),transforms.Normalize(0.5557,0.2051)])
    a = Image_dataset(path,transform=transforms.Resize((256,256)))
    print(a[0][0])

    # t = DataLoader(a,batch_size=4)
    # for img, mask, box in t:
    #     print(box)
    # print(a[0][0])
    # mean, std = get_mean_and_std(t)
    # print(mean,std)
    # 0.5592, 0.2034
    # (0.5527,0.5591,0.5766),(0.2022,0.2033,0.2139)
    # 141.7066, 52.2904 no convertion to float
    # 0.5557, 0.2051 after convertion to float