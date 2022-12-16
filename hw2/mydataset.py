from torch.utils.data import Dataset
from glob import glob
import os
from PIL import Image
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
import torchvision.transforms.functional as F
import json
import torch
# from PIL.ImageMath import float

class Image_dataset(Dataset):
    def __init__(self,data_path,transform=None):
        super().__init__()
        self.transform = transform
        self.img_path = glob(os.path.join(data_path,'image/*.png'))
        self.label_path = glob(os.path.join(data_path,'label/*.json'))
        self.mask_path = glob(os.path.join(data_path,'mask/*.png'))
    
    def __getitem__(self, index):
        # print(self.img_path[index])
        img = read_image(self.img_path[index],ImageReadMode.GRAY)
        mask = read_image(self.mask_path[index],ImageReadMode.GRAY)
        img = F.convert_image_dtype(img,dtype=torch.float)
        mask = F.convert_image_dtype(mask,dtype=torch.float)
        with open(self.label_path[index]) as file:
            labels = json.load(file)

        boxes = []
        for label in labels['shapes']:
            boxes.append([
                label['points'][0][0],
                label['points'][0][1],
                label['points'][1][0],
                label['points'][1][1]])
        boxes = torch.as_tensor(boxes,dtype=torch.float)
        # print(boxes)
        target = {'masks':mask,'boxes':boxes,'labels':torch.ones((3,),dtype=torch.int8)}
        if self.transform is not None:
            img = self.transform(img)
            # target = self.transform(target)

        return img, target
    
    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    path = './class_data/Train/*/'
    trans = T.Compose([T.Resize((256,256))])
    a = Image_dataset(path,trans)
    print(a[0][1])