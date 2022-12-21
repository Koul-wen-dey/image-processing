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
        self.hash = {'background':0,'powder_uncover':1,'powder_uneven':2,'scratch':3}
        self.normalize = T.Normalize((0.5557),(0.2051))


    def __getitem__(self, index):
        img = read_image(self.img_path[index],ImageReadMode.GRAY)
        mask = read_image(self.mask_path[index],ImageReadMode.GRAY)
        img = F.convert_image_dtype(img,dtype=torch.float)
        mask = F.convert_image_dtype(mask,dtype=torch.uint8)
        num_ids = len(torch.unique(mask)[1:])
        with open(self.label_path[index]) as file:
            table = json.load(file)

        boxes = []
        label_t = []
        for label in table['shapes']:
            x = [label['points'][0][0],label['points'][1][0]]
            y = [label['points'][0][1],label['points'][1][1]]
            boxes.append([
                min(x),
                min(y),
                max(x),
                max(y)
            ])
            label_t.append(self.hash[label['label']])
        boxes = torch.as_tensor(boxes,dtype=torch.float)
        
        mask_list = []
        for b in boxes:
            tmp = torch.zeros(mask.shape[1:])
            x1, y1 = int(torch.round(b[0]).item()), int(torch.round(b[1]).item())
            x2, y2 = int(torch.round(b[2]).item()), int(torch.round(b[3]).item())
            tmp[y1:y2, x1:x2] = 1
            mask_list.append(torch.where(tmp > 0, mask[0], tmp))
        mask = torch.tensor([m.tolist() for m in mask_list])

        past_len = mask.shape[-1]
        target = {
            'masks':mask,
            'boxes':boxes,
            'labels':torch.tensor(label_t,dtype=torch.int64),
            'area':(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'image_id':torch.tensor([index]),
            'iscrowd':torch.zeros((num_ids,),dtype=torch.int64)
        }
        img = self.normalize(img)
        if self.transform is not None:
            img = self.transform(img)
            target['masks'] = self.transform(target['masks'])
            target['boxes'] = target['boxes'] / past_len * img.shape[-1] 
            target['area'] = target['area'] / past_len**2 * img.shape[-1]**2

        return img, target
    
    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    path = './class_data/Train/*/'
    trans = T.Compose([T.Resize((256,256))])
    a = Image_dataset(path,trans)
    print(a[1][1])