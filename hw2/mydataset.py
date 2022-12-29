from torch.utils.data import Dataset
from glob import glob
import os
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
import torchvision.transforms.functional as F
import json
import torch


class Image_dataset(Dataset):
    def __init__(self,data_path,transform=None):
        super().__init__()
        self.transform = transform
        data_path += '/*/'
        self.img_path = glob(os.path.join(data_path,'image/*.png'))
        self.label_path = glob(os.path.join(data_path,'label/*.json'))
        self.mask_path = glob(os.path.join(data_path,'mask/*.png'))
        self.hash = {'background':0,'powder_uncover':1,'powder_uneven':2,'scratch':3}
        # self.normalize = T.Normalize((0.5557),(0.2051))
        self.l = []
        for path in self.img_path:
            if 'uncover' in path:
                self.l.append(1)
            elif 'uneven' in path:
                self.l.append(2)
            elif 'scratch' in path:
                self.l.append(3)
            else:
                self.l.append(0)
        # print(self.img_path)

    def __getitem__(self, index):
        img = read_image(self.img_path[index],ImageReadMode.RGB)
        mask = read_image(self.mask_path[index],ImageReadMode.GRAY)
        img = F.convert_image_dtype(img,dtype=torch.float32)
        num_ids = len(torch.unique(mask)[1:])
        with open(self.label_path[index]) as file:
            table = json.load(file)

        boxes = []
        label_t = []
        # print(table['shape'])
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
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        

        mask_list = []
        for b in boxes:
            tmp = torch.zeros(mask.shape[1:])
            x1, y1 = int(torch.round(b[0]).item()), int(torch.round(b[1]).item())
            x2, y2 = int(torch.round(b[2]).item()), int(torch.round(b[3]).item())
            tmp[y1:y2, x1:x2] = 1
            mask_list.append(torch.where(mask[0] > 0, tmp, 0))
        mask = torch.tensor([m.tolist() for m in mask_list],dtype=torch.uint8)
        # print(mask.shape)
        (height, width) = mask.shape[1:]
        # print(height, width)
        target = {
            'masks':mask,
            'boxes':boxes,
            'labels':torch.tensor(label_t,dtype=torch.int64),
            'area':(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'image_id':torch.tensor([index]),
            'iscrowd':torch.zeros((num_ids,),dtype=torch.uint8)
        }
        
        if self.transform is not None:
            img = self.transform(img)
            mask_resize = T.Resize((img.shape[-1],img.shape[-1]),interpolation=T.InterpolationMode.NEAREST)
            target['masks'] = mask_resize(target['masks'])

            # print(img.shape, target['masks'].shape)
            for t in target['boxes']:
                t[0] = t[0] / width * img.shape[-1]
                t[2] = t[2] / width * img.shape[-1]
                t[1] = t[1] / height * img.shape[-1]
                t[3] = t[3] / height * img.shape[-1]
            target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        return img, target
    
    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    path = './class_data/Train'
    trans = T.Compose([T.Resize((256,256))])
    a = Image_dataset(path,trans)
    # print(a.l)
    print(a[0][1])
    # a[1][1]