import torch
from torch.utils.data import DataLoader
import torchvision
from mydataset import Image_dataset
import torchvision.transforms as T
from train import my_collate_fn
import matplotlib.pyplot as plt
from torchvision.io import read_image,ImageReadMode
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms

def eval(model,):
    pass
num_classes = 4
hidden_layer = 256
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = './class_data/Val/*/'
transform = T.Compose([T.Resize((350,350))])

if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes)


    count = 0
    check = torch.load('./models/mask_rcnn_0.pt')
    # check = torch.load('./backup/mask_rcnn_0.pt')
    model.load_state_dict(check['model_state_dict'])
    model = model.to(device)
    # test = Image_dataset(path,transform=transform)
    # test_loader = DataLoader(test,batch_size=1,collate_fn=my_collate_fn)
    model.eval()
    with torch.no_grad():
        # for img, targets in test_loader:
        img = read_image('./class_data/Val/scratch/image/Converted_ 3138.png',ImageReadMode.RGB)
        img2 = F.convert_image_dtype(img,dtype=torch.float32)
        img2 = transform(img2)
        img2 = [img2.to(device)]
        output = model(img2)
        output = output[0]
        # print(output)
        # exit()
        # print(torch.unique(output['masks']))
        tmp = output['masks'][0].shape
        M = torch.zeros(tmp,dtype=torch.uint8)
        for a in output['masks']:
            M[torch.where(a>0.6)] = 255
        table = {0:'background',1:'uncover',2:'uneven',3:'scratch'}
        indexes = nms(output['boxes'],output['scores'],0.09)
        labels = [table[i.item()] for i in output['labels'][indexes]]
        output['boxes'] = output['boxes'][indexes]
        img = transform(img)
        img = draw_bounding_boxes(img,output['boxes'],labels)

        plt.imshow(img.permute(1,2,0).cpu(),cmap='gray')
        plt.show()
        plt.imshow(M.permute(1,2,0).cpu(),cmap='gray')
        plt.show()
        # if count == 1:
        #     break
        # count += 1