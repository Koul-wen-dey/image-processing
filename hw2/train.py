import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from mydataset import Image_dataset
import torchvision.transforms as T
from tqdm import tqdm
from myloss import IoULoss


def training(model, dataloader, epoch, loss, optimizer, device):
    model.train()
    for i in range(epoch):
        with tqdm(dataloader, unit='batch') as data:
            correct = 0
            for img, target in data:
                data.set_description(f'Epoch{i+1}')
                img, target = img.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(img,target)
                error = loss(output,)
                error.backward()
                optimizer.step()
            torch.save({'model_state_dict':model.state_dict()},'./models/mask_rcnn_adam.pt')
            print('save checkpoint')
    pass

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
batch_size = 32
learning_rate = 1.0e-3
epoch = 100
mean, std = 0.5557, 0.2051
hidden_layer = 256
path = './class_data/Train/*/'
size = (256,256)
transform = T.Compose([T.Resize(size),T.Normalize(mean,std)])

if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),learning_rate)
    loss_func = IoULoss()
    train_set = Image_dataset(path, transform)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    training(model,train_loader, epoch, loss_func, optimizer, device)