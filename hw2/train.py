import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from mydataset import Image_dataset
import torchvision.transforms as T
from tqdm import tqdm
from myloss import IoULoss
from sklearn.model_selection import KFold

def my_collate_fn(batch):
    return tuple(zip(*batch))

def training(model, dataloader, loss, optimizer, device, idx,e):

    model.train()
    with tqdm(dataloader, unit='batch') as data:
        for img, targets in data:
            data.set_description(f'Epoch{e+1}')
            img = [i.to(device) for i in img]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(img,targets)
            losses = sum(loss for loss in output.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            data.set_description(f'loss:{losses}')
            torch.cuda.empty_cache()
        torch.save({'model_state_dict':model.state_dict()},f'./models/mask_rcnn_{idx}.pt')
        print('save checkpoint')
    pass



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
batch_size = 4
learning_rate = 1.0e-3
epoch = 30
mean, std = 0.5557, 0.2051
hidden_layer = 256
path = './class_data/Train/*/'
size = (256,256)
transform = T.Compose([T.Resize(size)])
kf = KFold(n_splits=4,shuffle=True,random_state=1)

if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),learning_rate)
    loss_func = IoULoss()
    train_set = Image_dataset(path, transform=T.Normalize(mean,std))

    
    for idx, (train_id, valid_id) in enumerate(kf.split(train_set)):
        train_sample = SubsetRandomSampler(train_id)
        valid_sample = SubsetRandomSampler(valid_id)
        train_loader = DataLoader(train_set,batch_size=batch_size,collate_fn=my_collate_fn,sampler=train_sample)
        valid_loader = DataLoader(train_set,batch_size=batch_size,collate_fn=my_collate_fn,sampler=valid_sample)
        for e in range(epoch):
            training(model,train_loader,loss_func,optimizer,device,idx,e)
            