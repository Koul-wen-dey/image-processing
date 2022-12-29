import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from mydataset import Image_dataset
import torchvision.transforms as T
from tqdm import tqdm
from myloss import IoU
from sklearn.model_selection import KFold,StratifiedKFold
from ranger import Ranger
from torch.optim import lr_scheduler

def my_collate_fn(batch):
    return tuple(zip(*batch))


def training(model, dataloader, optimizer, device,e):

    model.train()
    with tqdm(dataloader, unit='batch') as data:
        total_loss = 0
        
        for img, targets in data:
            data.set_description(f'Epoch{e+1}')
            img = [i.to(device) for i in img]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(img,targets)
            losses = sum(loss for loss in output.values())
            total_loss += losses
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print(f'train loss : {total_loss.item()}')

    return total_loss.item()

def validation(model, dataloader, device, idx):
    model.eval()
    loss = 0
    with torch.no_grad():
        for img, targets in dataloader:
            img = [i.to(device) for i in img]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            output = model(img)
            torch.cuda.empty_cache()
    print(f'model_{idx} loss : {loss}')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
batch_size = 8
learning_rate = 1.0e-3
epoch = 40
mean, std = 0.5557, 0.2051
hidden_layer = 256
path = './class_data/Train'
size = (350,350)
transform = T.Compose([T.Resize(size)])
kf = KFold(n_splits=4,shuffle=True,random_state=1)
kf2 = StratifiedKFold(n_splits=4,shuffle=True,random_state=1)


if __name__ == '__main__':
    train_set = Image_dataset(path, transform=transform)
    for idx, (train_id, valid_id) in enumerate(kf2.split(train_set,train_set.l)):
        
        new_loss = float('inf')
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=4)
        
        # model.load_state_dict(torch.load(f'./models/mask_rcnn_{idx}.pt')['model_state_dict'])
        model = model.to(device)
        optimizer = Ranger(model.parameters(),lr=learning_rate)
        sc = lr_scheduler.StepLR(optimizer,10)
        train_sample = SubsetRandomSampler(train_id)
        valid_sample = SubsetRandomSampler(valid_id)
        train_loader = DataLoader(train_set,batch_size=batch_size,collate_fn=my_collate_fn,sampler=train_sample,shuffle=False)
        valid_loader = DataLoader(train_set,batch_size=batch_size,collate_fn=my_collate_fn,sampler=valid_sample,shuffle=False)
        history = []

        # Training phase
        for e in range(epoch):
            total_loss = training(model,train_loader,optimizer,device,e)
            history.append(total_loss)
            if total_loss < new_loss:
                new_loss = total_loss
                torch.save({'model_state_dict':model.state_dict()},f'./models/mask_rcnn_{idx}.pt')
                print('save checkpoint')
            sc.step()
        with open(f'./log/loss_{idx}.txt','w') as f:
            for his in history:
                f.write(str(his)+'\n')
            f.close()

        # Validation phase
        # validation(model,valid_loader,device,idx)