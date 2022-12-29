from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap,QImage
import hw2_ui
import sys
import os
from time import time
import torch
from torch.utils.data import DataLoader
from mydataset import Image_dataset
import torchvision.models.detection
import torchvision.transforms as T
import torchvision.transforms.functional as F
from train import my_collate_fn
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
idx = 0
model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=4)
model.load_state_dict(torch.load(f'./backup/mask_rcnn_{idx}.pt')['model_state_dict'])
model = model.to(device)
size = (350,350)
transform = T.Compose([T.Resize(size)])




class GUI(QMainWindow,hw2_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.images = []
        self.prediction = []
        self.annotation = []
        self.table = {0:'background',1:'uncover',2:'uneven',3:'scratch'}
        self.select_folder.clicked.connect(self.open_folder)
        self.show_result.clicked.connect(self.inference)
        self.next.clicked.connect(self.next_img)
        self.previous.clicked.connect(self.prev_img)
    
    def prev_img(self):
        tmp = self.idx - 1
        if tmp < 0:
            print('out of range')
        else:
            self.idx -= 1
            self.update_frame()

    def next_img(self):
        tmp = self.idx + 1
        if tmp+1 > self.data_length:
            print('out of range')
        else:
            self.idx += 1
            self.update_frame()

    def put_image(self,img,which:str):
        qimg = QImage(F.convert_image_dtype(img.permute(1,2,0),dtype=torch.uint8).cpu().numpy().tobytes(), 350, 350, 350*3, QImage.Format_RGB888)
        qimg = QPixmap.fromImage(qimg)
        if which == 'ori':
            self.ori_img.setPixmap(qimg)
            self.ori_img.show()    
        elif which == 'det':
            self.det_img.setPixmap(qimg)
            self.det_img.show()
        elif which == 'seg':
            self.seg_img.setPixmap(qimg)
            self.seg_img.show()

    def put_segmentation(self,idx):
        masks = self.prediction[idx]['masks']
        M = torch.zeros((3,350,350),dtype=torch.uint8)
        for a in masks:
            M[torch.where(a>0.9)] = 255
        self.put_image(M,'seg')

    def draw_bboxes(self,idx):
        predictions = self.prediction[idx]
        targets = self.annotation[idx]['boxes']
        indexes = nms(predictions['boxes'],predictions['scores'],0.05)
        labels = [self.table[i.item()] for i in predictions['labels'][indexes]]
        bbox = predictions['boxes'][indexes]
        img = draw_bounding_boxes(F.convert_image_dtype(self.images[idx],dtype=torch.uint8),bbox,labels,'Red')
        img = draw_bounding_boxes(img,targets,colors='Green')
        self.put_image(img,'det')

    def update_frame(self):
        try:
            tmp = self.prediction[self.idx]
            self.cur_img.setText(f'Current Image: {self.idx}/{self.data_length}')
            self.pred.setText(f'Prediction: {self.table[tmp["labels"][0].item()]}')
            self.put_image(self.images[self.idx],'ori')
            self.draw_bboxes(self.idx)
            self.put_segmentation(self.idx)
            self.show()
        except:
            pass
    
    def inference(self):
        infer_set = Image_dataset(self.folder,transform=transform)
        infer_loader = DataLoader(infer_set,batch_size=8,collate_fn=my_collate_fn,shuffle=False)

        count = 0
        model.eval()
        with torch.no_grad():
            start = time()
            for img, targets in infer_loader:
                img = [i.to(device) for i in img]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                output = model(img)
                self.images.extend(img)
                self.prediction.extend(output)
                self.annotation.extend(targets)
                torch.cuda.empty_cache()
                print(f'batch{count} done.')
                count += 1
                # if count == 1:
                #     break
            self.cost_time = time() - start
        self.data_length = len(infer_set)
        self.fps.setText(f'FPS: {self.cost_time / self.data_length}')
        self.idx = 0
        self.update_frame()
        self.show()
        pass

    def open_folder(self):
        try:
            self.folder = QFileDialog.getExistingDirectory(self,'開啟資料夾',os.getcwd())
            print(self.folder)
        except:
            pass

if __name__ == '__main__':
    a = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(a.exec_())