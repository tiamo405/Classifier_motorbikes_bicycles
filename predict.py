import numpy as np 
import cv2
import os
import torch
import torchvision
import argparse

from utils import utils_save_cfg, utils_model
from torch import nn
from torchvision import transforms
from config import config

class Model():
    
    def __init__(self, name_model, nb_classes, load_height, load_width ,
                 checkpoint_dir, num_train, num_ckpt, device):

        self.model = torchvision.models.resnet101(pretrained = False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2) # khởi tạo mô hình = trẻ 3 tuổi
        self.device = device # gán biến
        self.model.to(self.device)

        self.checkpoint_model = os.path.join(checkpoint_dir, name_model, num_train, num_ckpt+'.pth') # checkpoints/resnet101/0001/1.pth
 
        self.model.load_state_dict(torch.load(self.checkpoint_model, map_location=torch.device(self.device))['model_state_dict']) # load mô hình đã cho huấn luyện # 10 tuổi
        
        self.nb_classes = nb_classes
        self.load_height = load_height
        self.load_width = load_width
        self.labels = {0:'Cat', 1: 'Dog'}

        self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((self.load_height, self.load_width)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # hàm chỉnh sửa ảnh về dạng khác

        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 
        # self.device = torch.device('cpu')
        self.model.eval() # chuyển mô hình về dự đoán 
        print("Start ...")
        

    def preprocess(self, path_image):
        img = cv2.imread(path_image) # đọc ảnh
        img  = self.transform(img) # chuyển ảnh về dangh tensor
        return img.to(self.device).unsqueeze(0) # thêm 1 chiều cho ảnh 1,n,h,w thì phải # 1*3*224*224

    def predict(self, path_image):
        input = self.preprocess(path_image) # = 1 ảnh
        # self.model.eval()
        with torch.no_grad(): # tắt đạo hàm
            output = self.model(input) # dự đoán ảnh # [0.3 0.7] [[0.6 0.4]] = [0.6 0.4]
            output = output.softmax(1).to('cpu').numpy() # chuyển kết quả về numpy
            
        score = np.mean(output, axis=0)
        label = np.argmax(score) # lấy vị trí có xác suất cao nhất P(xj|x1x2...)
        return self.labels[label], score[label] # Dog, 0.7
    
def main(args):
    # khởi tạo các biến
    cfg = config[args.num_config]

    CHECKPOINT_DIR = args.checkpoint_dir
    #model
    NAME_MODEL = args.name_model
    NUM_CLASSES = cfg['NUM_CLASSES']

    DEVICE = cfg['DEVICE']

    #ckpt
    NUM_TRAIN = args.num_train
    NUM_CKPT = args.num_ckpt
    #data
    RESIZE = cfg['RESIZE']
    LOAD_WIDTH = cfg['LOAD_WIDTH']
    LOAD_HEIGHT = cfg['LOAD_HEIGHT']
    IMAGE = args.image
    
    model = Model(name_model= NAME_MODEL, nb_classes= NUM_CLASSES, load_height= LOAD_HEIGHT, load_width= LOAD_WIDTH,\
                  checkpoint_dir= CHECKPOINT_DIR, num_train=NUM_TRAIN, num_ckpt= NUM_CKPT, device= DEVICE)
    label, score = model.predict(IMAGE)

    print('path_image: {} \nlabel : {} \nxac xuat: {}'.format(IMAGE, label, score))

    

        
def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_config', type= str, default= 'test')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--name_model', type= str, default='resnet101')
    parser.add_argument('--num_train', type= str, default= '0001')
    parser.add_argument('--num_ckpt', type= str, default= '1')
    parser.add_argument('--image', type= str, default= 'test/tai-anh-cho-dep.webp')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = get_args_parser()

    main(args=args)
