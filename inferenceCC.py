import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import matplotlib.cm as CM
import cv2

class MCNN(nn.Module):
    '''
    Implementation of Multi-column CNN for crowd counting
    '''
    def __init__(self,load_weights=False):
        super(MCNN,self).__init__()

        self.branch1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU(inplace=True)
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse=nn.Sequential(nn.Conv2d(30,1,1,padding=0))

        if not load_weights:
            self._initialize_weights()

    def forward(self,img_tensor):
        x1=self.branch1(img_tensor)
        x2=self.branch2(img_tensor)
        x3=self.branch3(img_tensor)
        x=torch.cat((x1,x2,x3),1)
        x=self.fuse(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def preprocessing(image_name):
    img=plt.imread(image_name)
    img = img.astype(np.float32, copy=False)
    ht_1 = int(img.shape[0]/4)*4
    wd_1 = int(img.shape[1]/4)*4
    img = cv2.resize(img,(wd_1,ht_1))
    img=img.transpose((2,0,1))
    img_tensor=torch.tensor(img,dtype=torch.float).unsqueeze(0)
    return img_tensor



def get_heatmap(image_name):
    image = preprocessing(image_name)
    outputs = best_model(image)
    et_dmap = outputs.detach().squeeze().squeeze().cpu().numpy()
    count = np.sum(outputs.detach().cpu().numpy())*4
    return et_dmap, int(count)

if __name__ =='__main__':
    args = sys.argv[1:]
    model_path = args[0]
    list_dir = args[1]
    images = [i for i in os.listdir(list_dir)]
    output_dir = args[2]
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
    os.mkdir(output_dir)
    best_model = MCNN()
    best_model.load_state_dict(torch.load('./models/MCNN.pth', map_location='cpu'))
    for image_name in tqdm.tqdm(images):
        et_dmap, count = get_heatmap(os.path.join(list_dir,image_name))
        plt.figure(figsize=(20,20))
        plt.text(10,30,str(count), style='italic',fontsize=72)
        plt.imshow(et_dmap,cmap=CM.jet)
        plt.savefig(os.path.join(output_dir, image_name))