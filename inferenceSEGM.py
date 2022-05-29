import segmentation_models_pytorch as smp
import albumentations as A
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import sys
import os
import tqdm
from resize_functions import resize_with_pad, get_centers_simple, inv_convert_points
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import shutil
import pandas as pd


class WalrusDetectionSEG:
    """
    Model detect walrus
    """

    def __init__(self, path_model='./best_model_LinkNet34.pth', device='cpu', dsize=(1024, 1536)):

        self.model = torch.load(path_model, map_location=device)
        self.model.eval()
        self.device = torch.device(device)
        self.dsize = (1024, 1536)
        self.preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
        
    def preprocess_image(self, img):
        timg = self.preprocess_input(img)
        timg = np.transpose(timg, axes=(2, 0, 1))
        timg = torch.Tensor(timg).to(self.device)
        timg = timg.unsqueeze(0)
        return timg
        
    def get_heatmap(self, img, with_transform=False):
        timg = self.preprocess_image(img)

        with torch.no_grad():
            mask = self.model(timg)[0]
            mask = mask[0]
            mask = mask.detach().cpu().numpy()

        return mask

    def predict(self, image_path):
        """
        I want path of image
        """
        orig_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        resized_img, convertion_info = resize_with_pad(orig_img, (self.dsize[1], self.dsize[0]))
        pred_mask = self.get_heatmap(resized_img, with_transform=False)
        thresholded_pred_mask = ((pred_mask > 0.5) * 255).astype(np.uint8)
        centers = get_centers_simple(thresholded_pred_mask)
        centers = np.array(centers)
        centers = inv_convert_points(centers, convertion_info)
        return orig_img, thresholded_pred_mask, centers
    
    
if __name__ =='__main__':
    args = sys.argv[1:]
    model_path = args[0]
    list_dir = args[1]
    images = [i for i in os.listdir(list_dir)]
    output_dir = args[2]
    output_csv = output_dir + '_csv' if output_dir[-1] != '/' else output_dir[:-1] + '_csv'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(output_csv):
        shutil.rmtree(output_csv)
    os.mkdir(output_dir)
    os.mkdir(output_csv)
    best_model = WalrusDetectionSEG(path_model=model_path)
    for image_name in tqdm.tqdm(images):
        all_centers = []
        orig_img, mask, centers = best_model.predict(os.path.join(list_dir, image_name))
        plt.figure(figsize=(20,20))
        plt.text(10,30, str(len(centers)), style='italic',fontsize=72)
        plt.imshow(orig_img)
        plt.savefig(os.path.join(output_dir, image_name))
        plt.figure(figsize=(20,20))
        plt.text(10,30, str(len(centers)), style='italic',fontsize=72)
        plt.imshow(mask, cmap="gray")
        plt.savefig(os.path.join(output_dir, image_name.split('.')[0] + '_mask' + image_name.split('.')[1]))
        for i in range(len(centers)):
            all_centers.append({'x': centers[i][0], 'y': centers[i][1]})
        all_centers = pd.DataFrame(all_centers)
        all_centers.to_csv(os.path.join(output_csv, f"{image_name.split('.')[0]}.csv"), index=False)
    
            
        