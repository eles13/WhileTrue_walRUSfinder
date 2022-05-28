import tqdm
import sys
import os
import torch

if __name__ == '__main__':
    args = sys.argv[1:]
    imgdir = args[1]
    images = [i for i in os.listdir(imgdir)]
    path_model = args[0]
    outdir = args[2]
    if os.path.exists(outdir):
        os.rmdir(outdir)
    os.mkdir(outdir)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_model)
    for img in tqdm.tqdm(images):
        out = model(os.path.join(imgdir, img))
        df = out.pandas().xyxy[0]
        df['x_center'] = (df['xmax'] + df['xmin']) / 2
        df['y_center'] = (df['ymax'] + df['ymin']) / 2
        df.to_csv(os.path.join(outdir, img[:-3]+'csv'), index=False, columns = ['x_center', 'y_center'])
    