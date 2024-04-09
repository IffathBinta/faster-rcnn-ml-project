import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
np.set_printoptions(suppress=True)
import PIL
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch_snippets import Report
import os
import time
import pandas as pd
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import glob
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as fasterrcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm



def collate_fn(batch):
    return tuple(zip(*batch))


col_names=["frame","id","bb_left", "bb_top", "bb_width", "bb_height","conf"]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        # the root folder
        self.root = root

        # transformation for images
        self.transforms = transforms

        self.files = glob.glob("MOT20Det\\train\\MOT20-02\\img1\\*.jpg")

        self.label_df = None
        

        gt_paths = glob.glob("MOT20Det\\train\\MOT20-02\\gt\\*.txt")
        col_names = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'raw_conf', 'label', 'conf',]
        for gt_path in gt_paths:
            tmp_df = pd.read_csv(gt_path, delimiter=',', names=col_names)
            tmp_df['seqname'] = gt_path.split('\\')[2]
        
            if self.label_df is None:
                self.label_df = tmp_df
            else:
                self.label_df = pd.concat((label_df, tmp_df))
                
        self.label_df.loc[self.label_df.label != 1, 'label'] = 0
    
    
    def __getitem__(self,i):
        img_path = self.files[i]
        seqname, frame = img_path.split('\\')[2], int(img_path.split('\\')[4].replace('.jpg', ''))
        # load image
        img = PIL.Image.open(self.files[i]).convert("RGB")
        # load annotations
        ann = self.label_df[(self.label_df.seqname == seqname) & (self.label_df.frame == frame) ]
        # make dictionary of targets
        target = {}
        # boxes, labels, image_id
        x1 = ann['bb_left'].to_numpy()
        x2 = ann['bb_left'].to_numpy() + ann['bb_width'].to_numpy()
        y1 = ann['bb_top'].to_numpy()
        y2 = ann['bb_top'].to_numpy() + ann['bb_height'].to_numpy()
        
        target['boxes'] = torch.as_tensor(np.array([x1,y1,x2,y2]).T, dtype=torch.float32) / 2

        target['labels'] = torch.as_tensor(ann['label'].to_numpy(), dtype=torch.int64)
        
        target['image_id'] = torch.as_tensor(i, dtype=torch.float32)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.files)


        

if __name__ == '__main__':
    batch_size = 16
    transform_train = v2.Compose([
        v2.PILToTensor(),
        v2.Resize((540,960)),
        v2.ToDtype(torch.float32),
        lambda x, y: (x/255, y)
    ])

    device = torch.device("cuda")

    dataset = Dataset("./", transform_train)


    eval_loader = torch.utils.data.DataLoader(dataset, 
                                batch_size = batch_size, 
                                shuffle = False, 
                                collate_fn = collate_fn)
   
    model = fasterrcnn(pretrained = True)

    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                   2)

    model.load_state_dict(torch.load('model\\5.pth'))
    model.to(device)


    with torch.no_grad():
        model.eval()
        for ix, batch in tqdm(enumerate(eval_loader), total=len(dataset)/batch_size):
            X,y = batch
            
            # move batch to device
            X = [x.to(device) for x in X]
            y = [{k: v.to(device) for k, v in t.items()} for t in y]
            
            image_ids = torch.tensor( [[ v for k, v in t.items() if k =='image_id'] 
                        for t in y])
            
            predict = model(X)

            out = {
                'image_ids': image_ids,
                'predict': predict
            }

            torch.save( out, f'predict//{ix}.pth')