import torch
import numpy as np
import cv2 as cv
import os
from typing import List,Optional

def load_images(dir_path,new_size=(100,100),class_names:Optional[List[str]]=None):
    class_names= os.listdir(dir_path)
    class_name2id = {cls_name:i for i,cls_name in enumerate(class_names)}
    images_path=[]
    for cls_name in class_names:
        cls_path=os.path.join(dir_path,cls_name)
        if not os.path.exists(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path=os.path.join(cls_path,img_name)
            images_path.append((img_path,cls_name))
    images=[]
    for img_path,cls_name in images_path:
        img=cv.imread(img_path)
        images.append((img,cls_name))
    tensors=[]
    labels=[]
    for img,cls_name in images:
        img=cv.resize(img,new_size)
        #[h,w,c]-->[c,h,w]
        img=np.transpose(img,axes=(2,0,1))
        tensor=torch.tensor(img,dtype=torch.float32)
        tensors.append(tensor)
        cls_id=class_name2id[cls_name]
        labels.append(torch.tensor(cls_id))
    return tensors,labels,class_names







def training():
    train_path=" "
    val_path=" "
    train_images,train_labels,class_names=load_images(train_path)
    val_images,val_labels,_=load_images(val_path,class_names=class_names)