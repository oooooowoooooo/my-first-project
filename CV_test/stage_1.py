import os
import cv2 as cv
import torch

from PIL import Image

from torchvision import models, transforms


def stage1():

    img_path = "entropy_loss_softmax.png"
    img = cv.imread(img_path)
    output_dir = "C:\\Users\\hp\\Desktop\\DeepBlue\\image" + os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    print(type(img))
    h, w, c = img.shape

    sizes = [(224, 224), (300, 300)]
    k = 0
    step = 50
    for h_size, w_size in sizes:
        for i in range(0, h - h_size, step):
            for j in range(0, w - w_size, step):
                img_ = img[i:i + h_size, j:j + w_size, :]
                print(img_.shape)
                cv.imwrite(os.path.join(output_dir, f'{k:06d}_{h_size}_{w_size}_{i}_{j}.png'), img_)
                k += 1

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == '__main__':
    stage1()