import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2 as cv
#imread返回的结果是：【H，W，C】高，宽，通道。RGB图像返回的通道的顺序是BGR
#通道Channel --> 不同类别 --> 不同的特征值 --> FeatureMap
img=cv.imread("C:\\Users\\hp\\Desktop\\image\\R.jpg")
print(type(img),img.shape)
print(img[:,:,0])#蓝色通道 数字表示特征信息 矩阵表示图像信息
Gray=img[:,:,2]*0.3+img[:,:,1]*0.59+img[:,:,0]*0.11
print(Gray.shape)
#Gray=Gray/255
# 把浮点数转成0-255的整数（uint8是OpenCV默认图片类型）
Gray = Gray.astype('uint8')
# 创建可调整尺寸的窗口（避免自动缩放）
cv.namedWindow('Color_Original', cv.WINDOW_NORMAL)  # WINDOW_NORMAL允许调整窗口大小
cv.namedWindow('Gray_Original', cv.WINDOW_NORMAL)
cv.imshow('Color_Original', img)
cv.imshow('Gray_Original', Gray)
cv.waitKey(0)#保持显示，直到键盘有输入
cv.destroyWindow()