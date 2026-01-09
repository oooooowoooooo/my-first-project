import torch
import torch.nn as nn
import cv2 as cv
import torch.nn.functional as F
#神经网络本质上是在做特征的提取
class ImageClassifyNetwork(nn.Module):
    def __init__(self,num_classes,in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1,1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1,1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1,1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 1,1)
        self.pool4 = nn.AdaptiveMaxPool2d(8)#自适应池化，将feature map的大小变为8*8
        self.classify=nn.Linear(in_features=64*8*8,out_features=num_classes)
        #在图像中，通道表示特征
        #64个通道表示: 有64个卷积核，每个卷积核对输入的所有通道进行加权合并得到一个feature map
        #64个卷积核得到64个feature map，每个feature map表示一个通道，表示从64个方面对图像的特征描述-》64个特征
        #一个图像加载到内存中，初始状态是RGB三个通道，三个不同角度对图像进行描述
        #神经网络的前几层在提前初步的简单的特征，不需要很多的卷积核
        #随着特征的复杂与组合，用不同的卷积核提取不同的特征
    def forward(self,x):
            #形参x表示输入的批次图像数据，[N,C,H,W],N表示批次大小(每批几个图像)，C表示通道数，H表示高，W表示宽
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(F.relu(self.conv4(x)))
            #扁平化 [batch,64,8,8]-->[batch,64*8*8]
            #flatten(1, -1)的意思是：从第 1 维开始，到最后一维结束，全部展平成一维；
            #展平后第 0 维（batch）保持不变
            x=x.flatten(1,-1)
            #全连接，决策判断，得到每个样本属于各个类别的置信度
            score=self.classify(x)         #[N,num_classes]
            return score
def test():
    net = ImageClassifyNetwork(num_classes=17)  # 初始化分17类的实例对象
    images = torch.randn(4, 3, 100, 100)  # 按照批次图像形状生成正态分布随机数
    scores = net(images)  # 封装了__call__方法，调用forward方法
    print(scores.shape)  # 检查输出维度是否是[4,17]
test()