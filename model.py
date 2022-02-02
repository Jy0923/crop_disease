from tkinter import X
import torch
from torch import nn
from torchvision import models
import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        

    def forward(self, x):
        return x


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.model = models.resnet50(pretrained = True)
        self.model.fc = Identity()
        
    
    def forward(self, x):
        output = self.model(x)
        return output # output size : (b, 2048)


class VITEncoder(nn.Module):
    def __init__(self):
        super(VITEncoder, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained = True)
        self.model.head = Identity() 
        self.out_features = 768

    def forward(self, x):
        output = self.model(x)
        return output # output size : (b, 768)


class SWINEncoder(nn.Module):
    def __init__(self):
        super(SWINEncoder, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained = True)
        self.model.head = Identity()
        self.out_features = 1024


    def forward(self, x):
        output = self.model(x)
        return output # output size : (b, 1024)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
    

class TimeSeriesEncoder(nn.Module):
    def __init__(self, block = BasicBlock, init_weights = True):
        super(TimeSeriesEncoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = self._make_layer(block, 64, 2, (2, 1))
        self.conv3 = self._make_layer(block, 128, 2, (2, 1))
        self.conv4 = self._make_layer(block, 256, 2, (2, 2))
        self.conv5 = self._make_layer(block, 512, 2, (2, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_features = 512
        if init_weights:
            self._initialize_weights()
        

    def forward(self, x):
        x = x.unsqueeze(dim = 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, img_encoder, csv_encoder, init_weights = True):
        super(Model, self).__init__()
        self.img_encoder = img_encoder
        self.csv_encoder = csv_encoder
        in_features = self.img_encoder.out_features + self.csv_encoder.out_features

        self.fc = nn.Linear(in_features, 25)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, img, csv):
        x1 = self.img_encoder(img)
        x2 = self.csv_encoder(csv)

        x = torch.concat([x1, x2], dim = 1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)