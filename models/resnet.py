import torch
import torch.nn as nn
import torch.nn.functional as F
    
# Reference: torch给出的resnet实现
class BasicBlock(nn.Module):
    expansion = 1  # 残差块的输出通道数是否扩展（ResNet-18/34 不扩展）

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        ) # kernel size 是固定的
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.expansion * out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) 
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout_prob = 0.3):
        super().__init__()
        self.in_channels = 64  

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2) 
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个块可能降采样，其余不降
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 28, 28]
        out = self.layer1(out)                 # [B, 64, 28, 28]
        out = self.layer2(out)                 # [B, 128, 14, 14]
        out = self.layer3(out)                 # [B, 256, 7, 7]
        out = self.layer4(out)                 # [B, 512, 4, 4]
        out = self.avg_pool(out)               # [B, 512, 1, 1]
        out = out.view(out.size(0), -1)        # [B, 512]
        out = self.dropout(out)
        out = self.fc(out)                     # [B, num_classes]
        return out
    

class ResNet34(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.3):
        super().__init__()
        self.in_channels = 64  

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)   
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)  
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.layer1(out)                 
        out = self.layer2(out)                 
        out = self.layer3(out)                 
        out = self.layer4(out)                 
        out = self.avg_pool(out)               
        out = out.view(out.size(0), -1)        
        out = self.dropout(out)
        out = self.fc(out)                     
        return out
