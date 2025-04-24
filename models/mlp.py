import torch
import torch.nn as nn
import functools as F

# 验证training pipeline是否正常工作
class SimpleMLP(nn.Module):
    def __init__(self, img_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),                     # 28x28 -> 784
            nn.Linear(img_size*img_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)                
        )