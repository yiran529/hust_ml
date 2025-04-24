1. 构建training pipeline
2. 用简单的MLP验证pipeline编写的正确性
```python
# 1. 定义模型结构
class SimpleMLP(nn.Module):
    def __init__(self, img_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),                     # 28x28 -> 784
            nn.Linear(img_size*img_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)                # 输出 10 类
        )

    def forward(self, x):
        return self.layers(x)
```
3. 验证发现SimpleMLP大概没有问题 正确率 约87% 在012
4. 仿照torch实现了resnet
5. 初始化 ？
6. resnet第一次尝试(初始学习率为1e-3) 正确率94.45 在014
7. 试图使用一些技巧增强模型性能：
    - 使用更多数据增强
    - 增加一个dropout
    - 增加batch size
    - 采用余弦调度器
8. 分析预测错误的图片
9. 尝试ResNet34 batch=64 只用基础的数据增强 也不行
10. 调整split rate  +   修改norm的值  --> 94.47

-----
11. bsz 64 + resnet18 目前效果最好的-->94.96
12. bsz 64 + resnet34 94.87 但训练损失比较小