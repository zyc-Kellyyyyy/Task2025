import torch
from torch import nn

#####Kelly
class AlexNet(nn.Module):
    def __init__(self, num_classes=100):  # 修改为 100 类
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=4),  # → [B, 48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),               # → [B, 48, 27, 27]

            nn.Conv2d(48, 128, kernel_size=3),         # → [B, 128, 25, 25]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),               # → [B, 128, 12, 12]

            nn.Conv2d(128, 192, kernel_size=3),        # → [B, 192, 10, 10]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3),        # → [B, 192, 8, 8]
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)                # → [B, 192, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                              # → [B, 192 * 4 * 4] = [B, 3072]
            nn.Linear(192 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

#####

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#####Kelly
# ✅ 可选测试段落（实际训练时不执行）
if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = AlexNet(num_classes=100)
    y = model(x)
    print(y.shape)  # 应输出: torch.Size([1, 100])
