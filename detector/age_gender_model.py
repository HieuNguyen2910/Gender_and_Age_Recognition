import torch
import torch.nn as nn
from torchvision import models

# class AgeGenderNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2),

#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2),

#             nn.Conv2d(256, 512, 3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(512),
#             nn.MaxPool2d(2),
#         )

#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(0.5)
#         )

#         self.gender_out = nn.Linear(512 * 4 * 4, 1)
#         self.age_out = nn.Linear(512 * 4 * 4, 1)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.fc(x)

#         gender = torch.sigmoid(self.gender_out(x))  
#         age = self.age_out(x)                  
#         return gender, age

# class AgeGenderNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

#         self.backbone = base.features       # Output: [B,1280,7,7]
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.dropout = nn.Dropout(0.3)

#         self.gender_out = nn.Linear(1280, 1)
#         self.age_out = nn.Linear(1280, 1)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)

#         gender = torch.sigmoid(self.gender_out(x))
#         age = self.age_out(x)

#         return gender, age


# =============================================================
# Model: ResNet50 Multi-task
# =============================================================
class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # remove classification layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])    # output: [B, 2048, 1, 1]

        self.dropout = nn.Dropout(0.4)

        self.gender_out = nn.Linear(2048, 1)
        self.age_out = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        gender = torch.sigmoid(self.gender_out(x))
        age = self.age_out(x)

        return gender, age


def load_age_gender_model(path="./weights/age_gender_mobilenetv2.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgeGenderNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
