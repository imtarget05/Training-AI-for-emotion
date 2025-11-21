import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, architecture='resnet50', pretrained=False):
        super(ResNetEncoder, self).__init__()
        self.architecture = architecture

        # Chọn kiến trúc ResNet
        if architecture == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif architecture == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif architecture == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Bỏ FC cuối, giữ lại feature map [B, C, 1, 1]
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        features = self.resnet(x)                
        features = torch.flatten(features, 1)  
        return features


class FinetuneClassifier(nn.Module):
    def __init__(self, encoder, num_classes=7):
        super(FinetuneClassifier, self).__init__()
        self.encoder = encoder
        feature_dim = encoder.feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
