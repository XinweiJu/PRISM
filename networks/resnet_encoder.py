from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights


class ResNetMultiImageInput(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, channels_per_image=3):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * channels_per_image, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def get_resnet_model_general(num_layers=18, num_input_images=1, channels_per_image=3, pretrained=True):
    assert channels_per_image in [3, 4, 5], "Only supports 3 (RGB), 4 (RGB+Edge), or 5 (RGB+Edge+Lum) channels per image."

    # Get pretrained model
    weights = ResNet18_Weights.DEFAULT if num_layers == 18 else ResNet50_Weights.DEFAULT
    model = models.resnet18(weights=weights) if num_layers == 18 else models.resnet50(weights=weights)

    old_weight = model.conv1.weight  # [64, 3, 7, 7]

    # Basic RGB stacking
    rgb_weight = torch.cat([old_weight] * num_input_images, dim=1) / num_input_images  # [64, 3*num_input_images, 7, 7]

    if channels_per_image == 3:
        new_weight = rgb_weight

    elif channels_per_image in [4, 5]:
        # Edge weight from grayscale mean
        edge_weight = torch.mean(old_weight, dim=1, keepdim=True).repeat(1, num_input_images, 1, 1)  # [64, num_input_images, 7, 7]

        if channels_per_image == 4:
            # [R, G, B, E] per image
            reordered = []
            for i in range(num_input_images):
                reordered.extend([
                    rgb_weight[:, i * 3 + 0:i * 3 + 1, :, :],  # R
                    rgb_weight[:, i * 3 + 1:i * 3 + 2, :, :],  # G
                    rgb_weight[:, i * 3 + 2:i * 3 + 3, :, :],  # B
                    edge_weight[:, i:i + 1, :, :]              # E
                ])
            new_weight = torch.cat(reordered, dim=1)

        elif channels_per_image == 5:
            # Luminance weight: same as edge for initialization
            lum_weight = edge_weight.clone()  # or use different init if available

            # [R, G, B, E, L] per image
            reordered = []
            for i in range(num_input_images):
                reordered.extend([
                    rgb_weight[:, i * 3 + 0:i * 3 + 1, :, :],   # R
                    rgb_weight[:, i * 3 + 1:i * 3 + 2, :, :],   # G
                    rgb_weight[:, i * 3 + 2:i * 3 + 3, :, :],   # B
                    edge_weight[:, i:i + 1, :, :],              # E
                    lum_weight[:, i:i + 1, :, :]                # L
                ])
            new_weight = torch.cat(reordered, dim=1)

    # Replace conv1
    model.conv1 = torch.nn.Conv2d(num_input_images * channels_per_image, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight = torch.nn.Parameter(new_weight)

    return model



def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, channels_per_image=3):
    """Constructs a ResNet model supporting multi-frame and edge input."""
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    
    if pretrained:
        return get_resnet_model_general(num_layers=18, num_input_images=num_input_images, channels_per_image=channels_per_image, pretrained=True)
    else:
        return ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, channels_per_image=channels_per_image)


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder."""
    def __init__(self, num_layers, pretrained, num_input_images=1, channels_per_image=3):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # Always use the modified input version, even for 1 image (to support edge)
        self.encoder = resnet_multiimage_input(
            num_layers, pretrained, num_input_images=num_input_images, channels_per_image=channels_per_image)


        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
