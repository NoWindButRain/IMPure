from torchvision.models import resnet101, ResNet101_Weights
from .. import TargetModel


def resnet101_v2(pretrained=True, **kwargs):
    name = 'resnet101_v2'
    weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet101(weights=weights)
    target_model = TargetModel(
        name, model, weights, **kwargs
    )
    return target_model


