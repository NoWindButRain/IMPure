from torchvision.models import inception_v3, Inception_V3_Weights
from .. import TargetModel


def inceptionv3_v1(pretrained=True, **kwargs):
    name = 'inceptionv3_v1'
    weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = inception_v3(weights=weights)
    target_model = TargetModel(
        name, model, weights, **kwargs
    )
    return target_model
