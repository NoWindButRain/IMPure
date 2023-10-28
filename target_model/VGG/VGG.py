from types import MethodType

from torchvision.models import vgg19_bn, VGG19_BN_Weights
from .. import TargetModel


def vgg19bn_v1(pretrained=True, classifier=True, **kwargs):
    name = 'vgg19bn_v1'
    weights = VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg19_bn(weights=weights)
    if not classifier:
        model.forward = MethodType(lambda self, x: self.features(x), model)
    target_model = TargetModel(
        name, model, weights, **kwargs
    )
    return target_model