import timm
from .. import TargetModel


def inceptionresnetv2_v1(pretrained=True, **kwargs):
    name = 'inceptionresnetv2_v1'
    model = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=pretrained)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    target_model = TargetModel(
        name, model, preprocess=transform, **kwargs
    )
    return target_model

