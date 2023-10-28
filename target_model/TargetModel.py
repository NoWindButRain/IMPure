import torch
from torch import nn
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class TargetModel:
    def __init__(
            self, name: str, model: nn.Module,
            weights=None, preprocess=None, normalize=False,
            return_nodes=None
    ):
        self.name = name
        self.weights = weights
        if weights:
            preprocess = self.weights.transforms()
        self.preprocess = preprocess
        train_nodes, eval_nodes = get_graph_node_names(model)
        self.train_nodes = train_nodes
        self.eval_nodes = eval_nodes
        self.return_nodes = return_nodes
        if return_nodes:
            self.model = create_feature_extractor(model, return_nodes)
        else:
            self.model = model
        self.model.eval()
        self.split_normalize(normalize)

    def split_normalize(self, normalize):
        if not normalize:
            return
        prepro = self.preprocess
        if isinstance(prepro, transforms.Compose):
            norm = transforms.Compose(list(filter(
                lambda x: isinstance(x, transforms.Normalize),
                prepro.transforms
            )))
            rest = transforms.Compose(list(filter(
                lambda x: not isinstance(x, transforms.Normalize),
                prepro.transforms
            )))
        elif prepro.__class__.__name__ == 'ImageClassification':
            norm = transforms.Normalize(prepro.mean, prepro.std)
            rest = transforms.Compose([
                transforms.Resize(
                    prepro.resize_size, prepro.interpolation,
                    antialias=prepro.antialias),
                transforms.CenterCrop(prepro.crop_size),
                transforms.ToTensor()
            ])
        self.preprocess = rest
        self.model = NormalizeModel(self.model, norm)


    def __repr__(self):
        format_string = [
            self.__class__.__name__,
            'Model: ' + str(self.model.__class__.__name__),
        ]
        if self.preprocess:
            format_string.append(
                'Preprocess: ' + str(self.preprocess)
            )
        if self.return_nodes:
            format_string.append(
                'Return_nodes: ' + str(self.return_nodes)
            )
        return '\n'.join(format_string)


class NormalizeModel(nn.Module):
    def __init__(self, model, normalize):
        super(NormalizeModel, self).__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, x):
        x = self.normalize(x)
        x = self.model(x)
        return x