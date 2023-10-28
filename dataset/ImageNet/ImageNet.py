import os

from scipy.io import loadmat
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .. import base_wds

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


class ImageNet(ImageFolder):
    def __init__(
        self, root: str, split: str = "train",
        meta_path=None, transform=preprocess,
        **kwargs
    ) -> None:
        root = self.root = os.path.expanduser(root)
        self.name = 'ImageNet'
        self.split = split
        self.split_folder = os.path.join(self.root, self.split)
        meta_path = meta_path if meta_path else root
        idx_to_wnid, wnid_to_classes = parse_meta_mat(meta_path)
        super().__init__(
            self.split_folder, transform=transform,
            **kwargs
        )
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}


def parse_meta_mat(root):
    devkit_root = os.path.join(root, "ILSVRC2012_devkit_t12")
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = loadmat(metafile, squeeze_me=True)['synsets']
    # ILSVRC2012_ID WNID words gloss num_children children wordnet_height num_train_images
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(", ")) for clss in classes]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
    return idx_to_wnid, wnid_to_classes


def ImageNetWDS(
        urls, transform=preprocess, **kwargs
):
    map_key = ["jpg", "cls"]
    map_tran = [transform, lambda x: int(x)]
    dataset = base_wds(
        urls, map_key, map_tran, **kwargs
    )
    dataset.name = 'ImageNetWDS'

    return dataset
