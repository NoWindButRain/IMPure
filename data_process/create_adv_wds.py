import io
import os
import sys
import json
import math
import random
import datetime
from multiprocessing import Process
from torchvision import datasets
from dataset import AdvPurDataset
from torchvision.datasets.folder import ImageFolder
from webdataset import TarWriter
from  PIL import Image
from torchvision import transforms


def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))

    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, _samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, _samples, map_func, kwargs)



def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    stream = TarWriter(fname, **kwargs)
    size = 0
    try:
        for item in samples:
            size += stream.write(map_func(item))
    except Exception as e:
        print(e, flush=True)
    stream.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}")
    return size


if __name__ == "__main__":
    preprocesses = {
        'resnet101_v2': transforms.Compose([
            transforms.Resize([232]),
            transforms.CenterCrop([224]),
        ]),
        'inceptionv3_v1': transforms.Compose([
            transforms.Resize([342], interpolation=2),
            transforms.CenterCrop([299]),
        ]),
        'inceptionresnetv2_v1': transforms.Compose([
            transforms.Resize([333], interpolation=3),
            transforms.CenterCrop([299]),
        ]),
    }
    filter_data_root_path = '/root/workspace/AdverPurifier/results/filter_data_fix'
    filter_data_names = {
        'resnet101_v2': {
            'train': 'ImageNet_train_resnet101_v2_10_20231005222233.json',
            'val': 'ImageNet_val_resnet101_v2_5_20231005222233.json'
        },
        'inceptionv3_v1': {
            'train': 'ImageNet_train_inceptionv3_v1_10_20231005222236.json',
            'val': 'ImageNet_val_inceptionv3_v1_5_20231005222236.json'
        },
        'inceptionresnetv2_v1': {
            'train': 'ImageNet_train_inceptionresnetv2_v1_10_20231005222239.json',
            'val': 'ImageNet_val_inceptionresnetv2_v1_5_20231005222239.json'
        }
    }
    adv_format_names = [
        '{0}_raw',
        '{0}_FGSM_{1}',
        '{0}_PGD_{1}_10_{2}',
        '{0}_MIFGSM_{1}_10_{2}',
        '{0}_DIFGSM_{1}_10_{2}_0',
        '{0}_DIFGSM_{1}_10_{2}_1',
        '{0}_CW2_{1}_5_40_0.01',
        '{0}_DeepFool_{1}_50_0.02',
        '{0}_APGD_{1}_10_{2}_ce',
        '{0}_APGD_{1}_10_{2}_dlr'
    ]
    target_model_name = 'inceptionv3_v1'
    eps = 16
    i_eps = 4
    preprocess = preprocesses[target_model_name]
    adv_names = [n.format(target_model_name, eps, i_eps) for n in adv_format_names]
    for adv_name in adv_names:
        if not adv_name:
            continue
        adv_img = 'raw' not in adv_name
        root = "/data/object_class/ILSVRC2012"
        save_path = '/data/object_class/adverpuri/attack_fix_wds/{}'.format(adv_name)
        if adv_img:
            adv_path = '/data/object_class/adverpuri/attack_fix/{}'.format(adv_name)
        else:
            adv_path = root
        split = 'train'
        items = []
        filter_data_name = filter_data_names[target_model_name][split]
        filter_data_path = f'{filter_data_root_path}/{filter_data_name}'
        with open(filter_data_path, 'r') as file:
            filter_data = json.load(file)
        filter_data_dict = dict(filter_data)
        if adv_img:
            def valid_file(path):
                return path.replace(adv_path, '.')[: -4] in filter_data_dict
        else:
            def valid_file(path):
                return path.replace(adv_path, '.') in filter_data_dict
        dataset = AdvPurDataset(
            adv_path, clean_root=root, split=split, is_valid_file=valid_file,
            transform=None, loader=lambda x:x, adv_img=adv_img
        )
        for i in range(len(dataset)):
            item = {
                'adv': dataset[i][0],
                'clean': dataset[i][1],
                'label': dataset[i][2],
                'adv_img': adv_img,
                'preprocess': preprocess
            }
            items.append(item)
        print(dataset[0],os.path.splitext(os.path.basename(dataset[0][0]))[0])
        print(len(dataset))

        def map_func(item):
            adv_name, clean_name, class_idx, adv_img, preprocess = \
                item['adv'], item['clean'], item['label'], item['adv_img'], item['preprocess']
            clean_img = Image.open(clean_name)
            clean_img = preprocess(clean_img)
            img_byte = io.BytesIO()
            clean_img.save(img_byte, format='PNG')
            clean_image = img_byte.getvalue()
            if adv_img:
                with open(os.path.join(adv_name), "rb") as stream:
                    adv_image = stream.read()
            else:
                adv_image = clean_image
            # with open(os.path.join(clean_name), "rb") as stream:
            #     clean_image = stream.read()
            sample = {
                "__key__": os.path.splitext(os.path.basename(clean_name))[0],
                "adv.png": adv_image,
                "clean.jpg": clean_image,
                "cls": str(class_idx).encode("ascii")
            }
            return sample

        os.makedirs(os.path.join(save_path, split), exist_ok=True)

        make_wds_shards(
            pattern=os.path.join(save_path, split, '%06d.tar'),
            num_shards=math.ceil(len(dataset) // 100), # 设置分片数量
            num_workers=8, # 设置创建wds数据集的进程数
            samples=items,
            map_func=map_func,
        )

