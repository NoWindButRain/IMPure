#/bin/bash

cd ..

export PYTHONPATH=$(pwd):$PYTHONPATH

python data_process/adv_attack.py --model resnet101_v2 --attack FGSM --attack_param "{\"eps\": 16}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack PGD --attack_param "{\"eps\": 16, \"ieps\": 4}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack MIFGSM --attack_param "{\"eps\": 16, \"ieps\": 4}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack DIFGSM --attack_param "{\"eps\": 16, \"ieps\": 4, \"decay\":0}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack DIFGSM --attack_param "{\"eps\": 16, \"ieps\": 4, \"decay\":1}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack APGD --attack_param "{\"eps\": 16, \"ieps\": 4, \"loss\":\"ce\"}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack APGD --attack_param "{\"eps\": 16, \"ieps\": 4, \"loss\":\"dlr\"}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack CW2 --attack_param "{\"eps\": 16, \"max_iterations\": 50, \"lr\": 0.01, \"confidence\": 40, \"initial_const\": 0.1}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224
python data_process/adv_attack.py --model resnet101_v2 --attack DeepFool2 --attack_param "{\"eps\": 16, \"ieps\": 4}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_resnet101_v2_10.json --device "cuda:2" --input_size 224




