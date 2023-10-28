#/bin/bash

cd ..

export PYTHONPATH=$(pwd):$PYTHONPATH

python data_process/adv_attack.py --model inceptionv3_v1 --attack FGSM --attack_param "{\"eps\": 16}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack PGD --attack_param "{\"eps\": 16, \"ieps\": 4}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack MIFGSM --attack_param "{\"eps\": 16, \"ieps\": 4}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack DIFGSM --attack_param "{\"eps\": 16, \"ieps\": 4, \"decay\":0}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack DIFGSM --attack_param "{\"eps\": 16, \"ieps\": 4, \"decay\":1}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack APGD --attack_param "{\"eps\": 16, \"ieps\": 4, \"loss\":\"ce\"}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack APGD --attack_param "{\"eps\": 16, \"ieps\": 4, \"loss\":\"dlr\"}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack CW2 --attack_param "{\"eps\": 16, \"max_iterations\": 50, \"lr\": 0.01, \"confidence\": 40, \"initial_const\": 0.1}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299
python data_process/adv_attack.py --model inceptionv3_v1 --attack DeepFool2 --attack_param "{\"eps\": 16, \"ieps\": 4}" --attack_path /data/object_class/adverpuri/attack_fix --data_split train --filter_data /data/results/ImageNet_train_inceptionv3_v1_10.json --device "cuda:3" --input_size 299




