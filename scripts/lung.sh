#!/bin/bash

if [ "$1" == "train" ]
then
#	CUDA_VISIBLE_DEVICES=5,6 tools/dist_train.sh configs/lung/resnet50_3d_BN_b8x4_pretrained_6cls.py 2 \
#  --gpus 2

	CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_train.sh configs/lung/resnet50_3d_BN_b8x4_pretrained_6cls.py 4 \
  --gpus 4

#  CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_train.sh configs/lidc/resnet18_3d_BN_b32x8_scratch_LIDC.py 4 \
#  --gpus 4

elif [ "$1" == "test" ]
then
#	  CUDA_VISIBLE_DEVICES=4,5 tools/dist_test.sh configs/lidc/resnet18_3d_BN_b32x8_scratch_LIDC.py \
#		/data2/lianjie/lung_general_data/lidc/models/demo/epoch_1.pth 2 \
#		--out result.json

#		CUDA_VISIBLE_DEVICES=4 python ./tools/test.py \
#		configs/lidc/resnet18_3d_BN_b32x8_scratch_LIDC.py \
#		/data2/lianjie/lung_general_data/lidc/models/demo/epoch_1.pth \
#    --out result.json

	  CUDA_VISIBLE_DEVICES=4,5 tools/dist_test.sh configs/lung/resnet18_3d_BN_b16x4_pretrained.py \
		/data2/lianjie/lung_general_data/lung_batch1234/models/resnet18_3d_BN_b16x4_pretrained/epoch_50.pth 2 \
		--out result.json --metric auc_multi_cls

#	  CUDA_VISIBLE_DEVICES=4 python ./tools/test.py \
#		configs/lung/resnet18_3d_BN_b16x4_pretrained.py \
#		/data2/lianjie/lung_general_data/lung_batch1234/models/resnet18_3d_BN_b16x4_pretrained/epoch_50.pth \
#		--out result.json --metric auc_multi_cls

#		CUDA_VISIBLE_DEVICES=7 python ./tools/test.py \
#		configs/lung/resnet18_3d_BN_b16x4.py \
#		/data2/lianjie/lung_general_data/lung_batch1234/models/resnet18_3d_BN_b16x4_pretrained/epoch_50.pth \
#		--out result.json --metric auc_multi_cls

fi
