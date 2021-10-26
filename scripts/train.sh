#!/bin/bash
if [ "$1" == "train" ]
then
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/imagenet/resnet18_b32x8.py 4
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/imagenet_pretrain/resnet18_3d_BN_b32x8.py 4
	#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh ./configs/imagenet_pretrain/resnet50_3d_BN_b32x8.py 8 \
		#--resume-from /lung_general_data/lizihao/temp/epoch_52.pth
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./tools/dist_train.sh ./configs/imagenet_pretrain/resnet18_3d_b32x8.py 4 \
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh ./configs/imagenet_pretrain/resnet18_3d_GN16_b32x8.py 8  --resume-from ./work_dirs/epoch_59.pth
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./configs/imagenet_pretrain/resnet18_3d_GN16_b32x8.py 4  --resume-from ./work_dirs/epoch_59.pth
	#CUDA_VISIBLE_DEVICES=4 python tools/train.py ./configs/lidc/resnet18_3d_BN_b32x8_scratch_LIDC.py
	#CUDA_VISIBLE_DEVICES=4 python tools/train.py ./configs/lidc/resnet18_3d_i3d_BN_b32x8_scratch_LIDC.py 
	#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh ./configs/lidc/resnet18_3d_b32x8_LIDC.py 4
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh ./configs/lidc/resnet18_3d_BN_b32x8_scratch_LIDC.py 4
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh ./configs/lidc/resnet18_3d_acs_BN_b32x8_scratch_LIDC.py 4
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh ./configs/lidc/resnet18_3d_i3d_BN_b32x8_scratch_LIDC.py 4
	#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh ./configs/lidc/resnet18_3d_25d_BN_b32x8_scratch_LIDC.py 4
	#CUDA_VISIBLE_DEVICES=0 python tools/train.py ./configs/xinan/resnet18_3d_BN_b16x4.py
elif [ "$1" == "test" ]
then
	#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh ./configs/lidc/resnet18_3d_GN_b32x8_LIDC.py \
		#./work_dirs/resnet18_3d_b32x8/epoch_100.pth 4 --out result.json
	
	CUDA_VISIBLE_DEVICES=0 python ./tools/test.py \
		./configs/imagenet_pretrain/resnet101_3d_BN_b32x8.py \
		./work_dirs/resnet101_3d_BN_b32x8/epoch_50.pth --out result.json
fi
