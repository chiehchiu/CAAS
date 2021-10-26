#!/bin/bash
if [ "$1" == "train" ]
then
	#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./tools/dist_train.sh ./configs/imagenet_pretrain/resnet50_3d_BN_b32x8.py 8 \
		#--resume-from /lung_general_data/lizihao/temp/epoch_52.pth
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist_train.sh ./configs/xinan/resnet18_3d_BN_b32x8_scratch_xinancov.py 8
elif [ "$1" == "test" ]
then
	#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh ./configs/lidc/resnet18_3d_GN_b32x8_LIDC.py \
		#./work_dirs/resnet18_3d_b32x8/epoch_100.pth 4 --out result.json
	
	CUDA_VISIBLE_DEVICES=0 python ./tools/test.py \
		./configs/xinan/resnet18_3d_BN_b32x8_scratch_xinancov.py \
		./work_dirs/resnet18_3d_BN_b32x8_scratch_xinancov/epoch_40.pth --out result.json
fi
