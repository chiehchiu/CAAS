#!/bin/bash

#python tools/publish_model.py /lung_general_data/pretrained_model/mr3d/mr3d18_ms640_34.2-8f2bc773.pth \
	#./pretrained_models/res3d18_coco.pth --remove_backbone 
#python tools/publish_model.py work_dirs/resnet18_3d_BN_b32x8/epoch_53.pth ./pretrained_models/res3d18_imagenet_BN.pth --remove_backbone 
python tools/publish_model.py work_dirs/resnet101_3d_BN_b32x8/epoch_100.pth ./pretrained_models/res101_3d_imagenet_BN_final_78.69.pth --remove_backbone 
