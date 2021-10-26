#model_name=resnet18_3d_BN_b32_pretrained_30eps_lesion_0528_sidsampler_0528clrs_depth
model_name=resnet18_3d_BN_b32_pretrained_30eps_lesion_0612ndiagmod_sidsampler_0612clrs_depth_v2
#model_name=resnet18_3d_BN_b32_pretrained_30eps_lesion_0612ndiagmod_sidsampler_0612clrs_depth_v2
model_dir=./work_dirs/
dataset=cthx_mjc
epoch=latest.pth

if [ "$1" == "train" ]
then

    CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_train.sh configs/$dataset/$model_name.py 4 \
    --gpus 4 --work-dir $model_dir/$model_name/
    # --resume-from $model_dir/$model_name/$epoch

    #CUDA_VISIBLE_DEVICES=0 tools/dist_train.sh configs/$dataset/$model_name.py 1 \
    #--gpus 1 --work-dir $model_dir/$model_name/

elif [ "$1" == "trainsid" ]
then

    CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/$dataset/$model_name.py 4 \
    --gpus 4 --work-dir $model_dir/$model_name/ --sid_sampler
    # --resume-from $model_dir/$model_name/$epoch

elif [ "$1" == "test" ]
then

	  CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/$dataset/$model_name.py \
		$model_dir/$model_name/$epoch 4 \
		--out result.json --metric 'auc_multi_cls'

elif [ "$1" == "save" ]
then

     CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/$dataset/$model_name.py \
                $model_dir/$model_name/$epoch 4 \
                --out "./mjc_ct/${model_name}_result.pkl" --metric 'auc_multi_cls' 

elif [ "$1" == "cam" ]
then

     CUDA_VISIBLE_DEVICES=0 tools/dist_test.sh configs/$dataset/$model_name.py \
                $model_dir/$model_name/$epoch 1 \
                --out result.json \
                --cam2jpg './cam3d' \
                --metric 'auc_multi_cls' \
                --launcher 'none'

fi

