fp16 = dict(loss_scale=512.)
# model settings
model = dict(
    type='ImageClassifier',
    # pretrained='/lung_general_data/pretrained_model/imagenet_pretrain3d/res3d18_imagenet_BN_final_73.43.pth',
    pretrained=None,
    backbone=dict(
        type='ResNet3D',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        depth_stride=True,      # consistent with ACS
        stem_stride=True,
        strides=(1, 2, 2, 2),   # consistent with ACS
        in_channels=1,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        # norm_cfg=dict(type='BN3d'),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling', use_3d_gap=True),
    head=dict(
        type='LinearClsHead',
        num_classes=9,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # topk=(1, 2),
        multi_cls=True,
    ))
# dataset settings
dataset_type = 'LungDataset'

train_pipeline = [
    # 1. random crop 2. rotation 3. reflection(flip by 3 axis)
    # If use transpose, transpose (d, h, w) to (h, w, d)
    dict(type='LoadTensorFromFile', data_keys='data', transpose=False),
    #dict(type='TensorNormCrop', crop_size=(224, 224, 64), move=16, train=True),
    #dict(type='RandomResizedCrop', size=224),
    #dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadTensorFromFile', data_keys='data', transpose=False),
    #dict(type='TensorNormCrop', crop_size=(224, 224, 64), move=16, train=True),
    #dict(type='Resize', size=(256, -1)),
    #dict(type='CenterCrop', crop_size=224),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

root_dir = '/data2/lianjie/lung_general_data/lung_batch1234/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=root_dir + 'image_npz/',
        ann_file= root_dir + 'sub_dir_list_train.txt',
        #sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=root_dir + 'image_npz/',
        ann_file=root_dir + 'sub_dir_list_val.txt',
        #sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=root_dir + 'image_npz/',
        ann_file=root_dir + 'sub_dir_list_test.txt',
        #sub_set='info/lidc_nodule_info_new_with_subset.csv',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='auc_multi_cls')

# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 40])
runner = dict(type='EpochBasedRunner', max_epochs=50)

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# resume_from = '/data2/lianjie/lung_general_data/lung_batch1234/models/try/epoch_10.pth'
workflow = [('train', 1)]
work_dir = '/data2/lianjie/lung_general_data/lung_batch1234/models/resnet50_3d_GN_b4x4/'
# work_dir = '/data2/lianjie/lung_general_data/lung_batch1234/models/demo/'
