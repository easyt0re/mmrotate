_base_ = 'rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'

dataset_type = 'DDDOTADataset'
classes = ('ship', 'submarine')
n_classes = len(classes)

data_root = '../mmr/datasets/CASIA-Ship/'

# n_frozen_stages = 1
# n_frozen_epoch = 5

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=2,  # batch-size
    workers_per_gpu=2,
    train=dict(
        type='DDClassBalancedDataset',
        oversample_thr=0.5, # change this base on dataset
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=data_root + 'train/labels/',
            img_prefix=data_root + 'train/images/',
            pipeline=train_pipeline)
            ),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/labels/',
        img_prefix=data_root + 'val/images/'),

    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/labels/',
        img_prefix=data_root + 'test/images/'),
)

model = dict(
    # roi_head=dict(
        bbox_head=dict(
            num_classes=n_classes,
            # loss_cls=dict(class_weight=class_weight)
            # changed this based on this page
            # https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/losses/cross_entropy_loss.html
        )
    # ),
    # backbone=dict(frozen_stages=n_frozen_stages)
)

# custom_imports = dict(imports=['unfreeze_backbone_epoch_based_hook'], allow_failed_imports=False)
# custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=n_frozen_epoch)]

# schedule
optimizer = dict(lr=0.005) # given seems to be 0.02 step and warmup
# learning policy
# lr_config = None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3,
    step=[200]
    )
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)

#default runtime
# ckpt_name = 'epoch_1.pth' # specify ckpt file name here
# resume_from = './work_dirs/resume_ckpt/' + ckpt_name
workflow = [('train', 1), ('val', 1)]

# work_dir = './run_CASIA-Ship/rotated_retinanet'