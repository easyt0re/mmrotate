_base_ = 'oriented_reppoints_r50_fpn_1x_dota_le135.py'

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
pretrained = 'work_dirs/pretrained_models/swin_tiny_patch4_window7_224.pth'

dataset_type = 'DDDOTADataset'
classes = ('ship', 'submarine')
n_classes = len(classes)

class_weight = [1., 20., 0.5]

# classes = ('ship', 'submarine')
# data_root = 'datasets/split_data/'
# data_root = 'datasets/CASIA-Ship/'
# data_root = 'data/split_ss_dota/'
# data_root = '../mmr/datasets/SRSDD_DOTA/'
data_root = '../mmr/datasets/CASIA-Ship/'

# n_frozen_stages = 1
# n_frozen_epoch = 5

angle_version = 'le135'

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
    dict(
        type='PolyRandomRotate',
        mode='value',
        rotate_ratio=0.5,
        angles_range=[90, 180, -90],
        auto_bound=False,
        rect_classes=None,
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
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        _delete_=True,
        type='PAFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
            num_classes=n_classes)
    )

# custom_imports = dict(imports=['unfreeze_backbone_epoch_based_hook'], allow_failed_imports=False)
# custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=n_frozen_epoch)]

# schedule
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
# learning policy
# lr_config = None
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[200]
    )
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)

#default runtime
# ckpt_name = 'epoch_1.pth' # specify ckpt file name here
# resume_from = './work_dirs/resume_ckpt/' + ckpt_name
workflow = [('train', 1), ('val', 1)]