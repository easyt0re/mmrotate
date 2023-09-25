_base_ = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'

dataset_type = 'DOTADataset'
classes = ('ore-oil', 'Cell-Container', 'Fishing', 'LawEnforce', 'Dredger', 'Container')
# classes = ('ship', 'submarine')
# data_root = 'datasets/split_data/'
# data_root = 'datasets/CASIA-Ship/'
# data_root = 'data/split_ss_dota/'
data_root = '/home/hdd/lixiaohan/mmr/datasets/SRSDD_DOTA/'

data = dict(
    samples_per_gpu=8,  # batch-size
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/labels/',
        img_prefix=data_root + 'train/images/'),

    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/labels/',
        img_prefix=data_root + 'val/images/'),

    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/'),
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes))
    )
)

# copy over from what's from the top
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
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
