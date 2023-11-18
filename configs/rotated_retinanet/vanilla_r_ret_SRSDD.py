_base_ = 'rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'

dataset_type = 'DOTADataset'
classes = ('ore-oil', 'Cell-Container', 'Fishing', 'LawEnforce', 'Dredger', 'Container')
n_classes = len(classes)

data_root = '../mmr/datasets/SRSDD_DOTA/'

data = dict(
    samples_per_gpu=2,  # batch-size
    workers_per_gpu=2,
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
        ann_file=data_root + 'test/labels/',
        img_prefix=data_root + 'test/images/'),
)

model = dict(
    bbox_head=dict(
        num_classes=n_classes
    )
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3,
    step=[100]
    )
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=10)