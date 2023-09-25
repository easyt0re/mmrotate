_base_ = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'

dataset_type = 'DOTADataset'
classes = ('ore-oil', 'Cell-Container', 'Fishing', 'LawEnforce', 'Dredger', 'Container')
# classes = ('ship', 'submarine')
# data_root = 'datasets/split_data/'
# data_root = 'datasets/CASIA-Ship/'
# data_root = 'data/split_ss_dota/'
data_root = '../mmr/datasets/SRSDD_DOTA/'

n_frozen_stages = 1
n_frozen_epoch = 5

data = dict(
    samples_per_gpu=2,  # batch-size
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
    ),
    backbone=dict(frozen_stages=n_frozen_stages)
)

custom_imports = dict(imports=['unfreeze_backbone_epoch_based_hook'], allow_failed_imports=False)
custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=n_frozen_epoch)]