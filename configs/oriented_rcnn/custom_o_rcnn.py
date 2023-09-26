_base_ = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'

dataset_type = 'DOTADataset'
classes = ('ore-oil', 'Cell-Container', 'Fishing', 'LawEnforce', 'Dredger', 'Container')
class_weight = [1., 1., 1., 10., 1., 1.]
# classes = ('ship', 'submarine')
# data_root = 'datasets/split_data/'
# data_root = 'datasets/CASIA-Ship/'
# data_root = 'data/split_ss_dota/'
data_root = '../mmr/datasets/SRSDD_DOTA/'

# n_frozen_stages = 1
# n_frozen_epoch = 5

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
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/'),
)

model = dict(
    rpn_head=dict(
        loss_cls=dict(class_weight=class_weight)
        # changed this based on this page
        # https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/losses/cross_entropy_loss.html
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
            loss_cls=dict(class_weight=class_weight)
        )
    ),
    # backbone=dict(frozen_stages=n_frozen_stages)
)

# custom_imports = dict(imports=['unfreeze_backbone_epoch_based_hook'], allow_failed_imports=False)
# custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=n_frozen_epoch)]