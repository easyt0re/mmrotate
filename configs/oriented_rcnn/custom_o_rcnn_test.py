_base_ = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'

dataset_type = 'DDDOTADataset'
classes = ('ore-oil', 'Cell-Container', 'Fishing', 'LawEnforce', 'Dredger', 'Container')
n_classes = len(classes)

# # compute class weight in theory
# n_ore = 18.
# n_cel = 11.
# n_fis = 42.
# n_law = 3.
# n_dre = 22.
# n_con = 275.
# n_bac = 275. # assume n_bac == class has most
# n_total = n_ore + n_cel + n_fis + n_law + n_dre + n_con + n_bac

# w_ore = n_total / n_ore / (n_classes + 1)
# w_cel = n_total / n_cel / (n_classes + 1)
# w_fis = n_total / n_fis / (n_classes + 1)
# w_law = n_total / n_law / (n_classes + 1)
# w_dre = n_total / n_dre / (n_classes + 1)
# w_con = n_total / n_con / (n_classes + 1)
# w_bac = n_total / n_bac / (n_classes + 1)

# class_weight = torch.tensor([w_ore, w_cel, w_fis, w_law, w_dre, w_con, w_bac]) # no label last
# # this calculation should give weights around [5, 8, 2, 30, 4, 0.3, 0.3]
class_weight = [5., 1., 1., 10., 1., 0.5, 0.5] # one time with mAP > 0.4

# classes = ('ship', 'submarine')
# data_root = 'datasets/split_data/'
# data_root = 'datasets/CASIA-Ship/'
# data_root = 'data/split_ss_dota/'
data_root = '../mmr/datasets/SRSDD_DOTA/'

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
        oversample_thr=0.1,
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
    roi_head=dict(
        bbox_head=dict(
            num_classes=n_classes,
            loss_cls=dict(class_weight=class_weight)
            # changed this based on this page
            # https://mmdetection.readthedocs.io/en/v2.9.0/_modules/mmdet/models/losses/cross_entropy_loss.html
        )
    ),
    # backbone=dict(frozen_stages=n_frozen_stages)
)

# custom_imports = dict(imports=['unfreeze_backbone_epoch_based_hook'], allow_failed_imports=False)
# custom_hooks = [dict(type="UnfreezeBackboneEpochBasedHook", unfreeze_epoch=n_frozen_epoch)]

# schedule
optimizer = dict(lr=0.005) # default 0.005
# learning policy
lr_config = None
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[8, 11]
#     )
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=10)

#default runtime
# ckpt_name = 'epoch_1.pth' # specify ckpt file name here
# resume_from = './work_dirs/resume_ckpt/' + ckpt_name
workflow = [('train', 1)]