_base_ = 'Custom.py'

data_root = '../mmr/datasets/SRSDD_DOTA/'

data = dict(
    samples_per_gpu=2,  # batch-size
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'train/labels/',
        img_prefix=data_root + 'train/images/'),

    val=dict(
        ann_file=data_root + 'val/labels/',
        img_prefix=data_root + 'val/images/'),

    test=dict(
        ann_file=data_root + 'test/labels/',
        img_prefix=data_root + 'test/images/'),
)

evaluation = dict(save_best='auto')