# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto')
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3,
    step=[185])
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=2)
