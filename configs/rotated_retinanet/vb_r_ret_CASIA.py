_base_ = 't_custom_r_ret_CASIA.py'

model = dict(
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)
    )
)

# schedule
optimizer = dict(lr=0.008) # default 0.008
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