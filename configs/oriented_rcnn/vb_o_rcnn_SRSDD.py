_base_ = 't_custom_o_rcnn_SRSDD.py'

euqal_weight = [1., 1., 1., 1., 1., 1., 1.]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(class_weight=euqal_weight)
        )
    )
)

# schedule
optimizer = dict(lr=0.005) # default 0.005
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