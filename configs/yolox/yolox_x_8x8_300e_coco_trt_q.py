_base_ = ["yolox_x_8x8_300e_coco_trt.py"]

model = dict(
    backbone=dict(type="CSPDarknetQ", conv_cfg=dict(type="Conv2dQ")),
    neck=dict(type="YOLOXPAFPN", conv_cfg=dict(type="Conv2dQ")),
    bbox_head=dict(type="YOLOXHeadQ", conv_cfg=dict(type="Conv2dQ")),
)

data = dict(
    samples_per_gpu=1,
    quant=dict(
        type={{_base_.dataset_type}},
        ann_file={{_base_.data.train.dataset.ann_file}},
        img_prefix={{_base_.data.train.dataset.img_prefix}},
        pipeline={{_base_.test_pipeline}},
    ),
)

# learning policy
lr_config = dict(_delete_=True, policy="step", step=[10])

total_epochs = 1
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

optimizer = dict(
    type="SGD",
    lr=1e-6,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

load_from = "checkpoints/pytorch/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
