_base_ = [
    "centernet_resnet18_dcnv2_140e_coco.py",
    "../_base_/det2trt.py",
]

model = dict(type="CenterNetTRT")

# batch_size, num_classes, img_h, img_w
default_shapes = dict(batch_size=1, img_h=672, img_w=672, num_classes=80)  # 根据模型输入改

input_shapes = dict(image=["batch_size", 3, "img_h", "img_w"])

output_shapes = dict(
    center_heatmap_preds=["batch_size", "num_classes", "img_h//4", "img_w//4"],
    wh_preds=["batch_size", 2, "img_h//4", "img_w//4"],
    offset_preds=["batch_size", 2, "img_h//4", "img_w//4"],
)

dynamic_axes = dict(_delete_=True, image={0: "batch_size", 2: "img_h", 3: "img_w"})

dynamic_input = dict(
    dynamic_input=True,
    min=(1, 3, 336, 336),
    reg=(16, 3, 672, 672),
    max=(32, 3, 1344, 1344),
)
