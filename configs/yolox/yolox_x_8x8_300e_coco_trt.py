_base_ = ["yolox_x_8x8_300e_coco.py", "../_base_/lightweight.py"]

model = dict(type="YOLOXTRT")

# batch_size, num_classes, img_h, img_w
default_shapes = dict(batch_size=1, img_h=640, img_w=640, num_classes=80)

input_shapes = dict(image=["batch_size", 3, "img_h", "img_w"])

output_shapes = dict(
    cls_score_1=["batch_size", "num_classes", "img_h//8", "img_w//8"],
    cls_score_2=["batch_size", "num_classes", "img_h//16", "img_w//16"],
    cls_score_3=["batch_size", "num_classes", "img_h//32", "img_w//32"],
    bbox_pred_1=["batch_size", 4, "img_h//8", "img_w//8"],
    bbox_pred_2=["batch_size", 4, "img_h//16", "img_w//16"],
    bbox_pred_3=["batch_size", 4, "img_h//32", "img_w//32"],
    objectness_1=["batch_size", 1, "img_h//8", "img_w//8"],
    objectness_2=["batch_size", 1, "img_h//16", "img_w//16"],
    objectness_3=["batch_size", 1, "img_h//32", "img_w//32"],
)

dynamic_axes = dict(_delete_=True, image={0: "batch_size", 2: "img_h", 3: "img_w"})

dynamic_input = dict(
    dynamic_input=True,
    min=(1, 3, 224, 224),
    reg=(16, 3, 512, 512),
    max=(32, 3, 1024, 1024),
)
