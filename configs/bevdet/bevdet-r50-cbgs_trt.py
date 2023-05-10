_base_ = ["bevdet-r50-cbgs.py", "../_base_/det2trt.py"]

model = dict(
    type='BEVDetTRT',
    img_view_transformer=dict(type='LSSViewTransformerTRT'),
    pts_bbox_head=dict(type='BEVDetCenterHeadTRT')
)
