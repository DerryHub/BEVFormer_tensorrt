from mmdet.models import DETECTORS
from third_party.bev_mmdet3d import BEVDet
from third_party.bev_mmdet3d.core.bbox import bbox3d2result


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):
    def __init__(self, *args, **kwargs):
        super(BEVDetTRT, self).__init__(*args, **kwargs)

    def extract_img_feat_trt(self, image, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda):
        """Extract features of images."""
        x, _ = self.image_encoder(image)
        x, depth = self.img_view_transformer.forward_trt(x, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
        x = self.bev_encoder(x)
        return x

    def forward_trt(self, image, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda):
        img_feats = self.extract_img_feat_trt(image, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
        outs = self.pts_bbox_head.forward_trt(img_feats)
        reg, height, dim, rot, vel, heatmap = outs['reg'], outs['height'], outs['dim'], outs['rot'], outs['vel'], outs['heatmap']

        return reg, height, dim, rot, vel, heatmap

    def post_process(self, reg, height, dim, rot, vel, heatmap, img_metas):
        out_dict = [[dict(
            reg=reg, height=height, dim=dim, rot=rot, vel=vel, heatmap=heatmap
        )]]
        result_list = self.pts_bbox_head.get_bboxes(out_dict, img_metas, rescale=True)

        return [
            {
                "pts_bbox": bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in result_list
            }
        ]
