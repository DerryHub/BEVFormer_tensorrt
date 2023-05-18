import torch
from mmdet.models import DETECTORS
from ..functions import bev_pool_v2

from third_party.bev_mmdet3d import BEVDet
from third_party.bev_mmdet3d.core.bbox import bbox3d2result


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):
    def __init__(self, *args, **kwargs):
        super(BEVDetTRT, self).__init__(*args, **kwargs)

    def get_bev_pool_input(self, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda):
        coor = self.img_view_transformer.get_lidar_coor(sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
        ranks_bev, ranks_depth, ranks_feat, order, coor = self.img_view_transformer.voxel_pooling_prepare_v2(coor)
        return ranks_bev, ranks_depth, ranks_feat, order, coor

    def forward_trt(self, image, ranks_bev, ranks_depth, ranks_feat, order, coor):
        ranks_bev, ranks_depth, ranks_feat, order, coor = ranks_bev.int(), ranks_depth.int(), ranks_feat.int(), order.long(), coor.long()

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.img_view_transformer.grid_size[0]) & \
                   (coor[:, 1] >= 0) & (coor[:, 1] < self.img_view_transformer.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.img_view_transformer.grid_size[2])
        kept = kept[order]

        ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[kept].contiguous(), ranks_depth[kept].contiguous(), ranks_feat[kept].contiguous()

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.int)
        kept[1:] = (ranks_bev[1:] != ranks_bev[:-1]).int()
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

        x = self.img_backbone(image.flatten(0, 1))
        x = self.img_neck(x)
        x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
                self.img_view_transformer.D +
                self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)

        depth = depth.contiguous()
        tran_feat = tran_feat.contiguous()

        x = bev_pool_v2(depth, tran_feat,
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head.forward_trt(bev_feat)
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
