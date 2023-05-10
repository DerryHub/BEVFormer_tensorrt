import torch
from mmdet.models import NECKS

from third_party.bev_mmdet3d import LSSViewTransformer
from third_party.bev_mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2


@NECKS.register_module()
class LSSViewTransformerTRT(LSSViewTransformer):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """
    def __init__(self, *args, **kwargs):
        super(LSSViewTransformerTRT, self).__init__(*args, **kwargs)

    def pre_compute_trt(self, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda):
        if self.initial_flag:
            coor = self.get_lidar_coor(sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_trt(self, depth, tran_feat, x, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda):
        if self.accelerate:
            self.pre_compute_trt(sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
        B, N, C, H, W = x.shape

        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth

    def forward_trt(self, feat, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            inputs (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        B, N, C, H, W = feat.shape
        x = feat.view(B * N, C, H, W)
        x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)

        return self.view_transform_trt(depth, tran_feat, feat, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)
