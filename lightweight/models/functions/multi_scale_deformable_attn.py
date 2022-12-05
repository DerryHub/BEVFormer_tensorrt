import torch
from torch.autograd import Function
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


class _MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def symbolic(g, value, value_spatial_shapes, sampling_locations, attention_weights):
        return g.op(
            "MultiScaleDeformableAttnTRT",
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )

    @staticmethod
    def forward(
        ctx, value, value_spatial_shapes, sampling_locations, attention_weights
    ):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs, num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs, num_queries, num_heads, num_levels, num_points).

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        im2col_step = value.shape[0]
        ctx.im2col_step = im2col_step

        ctx.fp16 = False
        if value.dtype == torch.float16:
            ctx.fp16 = True
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()

        value_level_start_index = torch.zeros_like(value_spatial_shapes[:, 0])
        value_level_start_index[1:] = torch.cumsum(
            value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1], dim=0
        )[:-1]

        output = ext_module.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output.half() if ctx.fp16 else output

    @staticmethod
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors

        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        ext_module.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step,
        )

        if ctx.fp16:
            return (
                grad_value.half(),
                None,
                grad_sampling_loc.half(),
                grad_attn_weight.half(),
            )
        return grad_value, None, grad_sampling_loc, grad_attn_weight


class _MultiScaleDeformableAttnFunction2(_MultiScaleDeformableAttnFunction):
    @staticmethod
    def symbolic(g, value, value_spatial_shapes, sampling_locations, attention_weights):
        return g.op(
            "MultiScaleDeformableAttnTRT2",
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )


_multi_scale_deformable_attn_gpu = _MultiScaleDeformableAttnFunction.apply
_multi_scale_deformable_attn_gpu2 = _MultiScaleDeformableAttnFunction2.apply


def multi_scale_deformable_attn(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Multi-scale deformable attention.

    Support TensorRT plugin MultiScaleDeformableAttnTRT: FP32 and FP16(nv_half).

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    # if value.is_cuda:
    return _multi_scale_deformable_attn_gpu(
        value, value_spatial_shapes, sampling_locations, attention_weights
    )
    # else:
    #     assert torch.onnx.is_in_onnx_export() is False
    #     return multi_scale_deformable_attn_pytorch(
    #         value, value_spatial_shapes, sampling_locations, attention_weights
    #     )


def multi_scale_deformable_attn2(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Multi-scale deformable attention.

    Support TensorRT plugin MultiScaleDeformableAttnTRT2: FP32 and FP16(nv_half2).

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    if value.is_cuda:
        return _multi_scale_deformable_attn_gpu2(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
    else:
        assert torch.onnx.is_in_onnx_export() is False
        return multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
