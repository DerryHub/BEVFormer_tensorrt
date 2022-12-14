from .grid_sampler import grid_sampler, grid_sampler2
from .multi_scale_deformable_attn import (
    multi_scale_deformable_attn,
    multi_scale_deformable_attn2,
)
from .modulated_deformable_conv2d import (
    modulated_deformable_conv2d,
    modulated_deformable_conv2d2,
)
from .rotate import rotate, rotate2
from ..utils.register import TRT_FUNCTIONS


TRT_FUNCTIONS.register_module(module=grid_sampler)
TRT_FUNCTIONS.register_module(module=grid_sampler2)

TRT_FUNCTIONS.register_module(module=multi_scale_deformable_attn)
TRT_FUNCTIONS.register_module(module=multi_scale_deformable_attn2)

TRT_FUNCTIONS.register_module(module=modulated_deformable_conv2d)
TRT_FUNCTIONS.register_module(module=modulated_deformable_conv2d2)

TRT_FUNCTIONS.register_module(module=rotate)
TRT_FUNCTIONS.register_module(module=rotate2)
