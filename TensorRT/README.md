# TensorRT Plugins

## Install

```shell
cd build
cmake .. -DCMAKE_TENSORRT_PATH=/path/to/TensorRT
make -j$(nproc)
make install
```

## Plugins

### Grid Sampler

|     OP Name     |                          Attributes                          |        Inputs         |  Outputs  | FP32 Speed | FP16 Speed | Half Type | Test Device |
| :-------------: | :----------------------------------------------------------: | :-------------------: | :-------: | :--------: | :--------: | :-------: | :---------: |
| GridSamplerTRT  | interpolation_mode: int<br />padding_mode: int<br />align_corners: int | input: T<br />grid: T | output: T |     x1     |    x1.1    |  nv_half  | RTX 2080ti  |
| GridSamplerTRT2 | interpolation_mode: int<br />padding_mode: int<br />align_corners: int | input: T<br />grid: T | output: T |     x1     |    x1.5    | nv_half2  | RTX 2080ti  |

#### Inputs

* input: T[float/half/half2]

  Tensor shape: `[N, C, H_in, W_in]` (4D case) or `[N, C, D_in, H_in, W_in]` (5D case)

* grid: T[float/half/half2]

  Tensor shape: `[N, 2, H_out, W_out]` (4D case) or `[N, 3, D_out, H_out, W_out]` (5D case)

  `grid` specifies the sampling pixel locations normalized by the `input` spatial dimensions. Therefore, it should have most values in the range of ``[-10, 10]``. For example, values ``x = -10, y = -10`` is the left-top pixel of `input`, and values  ``x = 10, y = 10`` is the right-bottom pixel of `input`.

#### Attributes

* interpolation_mode: int

  Interpolation mode to calculate output values. (0: `bilinear` , 1: `nearest`, 2: `bicubic`) 

  Note:  `bicubic` supports only 4-D input.

* padding_mode: int

  Padding mode for outside grid values. (0: `zeros`, 1: `border`, 2: `reflection`)

* align_corners: int

  If `align_corners=1`, the extrema (`-1` and `1`) are considered as referring to the center points of the input's corner pixels. If `align_corners=0`, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic.

#### Outputs

* output: T[float/half/half2]

  Tensor shape: `[N, C, H_out, W_out]` (4D case) or `[N, C, D_out, H_out, W_out]` (5D case)

### Multi-scale Deformable Attention

|           OP Name            | Attributes |                            Inputs                            |  Outputs  | FP32 Speed | FP16 Speed | Half Type | Test Device |
| :--------------------------: | :--------: | :----------------------------------------------------------: | :-------: | :--------: | :--------: | :-------: | :---------: |
| MultiScaleDeformableAttnTRT  |     -      | value: T<br />value_spatial_shapes: T<br />sampling_locations: T<br />attention_weights: T | output: T |     x1     |    x1.4    |  nv_half  | RTX 2080ti  |
| MultiScaleDeformableAttnTRT2 |     -      | value: T<br />value_spatial_shapes: T<br />value_level_start_index: T<br />sampling_locations: T<br />attention_weights: T | output: T |     x1     |    x1.9    | nv_half2  | RTX 2080ti  |

#### Inputs

* value: T[float/half/half2]

  Tensor shape: `[N, num_keys, mum_heads, channel]` 

* value_spatial_shapes: T[int32]

  Spatial shape of each feature map, has shape `[num_levels, 2]`, last dimension 2 represent (h, w)

* sampling_locations: T[float/half/half2]

  The location of sampling points, has shape `[N ,num_queries, num_heads, num_levels, num_points, 2]`, the last dimension 2 represent (x, y).

* attention_weights: T[float/half/half2]

  The weight of sampling points used when calculate the attention, has shape` [N ,num_queries, num_heads, num_levels, num_points]`.

#### Attributes

​	-

#### Outputs

* output: T[float/half/half2]

  Tensor shape: `[N, num_queries, mum_heads*channel]`

### Modulated Deformable Conv2d

|            OP Name            |                          Attributes                          |                            Inputs                            |  Outputs  | FP32 Speed | FP16 Speed | Half Type | Test Device |
| :---------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-------: | :--------: | :--------: | :-------: | :---------: |
| ModulatedDeformableConv2dTRT  | stride: int[2]<br />padding: int[2]<br />dilation: int[2]<br />groups: int<br />deform_groups: int | input: T<br />offset: T<br />mask: T<br />weight: T<br />bias: T (optional) | output: T |     x1     |    x2.2    |  nv_half  | RTX 2080ti  |
| ModulatedDeformableConv2dTRT2 | stride: int[2]<br />padding: int[2]<br />dilation: int[2]<br />groups: int<br />deform_groups: int | input: T<br />offset: T<br />mask: T<br />weight: T<br />bias: T (optional) | output: T |     x1     |    x2.5    | nv_half2  | RTX 2080ti  |

#### Inputs

* input: T[float/half/half2]

  Tensor shape: `[N, C_in, H_in, W_in]` 

* offset: T[float/half/half2]

  Tensor shape: `[N, deform_groups*K_h*K_w*2, H_out, W_out]`

* mask: T[float/half/half2]

  Tensor shape: `[N, deform_groups*K_h*K_w, H_out, W_out]`

* weight: T[float/half/half2]

  Tensor shape: `[C_out, C_in/groups, K_h, K_w]`

* bias: T[float/half/half2] (optional)

  Tensor shape: `[C_out]`

#### Attributes

* stride: int[2]

  Same as torch.nn.Conv2d.

* padding: int[2]

  Same as torch.nn.Conv2d.

* dilation: int[2]

  Same as torch.nn.Conv2d.

* groups: int

  Same as torch.nn.Conv2d.

* deform_groups: int

  Deformable conv2d groups.

#### Outputs

* output: T[float/half/half2]

  Tensor shape: `[N, C_out, H_out, W_out]`

### Rotate

|  OP Name   |     Attributes     |               Inputs                |  Outputs  | FP32 Speed | FP16 Speed | Half Type | Test Device |
| :--------: | :----------------: | :---------------------------------: | :-------: | :--------: | :--------: | :-------: | :---------: |
| RotateTRT  | interpolation: int | img: T<br />angle: T<br />center: T | output: T |     x1     |    X1.6    |  nv_half  | RTX 2080ti  |
| RotateTRT2 | interpolation: int | img: T<br />angle: T<br />center: T | output: T |     x1     |    x1.7    | nv_half2  | RTX 2080ti  |

#### Inputs

* img: T[float/half/half2]

  Tensor shape: `[C, H, W]` 

* angle: T[float/half/half2]

  Tensor shape: `[1]`

* center: T[float/half/half2]

  Tensor shape: `[2]`


#### Attributes

* interpolation: int

  Interpolation mode to calculate output values. (0: `bilinear` , 1: `nearest`) 

#### Outputs

* output: T[float/half/half2]

  Tensor shape: `[C, H, W]`
