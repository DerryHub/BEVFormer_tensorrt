# Quantization

## 2D Detection

### YOLOx

| Model | Data | Float/Int | Quantization Method | Batch Size |                             mAP                              | Latency of Network (ms) | Size (MB) | Device |
| :---: | :--: | :-------: | :-----------------: | :--------: | :----------------------------------------------------------: | :---------------------: | :-------: | :----: |
| YOLOx | COCO |   FP32    |          -          |     32     | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.556<br/>mAP_l: 0.667 |          1.52           |    568    | 2080ti |
| YOLOx | COCO |   FP16    |          -          |     32     | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.556<br/>mAP_l: 0.668 |          1.09           |    191    | 2080ti |
| YOLOx | COCO | FP16/INT8 | PTQ max/per-tensor  |     32     | mAP: 0.481<br/>mAP_50: 0.673<br/>mAP_75: 0.527<br/>mAP_s: 0.295<br/>mAP_m: 0.524<br/>mAP_l: 0.651 |          1.43           |    102    | 2080ti |


### CenterNet

| Model | Data | Float/Int | Quantization Method | Batch Size |                             mAP                              | Latency of Network (ms) | Size (MB) | Device |
| :---: | :--: | :-------: | :-----------------: | :--------: | :----------------------------------------------------------: | :-------------------: | :-----: | :----: |
| CenterNet | COCO |   FP32    |          -          |     32     | mAP: 0.295<br/>mAP_50: 0.461<br/>mAP_75: 0.314<br/>mAP_s: 0.102<br/>mAP_m: 0.33<br/>mAP_l: 0.466 |         1.17          |   57   | 3060 |
| CenterNet | COCO |   FP16    |          -          |     32     | mAP: 0.294<br/>mAP_50: 0.46<br/>mAP_75: 0.313<br/>mAP_s: 0.102<br/>mAP_m: 0.329<br/>mAP_l: 0.464 |         1.05          |   29   | 3060 |
| CenterNet | COCO | FP16/INT8 | PTQ max/per-tensor  |     32     | mAP: 0.29<br/>mAP_50: 0.456<br/>mAP_75: 0.307<br/>mAP_s: 0.099<br/>mAP_m: 0.325<br/>mAP_l: 0.457 |         1.22          |   24   | 3060 |

## 3D Detection

### BEVFormer

|      Model      |   Data   | Float/Int | Quantization Method | Batch Size |         NDS/mAP         | Latency of Network (ms) | Size (MB) | Device |
| :-------------: | :------: | :-------: | :-----------------: | :--------: | :---------------------: | :---------------------: | :-------: | :----: |
| BEVFormer_tiny  | NuScenes |   FP32    |          -          |     1      | NDS: 0.35<br/>mAP: 0.25 |          54.15          |    161    | 2080ti |
| BEVFormer_tiny  | NuScenes |   FP16    |          -          |     1      | NDS: 0.35<br/>mAP: 0.25 |          27.45          |    70     | 2080ti |
| BEVFormer_tiny  | NuScenes | FP32/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.31<br/>mAP: 0.22 |          27.24          |    119    | 2080ti |
| BEVFormer_tiny  | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.31<br/>mAP: 0.22 |          26.17          |    111    | 2080ti |
| BEVFormer_small | NuScenes |   FP32    |          -          |     1      | NDS: 0.48<br/>mAP: 0.37 |         251.47          |    248    | 2080ti |
| BEVFormer_small | NuScenes |   FP16    |          -          |     1      | NDS: 0.48<br/>mAP: 0.37 |         240.60          |    121    | 2080ti |
| BEVFormer_small | NuScenes | FP32/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.47<br/>mAP: 0.36 |         183.17          |    690    | 2080ti |
| BEVFormer_small | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.47<br/>mAP: 0.36 |          123.9          |    573    | 2080ti |
| BEVFormer_base  | NuScenes |   FP32    |          -          |     1      |                         |                         |           | 2080ti |
| BEVFormer_base  | NuScenes |   FP16    |          -          |     1      |                         |                         |           | 2080ti |
| BEVFormer_base  | NuScenes | FP32/INT8 | PTQ max/per-tensor  |     1      |                         |                         |           | 2080ti |
| BEVFormer_base  | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      |                         |                         |           | 2080ti |

### BEVFormer with Custom Plugins (nv_half)

|      Model      |   Data   | Float/Int | Quantization Method | Batch Size |         NDS/mAP         | Latency of Network (ms) | Size (MB) | Device |
| :-------------: | :------: | :-------: | :-----------------: | :--------: | :---------------------: | :---------------------: | :-------: | :----: |
| BEVFormer_tiny  | NuScenes |   FP32    |          -          |     1      | NDS: 0.35<br/>mAP: 0.25 |          47.96          |    161    | 2080ti |
| BEVFormer_tiny  | NuScenes |   FP16    |          -          |     1      | NDS: 0.35<br/>mAP: 0.25 |          24.11          |    70     | 2080ti |
| BEVFormer_tiny  | NuScenes | FP32/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.31<br/>mAP: 0.22 |          23.64          |    54     | 2080ti |
| BEVFormer_tiny  | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.31<br/>mAP: 0.22 |          22.59          |    52     | 2080ti |
| BEVFormer_small | NuScenes |   FP32    |          -          |     1      | NDS: 0.48<br/>mAP: 0.37 |         231.06          |    247    | 2080ti |
| BEVFormer_small | NuScenes |   FP16    |          -          |     1      | NDS: 0.48<br/>mAP: 0.37 |          99.56          |    122    | 2080ti |
| BEVFormer_small | NuScenes | FP32/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.47<br/>mAP: 0.36 |          149.2          |    155    | 2080ti |
| BEVFormer_small | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.47<br/>mAP: 0.36 |          91.43          |    115    | 2080ti |
| BEVFormer_base  | NuScenes |   FP32    |          -          |     1      | NDS: 0.52<br/>mAP: 0.42 |         524.35          |    290    | 2080ti |
| BEVFormer_base  | NuScenes |   FP16    |          -          |     1      | NDS: 0.52<br/>mAP: 0.42 |          281.5          |    143    | 2080ti |
| BEVFormer_base  | NuScenes | FP32/INT8 | PTQ max/per-tensor  |     1      |                         |                         |           | 2080ti |
| BEVFormer_base  | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      |                         |                         |           | 2080ti |

### BEVFormer with Custom Plugins (nv_half2)

|      Model      |   Data   | Float/Int | Quantization Method | Batch Size |         NDS/mAP         | Latency of Network (ms) | Size (MB) | Device |
| :-------------: | :------: | :-------: | :-----------------: | :--------: | :---------------------: | :---------------------: | :-------: | :----: |
| BEVFormer_tiny  | NuScenes |   FP16    |          -          |     1      | NDS: 0.35<br/>mAP: 0.25 |          21.78          |    70     | 2080ti |
| BEVFormer_tiny  | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.3<br/>mAP: 0.22  |          21.87          |    52     | 2080ti |
| BEVFormer_small | NuScenes |   FP16    |          -          |     1      | NDS: 0.48<br/>mAP: 0.37 |          86.65          |    122    | 2080ti |
| BEVFormer_small | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      | NDS: 0.47<br/>mAP: 0.36 |          91.88          |           | 2080ti |
| BEVFormer_base  | NuScenes |   FP16    |          -          |     1      | NDS: 0.52<br/>mAP: 0.42 |         251.10          |    143    | 2080ti |
| BEVFormer_base  | NuScenes | FP16/INT8 | PTQ max/per-tensor  |     1      |                         |                         |           | 2080ti |
