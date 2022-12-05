# BEVFormer_tensorrt

## Benchmarks

### BEVFormer

#### BEVFormer PyTorch

|      Model      |   Data   | Batch Size |          NDS/mAP          | FPS  | Size (MB) | Memory (MB) |  Device  |
| :-------------: | :------: | :--------: | :-----------------------: | :--: | :-------: | :---------: | :------: |
| BEVFormer tiny  | NuScenes |     1      | NDS: 0.354<br/>mAP: 0.252 | 15.9 |    383    |    2167     | RTX 3090 |
| BEVFormer small | NuScenes |     1      | NDS: 0.478<br/>mAP: 0.370 | 5.1  |    680    |    3147     | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | NDS: 0.517<br/>mAP: 0.416 | 2.4  |    265    |    5435     | RTX 3090 |

#### BEVFormer TensorRT with MMDeploy Plugins

|         Model         |   Data   | Batch Size | Float/Int | Quantization Method |          NDS/mAP          |             FPS             |          Size (MB)          |          Memory (MB)          |  Device  |
| :-------------------: | :------: | :--------: | :-------: | :-----------------: | :-----------------------: | :-------------------------: | :-------------------------: | :---------------------------: | :------: |
|    BEVFormer tiny     | NuScenes |     1      |   FP32    |          -          | NDS: 0.354<br/>mAP: 0.252 |            37.9             |             136             |             2159              | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      |   FP16    |          -          | NDS: 0.354<br/>mAP: 0.252 | 69.2<br/>( $\uparrow$ 83%)  | 74<br/>( $\downarrow$ 46%)  | 1729<br/>( $\downarrow$ 20%)  | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.305<br/>mAP: 0.219 | 72.0<br/>( $\uparrow$ 90%)  | 50<br/>( $\downarrow$ 63%)  | 1745<br/>( $\downarrow$ 19%)  | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.305<br/>mAP: 0.218 | 75.7<br/>( $\uparrow$ 100%) | 50<br/>( $\downarrow$ 63%)  | 1727<br/>( $\downarrow$ 20%)  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      |   FP32    |          -          | NDS: 0.478<br/>mAP: 0.370 |             6.6             |             245             |             4663              | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      |   FP16    |          -          | NDS: 0.478<br/>mAP: 0.370 | 12.8<br/>( $\uparrow$ 94%)  | 126<br/>( $\downarrow$ 49%) | 3719<br/>( $\downarrow$ 20%)  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.471<br/>mAP: 0.364 |  8.7<br/>( $\uparrow$ 32%)  | 150<br/>( $\downarrow$ 39%) | 4195<br/>( $\downarrow$ 10%)  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.470<br/>mAP: 0.364 | 13.2<br/>( $\uparrow$ 100%) | 111<br/>( $\downarrow$ 55%) | 3661<br/>( $\downarrow$ 21%)  | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      |   FP32    |          -          | NDS: 0.517<br/>mAP: 0.416 |             1.5             |            1689             |             13893             | RTX 3090 |
|    BEVFormer base     | NuScenes |     1      |   FP16    |          -          | NDS: 0.517<br/>mAP: 0.416 |  1.8<br/>( $\uparrow$ 20%)  | 849<br/>( $\downarrow$ 50%) | 11865<br/>( $\downarrow$ 15%) | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.512<br/>mAP: 0.410 |  1.7<br/>( $\uparrow$ 13%)  | 1579<br/>( $\downarrow$ 7%) |  14019<br/>( $\uparrow$ 1%)   | RTX 3090 |
|    BEVFormer base     | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  |            ERR            |              -              |              -              |               -               | RTX 3090 |

**\*** TensorRT-8.4.3.1

#### BEVFormer TensorRT with Custom Plugins

**FP16 Plugins with nv_half**

|      Model      |   Data   | Batch Size | Float/Int | Quantization Method |          NDS/mAP          |         FPS/Improve         |          Size (MB)          |         Memory (MB)          |  Device  |
| :-------------: | :------: | :--------: | :-------: | :-----------------: | :-----------------------: | :-------------------------: | :-------------------------: | :--------------------------: | :------: |
| BEVFormer tiny  | NuScenes |     1      |   FP32    |          -          | NDS: 0.354<br/>mAP: 0.252 |  41.4<br/>( $\uparrow$ 9%)  | 135<br/>( $\downarrow$ 1%)  | 1699<br/>( $\downarrow$ 21%) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      |   FP16    |          -          | NDS: 0.354<br/>mAP: 0.252 | 76.8<br/>( $\uparrow$ 103%) | 73<br/>( $\downarrow$ 46%)  | 1203<br/>( $\downarrow$ 44%) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.305<br/>mAP: 0.219 | 78.9<br/>( $\uparrow$ 108%) | 48<br/>( $\downarrow$ 65%)  | 1323<br/>( $\downarrow$ 39%) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.305<br/>mAP: 0.219 | 89.0<br/>( $\uparrow$ 135%) | 48<br/>( $\downarrow$ 65%)  | 1253<br/>( $\downarrow$ 42%) | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP32    |          -          | NDS: 0.478<br/>mAP: 0.370 |  7.0<br/>( $\uparrow$ 6%)   | 246<br/>( $\downarrow$ 0%)  | 2645<br/>( $\downarrow$ 43%) | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP16    |          -          | NDS: 0.479<br/>mAP: 0.370 | 16.3<br/>( $\uparrow$ 147%) | 124<br/>( $\downarrow$ 49%) | 1789<br/>( $\downarrow$ 62%) | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.471<br/>mAP: 0.364 | 10.3<br/>( $\uparrow$ 56%)  | 149<br/>( $\downarrow$ 39%) | 2283<br/>( $\downarrow$ 51%) | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.471<br/>mAP: 0.364 | 16.5<br/>( $\uparrow$ 150%) | 110<br/>( $\downarrow$ 55%) | 2123<br/>( $\downarrow$ 54%) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP32    |          -          | NDS: 0.516<br/>mAP: 0.416 | 3.2<br/>( $\uparrow$ 113%)  | 283<br/>( $\downarrow$ 83%) | 5175<br/>( $\downarrow$ 63%) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP16    |          -          | NDS: 0.515<br/>mAP: 0.415 | 6.5<br/>( $\uparrow$ 333%)  | 144<br/>( $\downarrow$ 91%) | 3323<br/>( $\downarrow$ 76%) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.512<br/>mAP: 0.410 | 4.2<br/>( $\uparrow$ 180%)  | 173<br/>( $\downarrow$ 90%) | 5077<br/>( $\downarrow$ 63%) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.511<br/>mAP: 0.409 | 5.7<br/>( $\uparrow$ 280%)  | 135<br/>( $\downarrow$ 92%) | 4557<br/>( $\downarrow$ 67%) | RTX 3090 |

**FP16 Plugins with nv_half2**

|      Model      |   Data   | Batch Size | Float/Int | Quantization Method |          NDS/mAP          |             FPS              |          Size (MB)          |         Memory (MB)          |  Device  |
| :-------------: | :------: | :--------: | :-------: | :-----------------: | :-----------------------: | :--------------------------: | :-------------------------: | :--------------------------: | :------: |
| BEVFormer tiny  | NuScenes |     1      |   FP16    |          -          | NDS: 0.354<br/>mAP: 0.251 | 90.7<br/>( $\uparrow$ 139%)  | 73<br/>( $\downarrow$ 46%)  | 1211<br/>( $\downarrow$ 44%) | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.305<br/>mAP: 0.218 | 88.7<br/>( $\uparrow$ 134%)  | 48<br/>( $\downarrow$ 65%)  | 1253<br/>( $\downarrow$ 42%) | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP16    |          -          | NDS: 0.478<br/>mAP: 0.370 | 18.2<br/>( $\uparrow$ 176%)  | 124<br/>( $\downarrow$ 49%) | 1843<br/>( $\downarrow$ 60%) | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.471<br/>mAP: 0.364 | 17.5<br />( $\uparrow$ 165%) | 110<br/>( $\downarrow$ 55%) | 2013<br/>( $\downarrow$ 57%) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP16    |          -          | NDS: 0.515<br/>mAP: 0.415 | 7.3<br />( $\uparrow$ 387%)  | 144<br/>( $\downarrow$ 91%) | 3323<br/>( $\downarrow$ 76%) | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.512<br/>mAP: 0.410 | 6.3<br />( $\uparrow$ 320%)  | 116<br/>( $\downarrow$ 93%) | 4543<br/>( $\downarrow$ 67%) | RTX 3090 |

### 2D Detection Models

#### YOLOx

| Model | Data | Framework | Batch Size | Float/Int | Quantization Method |                             mAP                              |  FPS  | Size (MB) | Memory (MB) |  Device  |
| :---: | :--: | :-------: | :--------: | :-------: | :-----------------: | :----------------------------------------------------------: | :---: | :-------: | :---------: | :------: |
| YOLOx | COCO |  PyTorch  |     32     |   FP32    |          -          | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.557<br/>mAP_l: 0.667 | 1158  |    379    |    7617     | RTX 3090 |
| YOLOx | COCO | TensorRT  |     32     |   FP32    |          -          | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.556<br/>mAP_l: 0.667 | 11307 |    546    |    9943     | RTX 3090 |
| YOLOx | COCO | TensorRT  |     32     |   FP16    |          -          | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.556<br/>mAP_l: 0.668 | 29907 |    192    |    4567     | RTX 3090 |
| YOLOx | COCO | TensorRT  |     32     | FP32/INT8 | PTQ max/per-tensor  | mAP: 0.48<br/>mAP_50: 0.673<br/>mAP_75: 0.524<br/>mAP_s: 0.293<br/>mAP_m: 0.524<br/>mAP_l: 0.644 | 24806 |    98     |    3999     | RTX 3090 |
| YOLOx | COCO | TensorRT  |     32     | FP16/INT8 | PTQ max/per-tensor  | mAP: 0.48<br/>mAP_50: 0.673<br/>mAP_75: 0.528<br/>mAP_s: 0.295<br/>mAP_m: 0.523<br/>mAP_l: 0.642 | 25397 |    98     |    3719     | RTX 3090 |

#### CenterNet

|   Model   | Data | Framework | Batch Size | Float/Int | Quantization Method |                             mAP                              |  FPS  | Size (MB) | Memory (MB) |  Device  |
| :-------: | :--: | :-------: | :--------: | :-------: | :-----------------: | :----------------------------------------------------------: | :---: | :-------: | :---------: | :------: |
| CenterNet | COCO |  PyTorch  |     32     |   FP32    |          -          | mAP: 0.295<br/>mAP_50: 0.462<br/>mAP_75: 0.314<br/>mAP_s: 0.102<br/>mAP_m: 0.33<br/>mAP_l: 0.466 | 3271  |           |    5171     | RTX 3090 |
| CenterNet | COCO | TensorRT  |     32     |   FP32    |          -          | mAP: 0.295<br/>mAP_50: 0.461<br/>mAP_75: 0.314<br/>mAP_s: 0.102<br/>mAP_m: 0.33<br/>mAP_l: 0.466 | 15842 |    58     |    8241     | RTX 3090 |
| CenterNet | COCO | TensorRT  |     32     |   FP16    |          -          | mAP: 0.294<br/>mAP_50: 0.46<br/>mAP_75: 0.313<br/>mAP_s: 0.102<br/>mAP_m: 0.329<br/>mAP_l: 0.463 | 16162 |    29     |    5183     | RTX 3090 |
| CenterNet | COCO | TensorRT  |     32     | FP32/INT8 | PTQ max/per-tensor  | mAP: 0.29<br/>mAP_50: 0.456<br/>mAP_75: 0.306<br/>mAP_s: 0.101<br/>mAP_m: 0.324<br/>mAP_l: 0.457 | 14814 |    25     |    4673     | RTX 3090 |
| CenterNet | COCO | TensorRT  |     32     | FP16/INT8 | PTQ max/per-tensor  | mAP: 0.29<br/>mAP_50: 0.456<br/>mAP_75: 0.307<br/>mAP_s: 0.101<br/>mAP_m: 0.325<br/>mAP_l: 0.456 | 16754 |    19     |    4117     | RTX 3090 |