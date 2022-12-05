# BEVFormer_tensorrt



## Install

### Clone

```shell
git clone git@github.com:DerryHub/BEVFormer_tensorrt.git
cd BEVFormer_tensorrt
PROJECT_DIR=$(pwd)
```

### Data Preparation

#### MS COCO (For 2D Detection)

Download the [COCO 2017](https://cocodataset.org/#download) datasets to `/path/to/coco` and unzip them.

```shell
cd ${PROJECT_DIR}/data
ln -s /path/to/coco coco
```

#### NuScenes and CAN bus (For BEVFormer)

Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download) as `/path/to/nuscenes` and `/path/to/can_bus`.

Prepare nuscenes data like [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md).

```shell
cd ${PROJECT_DIR}/data
ln -s /path/to/nuscenes nuscenes
ln -s /path/to/can_bus can_bus

cd ${PROJECT_DIR}
sh samples/bevformer/create_data.sh
```

#### Tree

```shell
${PROJECT_DIR}/data/.
├── can_bus
│   ├── scene-0001_meta.json
│   ├── scene-0001_ms_imu.json
│   ├── scene-0001_pose.json
│   └── ...
├── coco
│   ├── annotations
│   ├── test2017
│   ├── train2017
│   └── val2017
└── nuscenes
    ├── maps
    ├── samples
    ├── sweeps
    └── v1.0-trainval
```

### Install Packages

#### CUDA/cuDNN/TensorRT

Download and install the `CUDA-11.6/cuDNN-8.6.0/TensorRT-8.5.1.7` following [NVIDIA](https://www.nvidia.com/en-us/).

#### PyTorch

Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/get-started/locally/).

```shell
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### MMCV-full

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.5.0
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e .
```

#### MMDetection

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.25.1
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

#### MMDeploy

```shell
git clone git@github.com:open-mmlab/mmdeploy.git
cd mmdeploy
git checkout v0.10.0

git clone git@github.com:NVIDIA/cub.git third_party/cub
cd third_party/cub
git checkout c3cceac115

# go back to third_party directory and git clone pybind11
cd ..
git clone git@github.com:pybind/pybind11.git pybind11
cd pybind11
git checkout 70a58c5
```

##### Build TensorRT Plugins of MMDeploy

**Make sure cmake version >= 3.14.0 and gcc version >= 7.**

```shell
export MMDEPLOY_DIR=/the/root/path/of/MMDeploy
export TENSORRT_DIR=/the/path/of/tensorrt
export CUDNN_DIR=/the/path/of/cuda

export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

cd ${MMDEPLOY_DIR}
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
make -j$(nproc) 
make install
```

##### Install MMDeploy

```shell
cd ${MMDEPLOY_DIR}
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

#### Install this Project

##### Build and Install Custom TensorRT Plugins

```shell
cd ${PROJECT_DIR}/TensorRT/build
cmake .. -DCMAKE_TENSORRT_PATH=/path/to/TensorRT
make -j$(nproc)
make install
```

**Run Unit Test of  Custom TensorRT Plugins**

```shell
cd ${PROJECT_DIR}
sh samples/test_trt_ops.sh
```

##### Build and Install Part of Ops in MMDetection3D

```shell
cd ${PROJECT_DIR}/third_party/bevformer
python setup.py build develop
```

## Benchmarks

### BEVFormer

#### BEVFormer PyTorch

|      Model      |   Data   | Batch Size |          NDS/mAP          | FPS  | Size (MB) | Memory (MB) |  Device  |
| :-------------: | :------: | :--------: | :-----------------------: | :--: | :-------: | :---------: | :------: |
| BEVFormer tiny  | NuScenes |     1      | NDS: 0.354<br/>mAP: 0.252 | 15.9 |    383    |    2167     | RTX 3090 |
| BEVFormer small | NuScenes |     1      | NDS: 0.478<br/>mAP: 0.370 | 5.1  |    680    |    3147     | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | NDS: 0.517<br/>mAP: 0.416 | 2.4  |    265    |    5435     | RTX 3090 |

#### BEVFormer TensorRT with MMDeploy Plugins (Only Support FP32)

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

**\*** `Out of Memory` when onnx2trt with TensorRT-8.5.1.7, but they convert successfully with TensorRT-8.4.3.1. So the version of these engines is TensorRT-8.4.3.1.

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

## TensorRT Plugins

Support TensorRT Ops in BEVFormer: `Grid Sampler`, `Multi-scale Deformable Attention`, `Modulated Deformable Conv2d` and `Rotate`.

Each operation is implemented as 2 versions: **FP32/FP16 (nv_half)** and **FP32/FP16 (nv_half2)**.

For specific speed comparison, see [Custom TensorRT Plugins](./TensorRT/).

## Run

