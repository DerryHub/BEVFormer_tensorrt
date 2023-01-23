# BEVFormer on TensorRT

This repository is a deployment project of [BEVFormer](https://github.com/fundamentalvision/BEVFormer) on [TensorRT](https://developer.nvidia.com/tensorrt), supporting **FP32/FP16/INT8** inference. Meanwhile, in order to improve the inference speed of BEVFormer on TensorRT, this project implements some TensorRT Ops that support [nv_half](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html#group__CUDA__MATH____HALF__ARITHMETIC) and [nv_half2](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html#group__CUDA__MATH____HALF2__ARITHMETIC). With the accuracy almost unaffected, the inference speed of the **BEVFormer base** can be increased by nearly **four times**, the engine size can be reduced by more than **90%**, and the GPU memory usage can be saved by more than **80%**. In addition, the project also supports common 2D object detection models in [MMDetection](https://github.com/open-mmlab/mmdetection), which support **INT8 Quantization** and **TensorRT Deployment** with a small number of code changes.

## Benchmarks

### BEVFormer


#### BEVFormer PyTorch

|                            Model                             |   Data   | Batch Size |          NDS/mAP          | FPS  | Size (MB) | Memory (MB) |  Device  |
| :----------------------------------------------------------: | :------: | :--------: | :-----------------------: | :--: | :-------: | :---------: | :------: |
| BEVFormer tiny<br />[download](https://drive.google.com/file/d/1VJ_WgA9UZZ-DJr5kUFYAFfIC0UF1oCeQ/view?usp=share_link) | NuScenes |     1      | NDS: 0.354<br/>mAP: 0.252 | 15.9 |    383    |    2167     | RTX 3090 |
| BEVFormer small<br />[download](https://drive.google.com/file/d/1n5_Ca2bqkCY3Q19M5U7futKUT1pEyQZ7/view?usp=sharing) | NuScenes |     1      | NDS: 0.478<br/>mAP: 0.370 | 5.1  |    680    |    3147     | RTX 3090 |
| BEVFormer base<br />[download](https://drive.google.com/file/d/1UMN35jPHeJbVK-8P4HhxIZl341KS1bDU/view?usp=share_link) | NuScenes |     1      | NDS: 0.517<br/>mAP: 0.416 | 2.4  |    265    |    5435     | RTX 3090 |

#### BEVFormer TensorRT with MMDeploy Plugins (Only Support FP32)

|         Model         |   Data   | Batch Size | Float/Int | Quantization Method |          NDS/mAP          |            FPS            |        Size (MB)         |        Memory (MB)         |  Device  |
| :-------------------: | :------: | :--------: | :-------: | :-----------------: | :-----------------------: | :-----------------------: | :----------------------: | :------------------------: | :------: |
|    BEVFormer tiny     | NuScenes |     1      |   FP32    |          -          | NDS: 0.354<br/>mAP: 0.252 |           37.9            |           136            |            2159            | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      |   FP16    |          -          | NDS: 0.354<br/>mAP: 0.252 |  69.2<br/>$\uparrow$ 83%  | 74<br/>$\downarrow$ 46%  | 1729<br/>$\downarrow$ 20%  | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.353<br/>mAP: 0.249 | 65.1<br />$\uparrow$ 72%  | 58<br/>$\downarrow$ 57%  | 1737<br/>$\downarrow$ 20%  | RTX 3090 |
|    BEVFormer tiny     | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.353<br/>mAP: 0.249 | 70.7<br />$\uparrow$ 87%  | 54<br/>$\downarrow$ 60%  | 1665<br/>$\downarrow$ 23%  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      |   FP32    |          -          | NDS: 0.478<br/>mAP: 0.370 |            6.6            |           245            |            4663            | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      |   FP16    |          -          | NDS: 0.478<br/>mAP: 0.370 |  12.8<br/>$\uparrow$ 94%  | 126<br/>$\downarrow$ 49% | 3719<br/>$\downarrow$ 20%  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.476<br/>mAP: 0.367 |  8.7<br />$\uparrow$ 32%  | 158<br/>$\downarrow$ 36% | 4079<br/>$\downarrow$ 13%  | RTX 3090 |
|    BEVFormer small    | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.477<br/>mAP: 0.368 | 13.3<br />$\uparrow$ 102% | 106<br/>$\downarrow$ 57% | 3441<br/>$\downarrow$ 26%  | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      |   FP32    |          -          | NDS: 0.517<br/>mAP: 0.416 |            1.5            |           1689           |           13893            | RTX 3090 |
|    BEVFormer base     | NuScenes |     1      |   FP16    |          -          | NDS: 0.517<br/>mAP: 0.416 |  1.8<br/>$\uparrow$ 20%   | 849<br/>$\downarrow$ 50% | 11865<br/>$\downarrow$ 15% | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.516<br/>mAP: 0.414 |  1.8<br/>$\uparrow$ 20%   | 426<br/>$\downarrow$ 75% | 12429<br/>$\downarrow$ 11% | RTX 3090 |
| BEVFormer base **\*** | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.515<br/>mAP: 0.414 |  2.2<br/>$\uparrow$ 47%   | 244<br/>$\downarrow$ 86% | 11011<br/>$\downarrow$ 21% | RTX 3090 |

**\*** `Out of Memory` when onnx2trt with TensorRT-8.5.1.7, but they convert successfully with TensorRT-8.4.3.1. So the version of these engines is TensorRT-8.4.3.1.

#### BEVFormer TensorRT with Custom Plugins (Support nv_half and nv_half2)

**FP16 Plugins with nv_half**

|      Model      |   Data   | Batch Size | Float/Int | Quantization Method |          NDS/mAP          |        FPS/Improve        |        Size (MB)         |        Memory (MB)        |  Device  |
| :-------------: | :------: | :--------: | :-------: | :-----------------: | :-----------------------: | :-----------------------: | :----------------------: | :-----------------------: | :------: |
| BEVFormer tiny  | NuScenes |     1      |   FP32    |          -          | NDS: 0.354<br/>mAP: 0.252 |  41.4<br/>$\uparrow$ 9%   | 135<br/>$\downarrow$ 1%  | 1699<br/>$\downarrow$ 21% | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      |   FP16    |          -          | NDS: 0.354<br/>mAP: 0.252 | 76.8<br/>$\uparrow$ 103%  | 73<br/>$\downarrow$ 46%  | 1203<br/>$\downarrow$ 44% | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.352<br/>mAP: 0.249 | 84.0<br />$\uparrow$ 122% | 57<br/>$\downarrow$ 58%  | 1077<br/>$\downarrow$ 50% | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.353<br/>mAP: 0.250 | 96.1<br />$\uparrow$ 154% | 54<br/>$\downarrow$ 60%  | 1109<br/>$\downarrow$ 49% | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP32    |          -          | NDS: 0.478<br/>mAP: 0.370 |   7.0<br/>$\uparrow$ 6%   | 246<br/>$\downarrow$ 0%  | 2645<br/>$\downarrow$ 43% | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP16    |          -          | NDS: 0.479<br/>mAP: 0.370 | 16.3<br/>$\uparrow$ 147%  | 124<br/>$\downarrow$ 49% | 1789<br/>$\downarrow$ 62% | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.477<br/>mAP: 0.368 | 10.4<br />$\uparrow$ 58%  | 157<br/>$\downarrow$ 36% | 1925<br/>$\downarrow$ 59% | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.477<br/>mAP: 0.368 | 17.8<br />$\uparrow$ 170% | 103<br/>$\downarrow$ 58% | 1627<br/>$\downarrow$ 65% | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP32    |          -          | NDS: 0.516<br/>mAP: 0.416 |  3.2<br/>$\uparrow$ 113%  | 283<br/>$\downarrow$ 83% | 5175<br/>$\downarrow$ 63% | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP16    |          -          | NDS: 0.515<br/>mAP: 0.415 |  6.5<br/>$\uparrow$ 333%  | 144<br/>$\downarrow$ 91% | 3323<br/>$\downarrow$ 76% | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP32/INT8 | PTQ max/per-tensor  | NDS: 0.516<br/>mAP: 0.414 | 4.2<br />$\uparrow$ 180%  | 188<br/>$\downarrow$ 89% | 3139<br/>$\downarrow$ 77% | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.516<br/>mAP: 0.414 | 5.8<br />$\uparrow$ 287%  | 125<br/>$\downarrow$ 93% | 3073<br/>$\downarrow$ 78% | RTX 3090 |

**FP16 Plugins with nv_half2**

|      Model      |   Data   | Batch Size | Float/Int | Quantization Method |          NDS/mAP          |            FPS            |        Size (MB)         |        Memory (MB)        |  Device  |
| :-------------: | :------: | :--------: | :-------: | :-----------------: | :-----------------------: | :-----------------------: | :----------------------: | :-----------------------: | :------: |
| BEVFormer tiny  | NuScenes |     1      |   FP16    |          -          | NDS: 0.354<br/>mAP: 0.251 | 90.7<br/>$\uparrow$ 139%  | 73<br/>$\downarrow$ 46%  | 1211<br/>$\downarrow$ 44% | RTX 3090 |
| BEVFormer tiny  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.353<br/>mAP: 0.250 | 98.4<br />$\uparrow$ 160% | 54<br/>$\downarrow$ 60%  | 1109<br/>$\downarrow$ 49% | RTX 3090 |
| BEVFormer small | NuScenes |     1      |   FP16    |          -          | NDS: 0.478<br/>mAP: 0.370 | 18.2<br/>$\uparrow$ 176%  | 124<br/>$\downarrow$ 49% | 1843<br/>$\downarrow$ 60% | RTX 3090 |
| BEVFormer small | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.477<br/>mAP: 0.368 | 18.4<br />$\uparrow$ 179% | 105<br/>$\downarrow$ 57% | 1629<br/>$\downarrow$ 65% | RTX 3090 |
| BEVFormer base  | NuScenes |     1      |   FP16    |          -          | NDS: 0.515<br/>mAP: 0.415 | 7.3<br />$\uparrow$ 387%  | 144<br/>$\downarrow$ 91% | 3323<br/>$\downarrow$ 76% | RTX 3090 |
| BEVFormer base  | NuScenes |     1      | FP16/INT8 | PTQ max/per-tensor  | NDS: 0.516<br/>mAP: 0.414 | 6.7<br />$\uparrow$ 347%  | 124<br/>$\downarrow$ 93% | 2437<br/>$\downarrow$ 82% | RTX 3090 |

### 2D Detection Models

This project also supports common 2D object detection models in MMDetection with little modification. The following are deployment examples of YOLOx and CenterNet.

#### YOLOx

|                            Model                             | Data | Framework | Batch Size | Float/Int | Quantization Method |                             mAP                              |  FPS  | Size (MB) | Memory (MB) |  Device  |
| :----------------------------------------------------------: | :--: | :-------: | :--------: | :-------: | :-----------------: | :----------------------------------------------------------: | :---: | :-------: | :---------: | :------: |
| YOLOx<br />[download](https://drive.google.com/file/d/10_mRoiLfK1JEIVq2uqtkBLHES9QgoFgm/view?usp=share_link) | COCO |  PyTorch  |     32     |   FP32    |          -          | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.557<br/>mAP_l: 0.667 | 63.1  |    379    |    7617     | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     |   FP32    |          -          | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.556<br/>mAP_l: 0.667 | 71.3  |    546    |    9943     | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     |   FP16    |          -          | mAP: 0.506<br/>mAP_50: 0.685<br/>mAP_75: 0.55<br/>mAP_s: 0.32<br/>mAP_m: 0.556<br/>mAP_l: 0.668 | 296.8 |    192    |    4567     | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     | FP32/INT8 | PTQ max/per-tensor  | mAP: 0.488<br/>mAP_50: 0.671<br/>mAP_75: 0.538<br/>mAP_s: 0.311<br/>mAP_m: 0.538<br/>mAP_l: 0.649 | 556.4 |    99     |    5225     | RTX 3090 |
|                            YOLOx                             | COCO | TensorRT  |     32     | FP16/INT8 | PTQ max/per-tensor  | mAP: 0.479<br/>mAP_50: 0.662<br/>mAP_75: 0.53<br/>mAP_s: 0.307<br/>mAP_m: 0.533<br/>mAP_l: 0.634 | 550.6 |    99     |    5119     | RTX 3090 |

#### CenterNet

|                            Model                             | Data | Framework | Batch Size | Float/Int | Quantization Method |                             mAP                              |  FPS   | Size (MB) | Memory (MB) |  Device  |
| :----------------------------------------------------------: | :--: | :-------: | :--------: | :-------: | :-----------------: | :----------------------------------------------------------: | :----: | :-------: | :---------: | :------: |
| CenterNet<br />[download](https://drive.google.com/file/d/1uZVIpDQEWIgY5-1IivQte-9Kue6k71A9/view?usp=share_link) | COCO |  PyTorch  |     32     |   FP32    |          -          | mAP: 0.299<br/>mAP_50: 0.466<br/>mAP_75: 0.319<br/>mAP_s: 0.106<br/>mAP_m: 0.337<br/>mAP_l: 0.463 | 337.4  |    56     |    5171     | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     |   FP32    |          -          | mAP: 0.299<br/>mAP_50: 0.466<br/>mAP_75: 0.319<br/>mAP_s: 0.106<br/>mAP_m: 0.337<br/>mAP_l: 0.463 | 475.6  |    58     |    8241     | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     |   FP16    |          -          | mAP: 0.297<br/>mAP_50: 0.463<br/>mAP_75: 0.316<br/>mAP_s: 0.106<br/>mAP_m: 0.336<br/>mAP_l: 0.46 | 1247.1 |    29     |    5183     | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     | FP32/INT8 | PTQ max/per-tensor  | mAP: 0.27<br/>mAP_50: 0.426<br/>mAP_75: 0.285<br/>mAP_s: 0.086<br/>mAP_m: 0.299<br/>mAP_l: 0.438 | 1534.0 |    20     |    6549     | RTX 3090 |
|                          CenterNet                           | COCO | TensorRT  |     32     | FP16/INT8 | PTQ max/per-tensor  | mAP: 0.285<br/>mAP_50: 0.448<br/>mAP_75: 0.303<br/>mAP_s: 0.096<br/>mAP_m: 0.319<br/>mAP_l: 0.451 | 1889.0 |    17     |    6453     | RTX 3090 |

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

#### Prepare the Checkpoints

Download above PyTorch checkpoints to `${PROJECT_DIR}/checkpoints/pytorch/`. The ONNX files and TensorRT engines will be saved in `${PROJECT_DIR}/checkpoints/onnx/` and `${PROJECT_DIR}/checkpoints/tensorrt/`.

## Custom TensorRT Plugins

Support Common TensorRT Ops in BEVFormer: `Grid Sampler`, `Multi-scale Deformable Attention`, `Modulated Deformable Conv2d` and `Rotate`.

Each operation is implemented as 2 versions: **FP32/FP16 (nv_half)** and **FP32/FP16 (nv_half2)**.

For specific speed comparison, see [**Custom TensorRT Plugins**](./TensorRT/).

## Run

The following tutorial uses `BEVFormer base` as an example.

* Evaluate with PyTorch

```shell
cd ${PROJECT_DIR}
# defult gpu_id is 0
sh samples/bevformer/base/pth_evaluate.sh -d ${gpu_id}
```

* Evaluate with TensorRT and MMDeploy Plugins

```shell
# convert .pth to .onnx
sh samples/bevformer/base/pth2onnx.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP32)
sh samples/bevformer/base/onnx2trt.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP16)
sh samples/bevformer/base/onnx2trt_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32)
sh samples/bevformer/base/trt_evaluate.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16)
sh samples/bevformer/base/trt_evaluate_fp16.sh -d ${gpu_id}

# Quantization
# calibration and convert .onnx to TensorRT engine (FP32/INT8)
sh samples/bevformer/base/onnx2trt_int8.sh -d ${gpu_id}
# calibration and convert .onnx to TensorRT engine (FP16/INT8)
sh samples/bevformer/base/onnx2trt_int8_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32/INT8)
sh samples/bevformer/base/trt_evaluate_int8.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16/INT8)
sh samples/bevformer/base/trt_evaluate_int8_fp16.sh -d ${gpu_id}

# quantization aware train
# defult gpu_ids is 0,1,2,3,4,5,6,7
sh samples/bevformer/base/quant_aware_train.sh -d ${gpu_ids}
# then following the post training quantization process
```

* Evaluate with TensorRT and Custom Plugins

```shell
# nv_half
# convert .pth to .onnx
sh samples/bevformer/plugin/base/pth2onnx.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP32)
sh samples/bevformer/plugin/base/onnx2trt.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP16-nv_half)
sh samples/bevformer/plugin/base/onnx2trt_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32)
sh samples/bevformer/plugin/base/trt_evaluate.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half)
sh samples/bevformer/plugin/base/trt_evaluate_fp16.sh -d ${gpu_id}

# nv_half2
# convert .pth to .onnx
sh samples/bevformer/plugin/base/pth2onnx_2.sh -d ${gpu_id}
# convert .onnx to TensorRT engine (FP16-nv_half2)
sh samples/bevformer/plugin/base/onnx2trt_fp16_2.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half2)
sh samples/bevformer/plugin/base/trt_evaluate_fp16_2.sh -d ${gpu_id}

# Quantization
# nv_half
# calibration and convert .onnx to TensorRT engine (FP32/INT8)
sh samples/bevformer/plugin/base/onnx2trt_int8.sh -d ${gpu_id}
# calibration and convert .onnx to TensorRT engine (FP16-nv_half/INT8)
sh samples/bevformer/plugin/base/onnx2trt_int8_fp16.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP32/INT8)
sh samples/bevformer/plugin/base/trt_evaluate_int8.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half/INT8)
sh samples/bevformer/plugin/base/trt_evaluate_int8_fp16.sh -d ${gpu_id}

# nv_half2
# calibration and convert .onnx to TensorRT engine (FP16-nv_half2/INT8)
sh samples/bevformer/plugin/base/onnx2trt_int8_fp16_2.sh -d ${gpu_id}
# evaluate with TensorRT engine (FP16-nv_half2/INT8)
sh samples/bevformer/plugin/base/trt_evaluate_int8_fp16_2.sh -d ${gpu_id}
```

## Acknowledgement

This project is mainly based on these excellent open source projects:

* [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
* [PyTorch](https://github.com/pytorch/pytorch)
* [MMCV](https://github.com/open-mmlab/mmcv)
* [MMDetection](https://github.com/open-mmlab/mmdetection)
* [MMDeploy](https://github.com/open-mmlab/mmdeploy)

