# Install

## Clone

```shell
git clone git@github.com:DerryHub/Lightweight.git
cd Lightweight
LW_DIR=$(pwd)
```

## Data Preparation

### MS COCO

Download the [COCO 2017](https://cocodataset.org/#download) datasets to `/path/to/coco` and unzip them.

```shell
cd ${LW_DIR}/data
ln -s /path/to/coco coco
```

### NuScenes and CAN bus

Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download) as `/path/to/nuscenes` and `/path/to/can_bus`.

Prepare nuscenes data like [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md).

```shell
cd ${LW_DIR}/data
ln -s /path/to/nuscenes nuscenes
ln -s /path/to/can_bus can_bus

cd ${LW_DIR}
sh samples/bevformer/create_data.sh
```

## Install Packages

### CUDA/cuDNN/TensorRT

Download and install the `CUDA-11.6/cuDNN-8.6.0/TensorRT-8.5.1.7` following [NVIDIA](https://www.nvidia.com/en-us/).

### PyTorch

Install PyTorch and TorchVision following the [official instructions](https://pytorch.org/get-started/locally/).

```shell
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### MMCV-full

```shell
pip install mmcv-full==1.5.0
```

### MMDetection

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.25.1
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### MMDeploy

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

#### Build TensorRT Plugins of MMDeploy

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

#### Install MMDeploy

```shell
cd ${MMDEPLOY_DIR}
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

### LightWeight

#### Build and Install Custom TensorRT Plugins

```shell
cd ${LW_DIR}/TensorRT/build
cmake .. -DCMAKE_TENSORRT_PATH=/path/to/TensorRT
make -j$(nproc)
make install
```

#### Build and Install Part of Ops in MMDetection3D

```shell
cd ${LW_DIR}/third_party/bevformer
python setup.py build develop
```







