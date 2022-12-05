# Q&A

## 安装依赖问题

* `ImportError: numpy.core.multiarray failed to import`

  强制安装 numpy==1.20.0

  ```shell
  pip install numpy==1.20.0
  ```

* `ImportError: cannot import name 'gcd' from 'fractions'`

  参考：https://stackoverflow.com/questions/66174862/import-error-cant-import-name-gcd-from-fractions

  强制安装 networkx==2.5

  ```shell
  pip install networkx==2.5
  ```

## 编译问题

* `fatal error: THC/THC.h: No such file or directory`

  由于 PyTorch 1.11 之后的版本移除了 THC 模块，mmdetection3d的编译并未支持该更改。

  修改 mmdetection3d 源码，手动移除使用到 THC 的算子：`mmdet3d/ops`

  形如：

  ```python
  from mmcv.ops import (RoIAlign, SigmoidFocalLoss, get_compiler_version,
                        get_compiling_cuda_version, nms, roi_align,
                        sigmoid_focal_loss)
  
  # from .ball_query import ball_query
  # from .furthest_point_sample import (Points_Sampler, furthest_point_sample,
                                      # furthest_point_sample_with_dist)
  # from .gather_points import gather_points
  # from .group_points import (GroupAll, QueryAndGroup, group_points,
                          #    grouping_operation)
  # from .interpolate import three_interpolate, three_nn
  # from .knn import knn
  from .norm import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d
  from .paconv import PAConv, PAConvCUDA, assign_score_withk
  # from .pointnet_modules import (PAConvCUDASAModule, PAConvCUDASAModuleMSG,
  #                                PAConvSAModule, PAConvSAModuleMSG,
  #                                PointFPModule, PointSAModule, PointSAModuleMSG,
  #                                build_sa_module)
  from .roiaware_pool3d import (RoIAwarePool3d, points_in_boxes_batch,
                                points_in_boxes_cpu, points_in_boxes_gpu)
  # from .sparse_block import (SparseBasicBlock, SparseBottleneck,
  #                            make_sparse_convmodule)
  from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization
  ```

## 运行问题

* 当使用 mmdet 中的模型时额外导入 mmdet3d 的 dataset 模块后会出现如下问题：

  ```shell
  Traceback (most recent call last):
    File "/home/derry/PyCharmCode/Lightweight/tools/2d/evaluate_pth.py", line 92, in <module>
      main()
    File "/home/derry/PyCharmCode/Lightweight/tools/2d/evaluate_pth.py", line 46, in main
      dataset[0]
    File "/home/derry/Code/mmdetection/mmdet/datasets/custom.py", line 218, in __getitem__
      data = self.prepare_train_img(idx)
    File "/home/derry/Code/mmdetection/mmdet/datasets/custom.py", line 241, in prepare_train_img
      return self.pipeline(results)
    File "/home/derry/Code/mmdetection/mmdet/datasets/pipelines/compose.py", line 41, in __call__
      data = t(data)
    File "/home/derry/Code/mmdetection/mmdet/datasets/pipelines/test_time_aug.py", line 107, in __call__
      data = self.transforms(_results)
    File "/home/derry/Code/mmdetection/mmdet/datasets/pipelines/compose.py", line 41, in __call__
      data = t(data)
    File "/home/derry/Code/mmdetection/mmdet/datasets/pipelines/formatting.py", line 343, in __call__
      img_meta[key] = results[key]
  KeyError: 'img_norm_cfg'
  ```

  猜测原因是在 mmdet 的配置中若没有 Normalize 则会默认添加 mean 为 0，std 为 1的 Normalize，而导入 mmdet3d 的 dataset 后，会替换原来的机制，不会默认添加 Normalize。

  解决办法：

  在原来的 mmdet 配置中添加 Normalize：

  ```python
  img_norm_cfg = dict(
      mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
  
  test_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(
          type='MultiScaleFlipAug',
          img_scale=img_scale,
          flip=False,
          transforms=[
              dict(type='Resize', keep_ratio=True),
              dict(type='RandomFlip'),
              dict(type='Normalize', **img_norm_cfg),
              dict(
                  type='Pad',
                  pad_to_square=True,
                  pad_val=dict(img=(114.0, 114.0, 114.0))),
              dict(type='DefaultFormatBundle'),
              dict(type='Collect', keys=['img'])
          ])
  ]
  ```