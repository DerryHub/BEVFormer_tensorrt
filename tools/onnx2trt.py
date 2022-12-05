import os
import argparse
from mmcv import Config
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

import sys

sys.path.append(".")
from lightweight.convert import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("config", help="config file path")
    parser.add_argument("onnx")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--max_workspace_size", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    load_tensorrt_plugin()

    config = Config.fromfile(args.config)
    if hasattr(config, "plugin"):
        import importlib

        if isinstance(config.plugin, list):
            for plu in config.plugin:
                importlib.import_module(plu)
        else:
            importlib.import_module(config.plugin)

    dynamic_input = None
    if config.dynamic_input.dynamic_input:
        dynamic_input = [
            config.dynamic_input.min,
            config.dynamic_input.reg,
            config.dynamic_input.max,
        ]

    output = os.path.split(args.onnx)[1].split(".")[0]
    if args.fp16:
        output += "_fp16"
    output = os.path.join(config.TENSORRT_PATH, output)

    build_engine(
        args.onnx,
        output + ".trt",
        dynamic_input=dynamic_input,
        int8=args.int8,
        fp16=args.fp16,
        max_workspace_size=args.max_workspace_size,
    )


if __name__ == "__main__":
    main()
