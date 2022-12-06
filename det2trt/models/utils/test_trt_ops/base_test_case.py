import pycuda.driver as cuda
import torch
import numpy as np
from ..register import TRT_FUNCTIONS
from .utils import createModel, pth2trt, trt_forward


class BaseTestCase:
    def __init__(
        self,
        model_name,
        input_shapes,
        output_shapes,
        func_name,
        params=None,
        device="cuda",
        delta=1e-5,
    ):
        assert "fp16" in func_name or "fp32" in func_name
        self.p16 = False
        if "fp16" in func_name:
            self.p16 = True
        self.stream = cuda.Stream()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        assert len(output_shapes) == 1
        self.device = device
        self.delta = delta
        if params is None:
            params = [{}]
        assert isinstance(params, list)
        module = TRT_FUNCTIONS.get(model_name)
        self.createInputs()
        self.models = self.createModels(params, module, input_shapes, output_shapes)

    def createModels(self, params, module, input_shapes, output_shapes):
        models = []
        for param in params:
            dic = {"param": param}
            if self.p16:
                model_fp16 = createModel(
                    module,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    fp16=True,
                    **param
                )
                dic["model_pth_fp16"] = model_fp16
            else:
                model_fp32 = createModel(
                    module,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    fp16=False,
                    **param
                )
                dic["model_pth_fp32"] = model_fp32
            models.append(dic)
        return models

    def createInputs(self):
        if self.p16:
            inputs_pth = self.getInputs()
            self.inputs_pth_fp16 = {key: val.half() for key, val in inputs_pth.items()}
            self.inputs_np_fp16 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp16.items()
            }
        else:
            self.inputs_pth_fp32 = self.getInputs()
            self.inputs_np_fp32 = {
                key: val.cpu().numpy() for key, val in self.inputs_pth_fp32.items()
            }

    def getInputs(self):
        torch.random.manual_seed(0)
        torch_type = torch.float32
        inputs = {}
        for key in self.input_shapes:
            inputs[key] = torch.randn(
                *self.input_shapes[key], dtype=torch_type, device=self.device
            )
        return inputs

    def buildEngine(self, opset_version):
        for dic in self.models:
            if self.p16:
                engine = pth2trt(
                    dic["model_pth_fp16"],
                    self.inputs_pth_fp16,
                    opset_version=opset_version,
                    fp16=True,
                )
                dic["engine_fp16"] = engine
            else:
                engine = pth2trt(
                    dic["model_pth_fp32"],
                    self.inputs_pth_fp32,
                    opset_version=opset_version,
                    fp16=False,
                )
                dic["engine_fp32"] = engine

    def engineForward(self, engine, fp16=False):
        if fp16:
            outputs_trt, t = trt_forward(
                engine,
                self.inputs_np_fp16,
                self.input_shapes,
                self.output_shapes,
                self.stream,
            )
        else:
            outputs_trt, t = trt_forward(
                engine,
                self.inputs_np_fp32,
                self.input_shapes,
                self.output_shapes,
                self.stream,
            )
        for val in outputs_trt.values():
            return val, t

    def torchForward(self, model, fp16=False):
        args = []
        if fp16:
            for val in self.inputs_pth_fp16.values():
                args.append(val)
        else:
            for val in self.inputs_pth_fp32.values():
                args.append(val)
        outputs_pth = model(*args)
        return outputs_pth.cpu().numpy()

    @staticmethod
    def getCost(array_1, array_2):
        delta_arr = np.abs(array_1 - array_2).mean()
        return delta_arr
