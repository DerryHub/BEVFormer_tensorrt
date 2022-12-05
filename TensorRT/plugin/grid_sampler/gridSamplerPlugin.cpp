//
// Created by Derry Lin on 2022/10/22.
//
#include "gridSamplerPlugin.h"
#include "checkMacrosPlugin.h"
#include "gridSamplerKernel.h"
#include "serialize.h"
#include <cuda_fp16.h>
#include <iostream>
#include <stdexcept>

using trt_plugin::GridSamplerPlugin;
using trt_plugin::GridSamplerPluginCreator;
using trt_plugin::GridSamplerPluginCreator2;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr char const *GS_PLUGIN_VERSION{"1"};
constexpr char const *GS_PLUGIN_NAME{"GridSamplerTRT"};
constexpr char const *GS_PLUGIN_NAME2{"GridSamplerTRT2"};
} // namespace

PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<PluginField> GridSamplerPluginCreator::mPluginAttributes;

PluginFieldCollection GridSamplerPluginCreator2::mFC{};
std::vector<PluginField> GridSamplerPluginCreator2::mPluginAttributes;

GridSamplerPlugin::GridSamplerPlugin(int mode, int paddingMode,
                                     bool alignCorners, bool use_h2)
    : mAlignCorners(alignCorners), use_h2(use_h2) {
  ASSERT(mode == 0 || mode == 1 || mode == 2)
  ASSERT(paddingMode == 0 || paddingMode == 1 || paddingMode == 2)
  switch (mode) {
  case 0:
    mMode = GridSamplerInterpolation::Bilinear;
    break;
  case 1:
    mMode = GridSamplerInterpolation::Nearest;
    break;
  case 2:
    mMode = GridSamplerInterpolation::Bicubic;
    break;
  default:
    break;
  }
  switch (paddingMode) {
  case 0:
    mPaddingMode = GridSamplerPadding::Zeros;
    break;
  case 1:
    mPaddingMode = GridSamplerPadding::Border;
    break;
  case 2:
    mPaddingMode = GridSamplerPadding::Reflection;
    break;
  default:
    break;
  }
}

GridSamplerPlugin::GridSamplerPlugin(const void *serialData,
                                     size_t serialLength, bool use_h2)
    : use_h2(use_h2) {
  deserialize_value(&serialData, &serialLength, &mMode);
  deserialize_value(&serialData, &serialLength, &mPaddingMode);
  deserialize_value(&serialData, &serialLength, &mAlignCorners);
}

GridSamplerPlugin::~GridSamplerPlugin() { terminate(); }

int32_t GridSamplerPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs GridSamplerPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
  DimsExprs outputDim;
  outputDim.nbDims = inputs[0].nbDims;
  outputDim.d[0] = inputs[0].d[0];
  outputDim.d[1] = inputs[0].d[1];
  for (int i = 2; i < outputDim.nbDims; i++) {
    outputDim.d[i] = inputs[1].d[i];
  }
  return outputDim;
}

int32_t GridSamplerPlugin::initialize() noexcept { return 0; }

void GridSamplerPlugin::terminate() noexcept {}

size_t
GridSamplerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                    int32_t nbInputs,
                                    const nvinfer1::PluginTensorDesc *outputs,
                                    int32_t nbOutputs) const noexcept {
  return 0;
}

int32_t GridSamplerPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workspace,
                                   cudaStream_t stream) noexcept {
  Dims input_dims = inputDesc[0].dims;
  Dims grid_dims = inputDesc[1].dims;
  Dims output_dims = outputDesc[0].dims;

  auto data_type = inputDesc[0].type;
  ASSERT(data_type == DataType::kFLOAT || data_type == DataType::kHALF)

  switch (data_type) {
  case DataType::kFLOAT:
    grid_sample<float>((float *)outputs[0], (float *)inputs[0],
                       (float *)inputs[1], &(output_dims.d[0]),
                       &(input_dims.d[0]), &(grid_dims.d[0]), input_dims.nbDims,
                       mMode, mPaddingMode, mAlignCorners, stream);
    break;
  case DataType::kHALF:
    if (use_h2) {
      grid_sample<__half2>(
          (__half2 *)outputs[0], (__half2 *)inputs[0], (__half2 *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, mMode, mPaddingMode, mAlignCorners, stream);
    } else {
      grid_sample<__half>(
          (__half *)outputs[0], (__half *)inputs[0], (__half *)inputs[1],
          &(output_dims.d[0]), &(input_dims.d[0]), &(grid_dims.d[0]),
          input_dims.nbDims, mMode, mPaddingMode, mAlignCorners, stream);
    }
    break;
  default:
    return 1;
  }
  return 0;
}

size_t GridSamplerPlugin::getSerializationSize() const noexcept {
  return serialized_size(mMode) + serialized_size(mPaddingMode) +
         serialized_size(mAlignCorners);
}

void GridSamplerPlugin::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, mMode);
  serialize_value(&buffer, mPaddingMode);
  serialize_value(&buffer, mAlignCorners);
}

bool GridSamplerPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  if (pos == 0) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
             inOut[pos].type == nvinfer1::DataType::kHALF) &&
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return inOut[pos].type == inOut[0].type &&
           inOut[pos].format == inOut[0].format;
  }
}

char const *GridSamplerPlugin::getPluginType() const noexcept {
  return use_h2 ? GS_PLUGIN_NAME2 : GS_PLUGIN_NAME;
}

char const *GridSamplerPlugin::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

void GridSamplerPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt *GridSamplerPlugin::clone() const noexcept {
  try {
    int mode, paddingMode;
    switch (mMode) {
    case GridSamplerInterpolation::Bilinear:
      mode = 0;
      break;
    case GridSamplerInterpolation::Nearest:
      mode = 1;
      break;
    case GridSamplerInterpolation::Bicubic:
      mode = 2;
      break;
    default:
      break;
    }
    switch (mPaddingMode) {
    case GridSamplerPadding::Zeros:
      paddingMode = 0;
      break;
    case GridSamplerPadding::Border:
      paddingMode = 1;
      break;
    case GridSamplerPadding::Reflection:
      paddingMode = 2;
      break;
    default:
      break;
    }
    auto *plugin =
        new GridSamplerPlugin{mode, paddingMode, mAlignCorners, use_h2};
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void GridSamplerPlugin::setPluginNamespace(
    const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

char const *GridSamplerPlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace.c_str();
}

DataType
GridSamplerPlugin::getOutputDataType(int32_t index,
                                     const nvinfer1::DataType *inputTypes,
                                     int32_t nbInputs) const noexcept {
  PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0)
  return inputTypes[0];
}

void GridSamplerPlugin::attachToContext(
    cudnnContext *cudnn, cublasContext *cublas,
    nvinfer1::IGpuAllocator *allocator) noexcept {}

void GridSamplerPlugin::detachFromContext() noexcept {}

void GridSamplerPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(nbInputs == 2)
        PLUGIN_ASSERT(!(mMode == GridSamplerInterpolation::Bicubic &&
                        in[0].desc.dims.nbDims != 4))}

GridSamplerPluginCreator::GridSamplerPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("interpolation_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *GridSamplerPluginCreator::getPluginName() const noexcept {
  return GS_PLUGIN_NAME;
}

char const *GridSamplerPluginCreator::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

PluginFieldCollection const *
GridSamplerPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *GridSamplerPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    int mode, paddingMode;
    bool alignCorners;
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "interpolation_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        mode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "padding_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        paddingMode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "align_corners")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        alignCorners = (bool)(*(static_cast<int const *>(fields[i].data)));
      }
    }

    auto *plugin =
        new GridSamplerPlugin(mode, paddingMode, alignCorners, false);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *GridSamplerPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin = new GridSamplerPlugin{serialData, serialLength, false};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

GridSamplerPluginCreator2::GridSamplerPluginCreator2() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("interpolation_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("padding_mode", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("align_corners", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *GridSamplerPluginCreator2::getPluginName() const noexcept {
  return GS_PLUGIN_NAME2;
}

char const *GridSamplerPluginCreator2::getPluginVersion() const noexcept {
  return GS_PLUGIN_VERSION;
}

PluginFieldCollection const *
GridSamplerPluginCreator2::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *GridSamplerPluginCreator2::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
    int mode, paddingMode;
    bool alignCorners;
    PluginField const *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; i++) {
      char const *attrName = fields[i].name;
      if (!strcmp(attrName, "interpolation_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        mode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "padding_mode")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        paddingMode = *(static_cast<int const *>(fields[i].data));
      } else if (!strcmp(attrName, "align_corners")) {
        PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
        alignCorners = (bool)(*(static_cast<int const *>(fields[i].data)));
      }
    }

    auto *plugin = new GridSamplerPlugin(mode, paddingMode, alignCorners, true);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *GridSamplerPluginCreator2::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) noexcept {
  try {
    auto *plugin = new GridSamplerPlugin{serialData, serialLength, true};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);
REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator2);
