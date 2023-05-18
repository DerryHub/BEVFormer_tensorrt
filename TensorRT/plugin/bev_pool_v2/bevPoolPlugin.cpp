//
// Created by Derry Lin on 2023/05/11.
//

#include "bevPoolPlugin.h"
#include "bevPoolKernel.h"
#include "checkMacrosPlugin.h"
#include "serialize.h"
#include <cuda_fp16.h>
#include <stdexcept>

using trt_plugin::BEVPoolPlugin;
using trt_plugin::BEVPoolPluginCreator;
using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
constexpr char const *I_PLUGIN_VERSION{"1"};
constexpr char const *I_PLUGIN_NAME{"BEVPoolV2TRT"};
} // namespace

PluginFieldCollection BEVPoolPluginCreator::mFC{};
std::vector<PluginField> BEVPoolPluginCreator::mPluginAttributes;

BEVPoolPlugin::BEVPoolPlugin(int outWidth, int outHeight) : mOutWidth(outWidth), mOutHeight(outHeight) {}

BEVPoolPlugin::BEVPoolPlugin(const void *serialData, size_t serialLength) {
    deserialize_value(&serialData, &serialLength, &mOutWidth);
    deserialize_value(&serialData, &serialLength, &mOutHeight);
}

BEVPoolPlugin::~BEVPoolPlugin() { terminate(); }

int32_t BEVPoolPlugin::getNbOutputs() const noexcept { return 1; }

DimsExprs BEVPoolPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept {
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = exprBuilder.constant(1);
    ret.d[1] = exprBuilder.constant(mOutHeight);
    ret.d[2] = exprBuilder.constant(mOutWidth);
    ret.d[3] = inputs[1].d[3];
    return ret;
}

int32_t BEVPoolPlugin::initialize() noexcept { return 0; }

void BEVPoolPlugin::terminate() noexcept {
}

size_t BEVPoolPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                      int32_t nbInputs,
                                      const nvinfer1::PluginTensorDesc *outputs,
                                      int32_t nbOutputs) const noexcept {
    return 0;
}

int32_t BEVPoolPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                              const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs,
                              void *workspace, cudaStream_t stream) noexcept {
    nvinfer1::Dims feat_dims = inputDesc[1].dims; // bnhwc
    nvinfer1::Dims interval_dims = inputDesc[5].dims; // n
    nvinfer1::Dims out_dims = outputDesc[0].dims; //bhwc
    auto data_type = inputDesc[0].type;
    int num_points = out_dims.d[0]*out_dims.d[1]*out_dims.d[2]*out_dims.d[3];
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            bev_pool_v2_set_zero(num_points, (float *)outputs[0]);
            bev_pool_v2(feat_dims.d[3], interval_dims.d[0], (float *)inputs[0], (float *)inputs[1],
                        (int *)inputs[2], (int *)inputs[3], (int *)inputs[4], (int *)inputs[5],(int *)inputs[6], (float *)outputs[0],
                        stream);
            break;
        default:
            return 1;
    }

  return 0;
}

size_t BEVPoolPlugin::getSerializationSize() const noexcept {
    return serialized_size(mOutWidth) + serialized_size(mOutHeight);
}

void BEVPoolPlugin::serialize(void *buffer) const noexcept {
    serialize_value(&buffer, mOutWidth);
    serialize_value(&buffer, mOutHeight);
}

bool BEVPoolPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (pos == 0 || pos==1 || pos == 7) {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT &&
                inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    } else {
        return (inOut[pos].type == nvinfer1::DataType::kINT32 &&
                inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
    }
}

char const *BEVPoolPlugin::getPluginType() const noexcept {
  return I_PLUGIN_NAME;
}

char const *BEVPoolPlugin::getPluginVersion() const noexcept {
  return I_PLUGIN_VERSION;
}

void BEVPoolPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt *BEVPoolPlugin::clone() const noexcept {
  try {
    auto *plugin = new BEVPoolPlugin(mOutWidth, mOutHeight);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void BEVPoolPlugin::setPluginNamespace(const char *pluginNamespace) noexcept {
  mPluginNamespace = pluginNamespace;
}

char const *BEVPoolPlugin::getPluginNamespace() const noexcept {
  return mPluginNamespace.c_str();
}

DataType BEVPoolPlugin::getOutputDataType(int32_t index,
                                         const nvinfer1::DataType *inputTypes,
                                         int32_t nbInputs) const noexcept {
  PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0)
  return inputTypes[0];
}

void BEVPoolPlugin::attachToContext(
    cudnnContext *cudnn, cublasContext *cublas,
    nvinfer1::IGpuAllocator *allocator) noexcept {
}

void BEVPoolPlugin::detachFromContext() noexcept {}

void BEVPoolPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                                   int32_t nbInputs,
                                   const nvinfer1::DynamicPluginTensorDesc *out,
                                   int32_t nbOutputs) noexcept {
    PLUGIN_ASSERT(nbInputs == 7);
    PLUGIN_ASSERT(nbOutputs == 1);
}

BEVPoolPluginCreator::BEVPoolPluginCreator() {
  mPluginAttributes.clear();
    mPluginAttributes.emplace_back(
            PluginField("out_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
            PluginField("out_width", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

char const *BEVPoolPluginCreator::getPluginName() const noexcept {
  return I_PLUGIN_NAME;
}

char const *BEVPoolPluginCreator::getPluginVersion() const noexcept {
  return I_PLUGIN_VERSION;
}

PluginFieldCollection const *BEVPoolPluginCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2DynamicExt *BEVPoolPluginCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept {
  try {
      int outWidth, outHeight;
      PluginField const *fields = fc->fields;
      for (int i = 0; i < fc->nbFields; i++) {
          char const *attrName = fields[i].name;
          if (!strcmp(attrName, "out_height")) {
              PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
              outHeight = *(static_cast<int const *>(fields[i].data));
          } else if (!strcmp(attrName, "out_width")) {
              PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
              outWidth = *(static_cast<int const *>(fields[i].data));
          }
      }
    auto *plugin = new BEVPoolPlugin(outWidth, outHeight);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2DynamicExt *
BEVPoolPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                       size_t serialLength) noexcept {
  try {
    auto *plugin = new BEVPoolPlugin{serialData, serialLength};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}


REGISTER_TENSORRT_PLUGIN(BEVPoolPluginCreator);
