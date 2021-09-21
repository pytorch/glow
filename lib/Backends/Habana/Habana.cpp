/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Habana.h"
#include "HabanaUtils.h"

#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Support.h"

#include "perf_lib_layer_params.h"
#include "synapse.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <chrono>
#include <fstream>
#include <glog/logging.h>
#include <numeric>
#include <unordered_map>

using namespace glow;

/// Get a path to a temporary file for the compiled recipe.
static Expected<std::string> getRecipeFile() {
  llvm::SmallString<64> path;
  const auto err = llvm::sys::fs::createTemporaryFile("glow", "recipe", path);
  RETURN_ERR_IF_NOT(
      !err, strFormat("Unable to create recipe file at %s", path.str().data()));
  return path.str().str();
}

/// Convert a Glow data type to a Synapse data type.
static synDataType getSynType(ElemKind kind) {
  switch (kind) {
  case ElemKind::FloatTy:
    return syn_type_single;
  case ElemKind::Int8QTy:
    return syn_type_fixed;
  case ElemKind::UInt8QTy:
    return syn_type_fixed;
  case ElemKind::Int16QTy:
    return syn_type_int16;
  case ElemKind::Int32QTy:
    return syn_type_int32;
  case ElemKind::Int32ITy:
    return syn_type_int32;
  case ElemKind::Int64ITy:
    // TODO: This backend does not have a 64-bit type, but Glow uses
    // it pervasively as an "index" type. Is this a problem?
    return syn_type_int32;
  case ElemKind::UInt8FusedQTy:
    return syn_type_fixed;
  default:
    LOG(FATAL) << "Unsupported data type: " << Type::getElementName(kind).str();
  }
}

static const char *getKernelSuffix(ElemKind kind) {
  switch (kind) {
  case ElemKind::Int8QTy:
    return "_i8";
  case ElemKind::Int16QTy:
    return "_i16";
  case ElemKind::FloatTy:
    return "_f32";
  default:
    LOG(FATAL) << "Unsupported data type: " << Type::getElementName(kind).str();
  }
}

static std::string getKernelName(llvm::StringRef kernelBase, ElemKind kind) {
  return std::string(kernelBase) + getKernelSuffix(kind);
}

namespace {

/// Enum describing how the tensor will be used by the model.
enum class IOType {
  /// Intermediate result between nodes.
  Intermediate,
  /// Tensor is a model input.
  Input,
  /// Tensor is a model output.
  Output,
  /// Tensor is a static parameter of the model.
  Static,
};

/// Handle to a synTensor.
class TensorHandle final {
  /// Underlying storage for the tensor.
  void *buffer_;
  bool allocated_{false};

  /// Name of the tensor.
  std::string name_;

  /// The tensor object.
  synTensor tensor_;

  /// The dimensions of the tensor object.
  llvm::SmallVector<unsigned, SYN_MAX_TENSOR_DIM> dims_;

  /// Valid bit to avoid destruction of empty handles.
  bool valid_{false};

public:
  /// Create an un-allocated handle.  Useful for moving objects.
  TensorHandle() = default;

  /// Constructor. Create a Synapse tensor.
  /// \param downcast64To32 If this tensor is a 64-bit type, forcible cast to
  /// 32-bits. \param V Type of the tensor \param name a string identifying the
  /// tensor \param buffer the static weights of the tensor \param ioType the
  /// I/O class of the tensor
  TensorHandle(bool downcast64To32, TypeRef V, llvm::StringRef name,
               void *buffer = nullptr, IOType ioType = IOType::Intermediate)
      : buffer_(buffer), name_(name) {
    if (!buffer_) {
      DCHECK_GT(V->getSizeInBytes(), 0);
      DCHECK(ioType != IOType::Static);
      buffer_ = malloc(ioType == IOType::Intermediate ? sizeof(float)
                                                      : V->getSizeInBytes());
      DCHECK_NOTNULL(buffer_);
      allocated_ = true;
    }

    auto elemType = getSynType(V->getElementType());
    auto dims = V->dims();
    dims_.append(dims.begin(), dims.end());

    DCHECK_LE(dims.size(), SYN_MAX_TENSOR_DIM);
    llvm::SmallVector<unsigned, SYN_MAX_TENSOR_DIM> rdims(dims.rbegin(),
                                                          dims.rend());

    // Unless we can safely downcast, we fake 64-bit indices by reading every
    // other element of a 32-bit tensor, so we need to double the fastest
    // moving (first) dimension.
    if (V->getElementType() == ElemKind::Int64ITy && !downcast64To32) {
      rdims[0] *= 2;
    }

    // Model params need to be floats, even if the tensor is integral or
    // quantized.
    if (ioType == IOType::Static) {
      // Int32ITy: Cast to floats.
      if (V->getElementType() == ElemKind::Int32ITy) {
        float *floats_ = (float *)malloc(V->size() * sizeof(float));
        for (size_t i = 0; i < V->size(); i++) {
          floats_[i] = static_cast<int32_t *>(buffer_)[i];
        }
        buffer_ = floats_;
        allocated_ = true;
      }

      // Int64ITy: Remember 64-bit indices are faked as pairs of (0, int32),
      // unless we are able to safely downcast to int32.
      if (V->getElementType() == ElemKind::Int64ITy) {
        size_t size = downcast64To32 ? V->size() : 2 * V->size();
        float *floats_ = (float *)calloc(size, sizeof(float));
        for (size_t i = 0; i < V->size(); i++) {
          floats_[i] = static_cast<int64_t *>(buffer_)[i];
          if (!downcast64To32) {
            i++;
          }
        }
        buffer_ = floats_;
        allocated_ = true;
      }
    }

    // Create tensor descriptor, with quantization params if needed.
    synTensorDescriptor desc(elemType, rdims.size(), rdims.data(), buffer_,
                             synMemoryHost, false, name_.data());
    if (V->isQuantizedType()) {
      if (V->getElementType() == ElemKind::UInt8FusedQTy) {
        desc.m_quantizationParams[0].m_zp = 0;
        desc.m_quantizationParams[0].m_scale = 1;
      } else {
        desc.m_quantizationParams[0].m_zp = V->getOffset();
        desc.m_quantizationParams[0].m_scale = V->getScale();
      }

      desc.m_quantizationParams[0].m_qDataType = elemType;
      if (ioType == IOType::Static) {
        desc.m_isQuantized = true;
      }
    }

    chk_kill(synCreateTensor(&desc, &tensor_, ioType == IOType::Output, false,
                             ioType == IOType::Static));
    valid_ = true;
  }

  /// Constructor. Create a tensor with Type \p V.
  TensorHandle(TypeRef V, llvm::StringRef name, void *buffer = nullptr,
               IOType ioType = IOType::Intermediate)
      : TensorHandle(/* downcast64To32 */ false, V, name, buffer, ioType) {}

  /// Non-copyable.
  ///@{
  TensorHandle(const TensorHandle &) = delete;
  TensorHandle &operator=(const TensorHandle &) = delete;
  ///@}

  /// Move constructor.
  TensorHandle(TensorHandle &&o) { *this = std::move(o); }

  /// Move assignment.
  TensorHandle &operator=(TensorHandle &&o) {
    std::swap(buffer_, o.buffer_);
    std::swap(allocated_, o.allocated_);
    std::swap(name_, o.name_);
    std::swap(tensor_, o.tensor_);
    std::swap(valid_, o.valid_);
    std::swap(dims_, o.dims_);
    return *this;
  }

  /// Destroy the managed tensor.
  ~TensorHandle() {
    if (valid_) {
      chk_kill(synDestroyTensor(tensor_));

      if (allocated_) {
        free(buffer_);
      }
    }
  }

  /// Get the managed tensor.
  synTensor &get() { return tensor_; }

  /// Get the underlying data buffer.
  void *getData() const { return buffer_; }

  /// Get the name of the managed tensor
  const std::string &getName() const { return name_; }

  /// Get the dimensions of the stored tensor.
  llvm::ArrayRef<unsigned> dims() const { return dims_; }

  /// \returns true if this handle has an allocated tensor.
  bool isValid() const { return valid_; }
};
} // namespace

static std::unique_ptr<synConvolutionParams> makeSynConvolutionParams(
    llvm::ArrayRef<unsigned_t> kernel, llvm::ArrayRef<unsigned_t> stride,
    llvm::ArrayRef<unsigned_t> pad, unsigned_t groups, bool doRelu) {
  auto params = glow::make_unique<synConvolutionParams>();

  // Kernel
  params->kH = kernel[0];
  params->kW = kernel[1];
  // Stride
  params->dH = stride[0];
  params->dW = stride[1];
  // Padding
  params->padT = pad[0];
  params->padL = pad[1];
  params->padB = pad[2];
  params->padR = pad[3];
  // Dilation
  params->dilW = 1;
  params->dilH = 1;
  // Activation params
  params->activation.reluEnable = doRelu;
  // Number of convolution groups, 1 means regular convolution
  params->nGroups = groups;

  return params;
}

static std::unique_ptr<ns_SpatialReduction::Params>
makeSynPoolParams(llvm::ArrayRef<unsigned_t> kernel,
                  llvm::ArrayRef<unsigned_t> stride,
                  llvm::ArrayRef<unsigned_t> pad) {
  auto params = glow::make_unique<ns_SpatialReduction::Params>();

  // Kernel
  params->kernel_w = kernel[0];
  params->kernel_h = kernel[1];
  // Stride
  params->stride_w = stride[0];
  params->stride_h = stride[1];
  // Padding
  params->pad_h_begin = pad[0];
  params->pad_w_begin = pad[1];
  params->pad_h_end = pad[2];
  params->pad_w_end = pad[3];
  // Dilation
  params->dilation_w = 1;
  params->dilation_h = 1;

  return params;
}

static std::unique_ptr<synTransposeParams>
makeSynTransposeParams(llvm::ArrayRef<unsigned_t> shuffle) {
  auto params = glow::make_unique<synTransposeParams>();

  params->tensorDim = shuffle.size();

  // To convert from Glow's shuffle convention to Habana's, reverse the
  // dimensions and take the tensorDim complement of each dimension (subtract 1
  // to account for 0-based indexing).
  size_t i = 0;
  for (auto it = shuffle.rbegin(), end = shuffle.rend(); it != end; ++it) {
    params->permutation[i++] =
        (TransposePermutationDim)(params->tensorDim - *it - 1);
  }

  return params;
}

static std::unique_ptr<synSliceAxisParams>
makeSynSliceAxisParams(unsigned axis, unsigned axes, unsigned outputAxisSize,
                       unsigned axisOffset) {
  auto params = glow::make_unique<synSliceAxisParams>();

  // The axis complement must be taken since Habana's axes in reverse order
  // compared to Glow.
  params->axis = axes - axis - 1;
  params->begin = axisOffset;
  params->end = axisOffset + outputAxisSize;

  return params;
}

static std::unique_ptr<ns_LrnKernel::Params>
makeLrnParams(float alpha, float beta, float knorm, int halfWindowSize) {
  auto params = glow::make_unique<ns_LrnKernel::Params>();
  params->alpha = alpha;
  params->beta = beta;
  params->knorm = knorm;
  params->nsize = 2 * halfWindowSize + 1;
  return params;
}

static std::unique_ptr<ns_ConstantKernel::Params>
makeConstantParams(float value) {
  auto params = glow::make_unique<ns_ConstantKernel::Params>();
  params->constant.f = value;
  return params;
}

static std::unique_ptr<ns_FullyConnected::Params> makeFCFPParams(bool isRelu) {
  auto params = glow::make_unique<ns_FullyConnected::Params>();
  params->is_relu = isRelu;
  return params;
}

static std::unique_ptr<ns_TileKernel::Params> makeTileParams(unsigned count,
                                                             unsigned axis) {
  auto params = glow::make_unique<ns_TileKernel::Params>();

  // The repeat member of ns_TileKernel::Params has an explicit size of 4.
  for (size_t i = 0; i < 4; ++i) {
    params->repeat[i] = 1;
  }

  // The axis complement must be taken since Habana's axes in reverse order
  // compared to Glow. SYN_MAX_TENSOR_DIM - 2 is the maximum number of axes
  // except the batch dimension.
  params->repeat[SYN_MAX_TENSOR_DIM - axis - 2] = count;

  return params;
}

static std::unique_ptr<unsigned> makeConcatParams(unsigned axis,
                                                  unsigned axes) {
  return glow::make_unique<unsigned>(axes - axis - 1);
}

static std::unique_ptr<ns_Softmax::Params> makeSoftmaxParams() {
  auto params = glow::make_unique<ns_Softmax::Params>();
  params->dim = 0;
  return params;
}

/// Allocate synTensors for every tensor used in \p F.
static std::unordered_map<const Node *, TensorHandle>
allocateGraphTensors(Function *F) {
  std::unordered_map<const Node *, TensorHandle> tensors;

  // Weights (Constants).  Assigns payloads as well.
  for (auto &V : F->getParent()->getConstants()) {
    if (V->getNumUsers() == 0) {
      continue;
    }
    bool downcast = allows64To32Downcast(V, F);
    tensors.emplace(V, TensorHandle(downcast, V->getType(), V->getName(),
                                    V->getPayload().getUnsafePtr(),
                                    IOType::Static));
  }

  // Placeholders (input/output).
  for (auto &V : F->getParent()->getPlaceholders()) {
    if (V->getNumUsers() == 0) {
      continue;
    }
    if (auto *save = getOutputSave(F, V)) {
      // Naively, we'd generate a memcpy for any SaveNode, but that's a waste
      // so we want to avoid it.  We can optimize it away by mapping the
      // SaveNode's input node (N, below) to the output tensor, and then simply
      // not generating a memcpy if the SaveNode itself has no associated
      // tensor.
      auto *N = save->getInput().getNode();
      if (llvm::isa<Storage>(N) || llvm::isa<ReshapeNode>(N) ||
          N->getNumUsers() > 1) {
        N = save;
      }
      tensors.emplace(
          N, TensorHandle(V->getType(), V->getName(), nullptr, IOType::Output));
    } else {
      bool downcast = allows64To32Downcast(V, F);
      tensors.emplace(V, TensorHandle(downcast, V->getType(), V->getName(),
                                      nullptr, IOType::Input));
    }
  }

  // Activations.
  for (auto const &N : F->getNodes()) {
    // Skip anything we allocated while processing output placeholders.
    if (tensors.count(&N)) {
      continue;
    }
    if (llvm::isa<SaveNode>(N)) {
      continue;
    }
    auto result = N.getNthResult(0);
    tensors.emplace(&N, TensorHandle(result.getType(), N.getName(), nullptr,
                                     IOType::Intermediate));
  }
  return tensors;
}

Expected<std::unique_ptr<CompiledFunction>>
HabanaBackend::compile(Function *F, const BackendOptions &opts) const {
  chk(synCreateGraph(synDeviceGoya));

  // Allocate all the tensors.
  auto tensors = allocateGraphTensors(F);

  // Keep references to any node parameters
  // until the compilation is done.
  std::vector<std::unique_ptr<synConvolutionParams>> convParams;
  std::vector<std::unique_ptr<synTransposeParams>> transposeParams;
  std::vector<std::unique_ptr<ns_SpatialReduction::Params>> poolParams;
  std::vector<std::unique_ptr<synFCParams>> fcParams;
  std::vector<std::unique_ptr<ns_FullyConnected::Params>> fcFp32Params;
  std::vector<std::unique_ptr<ns_Reduction::Params>> reductionParams;
  std::vector<std::unique_ptr<synSliceAxisParams>> sliceParams;
  std::vector<std::unique_ptr<ns_ConstantKernel::Params>> constantParams;
  std::vector<std::unique_ptr<ns_TileKernel::Params>> tileParams;
  std::vector<std::unique_ptr<unsigned>> concatParams;
  std::vector<std::unique_ptr<ns_GatherKernel::Params>> gatherParams;
  std::vector<std::unique_ptr<ns_LrnKernel::Params>> lrnParams;
  std::vector<std::unique_ptr<synGEMMParams>> gemmParams;
  std::vector<std::unique_ptr<ns_Softmax::Params>> softmaxParams;
  std::vector<std::unique_ptr<float>> batchBoxCoxParams;

  // Keep references to tensor pointer arrays passed into multi-input nodes
  // until the compilation is done.
  std::vector<std::vector<synTensor>> multiInputs;

  std::vector<TensorHandle> tempTensors;

  for (const auto &I : F->getNodes()) {
    RETURN_ERR_IF_NOT(isOpSupported(I), strFormat("Unsupported operator: %s",
                                                  I.getDebugDesc().c_str()));

    switch (I.getKind()) {
    case Kinded::Kind::HabanaFullyConnectedNodeKind: {
      auto *NI = llvm::cast<HabanaFullyConnectedNode>(&I);

      if (NI->getInput().getType()->isQuantizedType()) {
        auto params = glow::make_unique<synFCParams>();
        params->activation.reluEnable = NI->getDoRelu();
        chk(synFullyConnected(
            tensors[NI->getInput()].get(), tensors[NI->getWeights()].get(),
            tensors[NI->getBias()].get(), tensors[NI].get(), *params, ""));
        fcParams.emplace_back(std::move(params));
      } else {
        std::vector<synTensor> inputs;
        inputs.push_back(tensors[NI->getInput()].get());
        inputs.push_back(tensors[NI->getWeights()].get());
        inputs.push_back(tensors[NI->getBias()].get());

        auto params = makeFCFPParams(NI->getDoRelu());
        const char *fcLayoutIn[3] = {"CB", "CK", ""};
        const char *fcLayoutOut[1] = {"KB"};
        const char **inLayout = {nullptr};
        const char **outLayout = {nullptr};
        if (isVersionBiggerEqualTo("0.1.9")) {
          inLayout = fcLayoutIn;
          outLayout = fcLayoutOut;
        }

        chk(synCreateGenericNode(
            inputs.data(), &tensors[NI].get(), 3, 1, params.get(),
            getKernelName("fully_connected", NI->getResult().getElementType())
                .c_str(),
            NI->getName().data(), inLayout, outLayout));
        multiInputs.emplace_back(std::move(inputs));
        fcFp32Params.emplace_back(std::move(params));
      }
      break;
    }
    case Kinded::Kind::SigmoidNodeKind: {
      auto *SI = llvm::cast<SigmoidNode>(&I);
      chk(synCreateGenericNode(
          &tensors[SI->getInput()].get(), &tensors[SI].get(), 1, 1, nullptr,
          getKernelName("sigmoid", SI->getInput().getElementType()).c_str(),
          SI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::TanhNodeKind: {
      auto *TI = llvm::cast<TanhNode>(&I);
      chk(synCreateGenericNode(
          &tensors[TI->getInput()].get(), &tensors[TI].get(), 1, 1, nullptr,
          getKernelName("tanh", TI->getInput().getElementType()).c_str(),
          TI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::BatchedAddNodeKind: {
      auto *BA = llvm::cast<BatchedAddNode>(&I);
      std::vector<synTensor> inputs;

      // Broadcast slice to match input dims.
      TensorHandle broadcastedSlice(BA->getResult().getType(), BA->getName());

      chk(synCreateGenericNode(
          &tensors[BA->getSlice()].get(), &broadcastedSlice.get(), 1, 1,
          nullptr, "broadcast", BA->getName().data(), nullptr, nullptr));

      // Perform element-wise add.
      inputs.push_back(broadcastedSlice.get());
      inputs.push_back(tensors[BA->getBatch()].get());

      chk(synCreateGenericNode(
          inputs.data(), &tensors[BA].get(), 2, 1, nullptr,
          getKernelName("add", BA->getResult().getElementType()).data(),
          BA->getName().data(), nullptr, nullptr));

      tempTensors.emplace_back(std::move(broadcastedSlice));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::BatchedReduceAddNodeKind: {
      auto *RA = llvm::cast<BatchedReduceAddNode>(&I);
      auto params = glow::make_unique<ns_Reduction::Params>();
      params->reductionDimension =
          RA->getBatch().dims().size() - RA->getAxis() - 1;

      std::vector<size_t> tmpResultDims = RA->getBatch().dims();
      tmpResultDims[RA->getAxis()] = 1;
      Type tmpType = Type::newShape(*RA->getBatch().getType(), tmpResultDims);

      // Temporary result of reduce_sum op which needs to be reshaped later.
      TensorHandle tempResult(&tmpType, "tmp." + RA->getName().str());

      chk(synCreateGenericNode(
          &tensors[RA->getBatch()].get(), &tempResult.get(), 1, 1, params.get(),
          getKernelName("reduce_sum", RA->getResult().getElementType()).data(),
          RA->getName().data(), nullptr, nullptr));

      // Reshape temp result to the shape Glow expects.
      chk(synCreateGenericNode(
          &tempResult.get(), &tensors[RA->getResult()].get(), 1, 1, nullptr,
          "Reshape", RA->getName().data(), nullptr, nullptr));

      reductionParams.emplace_back(std::move(params));
      tempTensors.emplace_back(std::move(tempResult));
      break;
    }
    case Kinded::Kind::ReshapeNodeKind: {
      auto *RI = llvm::cast<ReshapeNode>(&I);
      chk(synCreateGenericNode(&tensors[RI->getInput()].get(),
                               &tensors[RI].get(), 1, 1, nullptr, "Reshape",
                               RI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::ReluNodeKind: {
      auto *RI = llvm::cast<ReluNode>(&I);
      chk(synCreateGenericNode(
          &tensors[RI->getInput()].get(), &tensors[RI].get(), 1, 1, nullptr,
          getKernelName("relu", RI->getInput().getElementType()).c_str(),
          RI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::MaxPoolNodeKind: {
      auto *PI = llvm::cast<MaxPoolNode>(&I);
      auto params =
          makeSynPoolParams(PI->getKernels(), PI->getStrides(), PI->getPads());
      chk(synCreateGenericNode(
          &tensors[PI->getInput()].get(), &tensors[PI].get(), 1, 1,
          (void *)params.get(),
          getKernelName("maxpool_2d", PI->getInput().getElementType()).c_str(),
          PI->getName().str().c_str(), nullptr, nullptr));
      poolParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::AvgPoolNodeKind: {
      auto *PI = llvm::cast<AvgPoolNode>(&I);
      auto params =
          makeSynPoolParams(PI->getKernels(), PI->getStrides(), PI->getPads());
      chk(synCreateGenericNode(
          &tensors[PI->getInput()].get(), &tensors[PI].get(), 1, 1,
          (void *)params.get(),
          getKernelName("avg_pool_2d", PI->getInput().getElementType()).c_str(),
          PI->getName().str().c_str(), nullptr, nullptr));
      poolParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::LogNodeKind: {
      auto *LI = llvm::cast<LogNode>(&I);
      chk(synCreateGenericNode(
          &tensors[LI->getInput()].get(), &tensors[LI].get(), 1, 1, nullptr,
          getKernelName("log", LI->getResult().getElementType()).c_str(),
          LI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::MaxNodeKind: {
      auto *MI = llvm::cast<MaxNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[MI->getLHS()].get());
      inputs.push_back(tensors[MI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[MI].get(), inputs.size(), 1, nullptr,
          getKernelName("max", MI->getResult().getElementType()).c_str(),
          MI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::MinNodeKind: {
      auto *MI = llvm::cast<MinNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[MI->getLHS()].get());
      inputs.push_back(tensors[MI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[MI].get(), inputs.size(), 1, nullptr,
          getKernelName("min", MI->getResult().getElementType()).c_str(),
          MI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::AddNodeKind: {
      auto *AI = llvm::cast<AddNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[AI->getLHS()].get());
      inputs.push_back(tensors[AI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[AI].get(), inputs.size(), 1, nullptr,
          getKernelName("add", AI->getResult().getElementType()).c_str(),
          AI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::SubNodeKind: {
      auto *SI = llvm::cast<SubNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[SI->getLHS()].get());
      inputs.push_back(tensors[SI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[SI].get(), inputs.size(), 1, nullptr,
          getKernelName("sub", SI->getResult().getElementType()).c_str(),
          SI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::MulNodeKind: {
      auto *SI = llvm::cast<MulNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[SI->getLHS()].get());
      inputs.push_back(tensors[SI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[SI].get(), inputs.size(), 1, nullptr,
          getKernelName("mult", SI->getResult().getElementType()).c_str(),
          SI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::DivNodeKind: {
      auto *DI = llvm::cast<DivNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[DI->getLHS()].get());
      inputs.push_back(tensors[DI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[DI].get(), inputs.size(), 1, nullptr,
          getKernelName("div", DI->getResult().getElementType()).c_str(),
          DI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::MatMulNodeKind: {
      auto *MI = llvm::cast<MatMulNode>(&I);

      if (MI->getLHS().getType()->isQuantizedType()) {
        // Let GEMM run on MME via FullyConnected node.
        // MME only runs on quantized types, e.g., int8 or int16.
        // The default params are OK - don't transpose A and B
        auto params = glow::make_unique<synGEMMParams>();
        std::vector<synTensor> inputs;
        inputs.push_back(tensors[MI->getLHS()].get());
        inputs.push_back(tensors[MI->getRHS()].get());
        inputs.push_back(/* bias */ nullptr);
        chk(synCreateGenericNode(inputs.data(), &tensors[MI].get(),
                                 inputs.size(), 1, nullptr, "gemm",
                                 MI->getName().data(), nullptr, nullptr));
        gemmParams.emplace_back(std::move(params));

      } else {
        std::vector<synTensor> inputs;
        inputs.push_back(tensors[MI->getLHS()].get());
        inputs.push_back(tensors[MI->getRHS()].get());

        chk(synCreateGenericNode(
            inputs.data(), &tensors[MI].get(), 2, 1, nullptr,
            getKernelName("matrix_multiply", MI->getResult().getElementType())
                .c_str(),
            MI->getName().data(), nullptr, nullptr));
        multiInputs.emplace_back(std::move(inputs));
      }
      break;
    }
    case Kinded::Kind::BatchMatMulNodeKind: {
      auto *MI = llvm::cast<BatchMatMulNode>(&I);
      std::vector<synTensor> inputs;
      inputs.push_back(tensors[MI->getLHS()].get());
      inputs.push_back(tensors[MI->getRHS()].get());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[MI].get(), 2, 1, nullptr,
          getKernelName("matrix_multiply", MI->getResult().getElementType())
              .c_str(),
          MI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::HabanaConvolutionNodeKind: {
      auto *NI = llvm::cast<HabanaConvolutionNode>(&I);

      std::unique_ptr<synConvolutionParams> params = makeSynConvolutionParams(
          NI->getKernels(), NI->getStrides(), NI->getPads(), NI->getGroup(),
          NI->getDoRelu());

      chk(synSpatialConvolution(tensors[NI->getInput()].get(),
                                tensors[NI->getFilter()].get(),
                                tensors[NI->getBias()].get(), tensors[NI].get(),
                                nullptr, *params, ""));

      convParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::HabanaConvolutionAddNodeKind: {
      auto *NI = llvm::cast<HabanaConvolutionAddNode>(&I);

      std::unique_ptr<synConvolutionParams> params = makeSynConvolutionParams(
          NI->getKernels(), NI->getStrides(), NI->getPads(), NI->getGroup(),
          NI->getDoRelu());

      chk(synSpatialConvolution(tensors[NI->getInput()].get(),
                                tensors[NI->getFilter()].get(),
                                tensors[NI->getBias()].get(), tensors[NI].get(),
                                tensors[NI->getAddend()].get(), *params, ""));

      convParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::LocalResponseNormalizationNodeKind: {
      auto *NI = llvm::cast<LocalResponseNormalizationNode>(&I);
      std::unique_ptr<ns_LrnKernel::Params> params = makeLrnParams(
          NI->getAlpha(), NI->getBeta(), NI->getK(), NI->getHalfWindowSize());

      chk(synCreateGenericNode(&tensors[NI->getInput()].get(),
                               &tensors[NI].get(), 1, 1, (void *)params.get(),
                               "lrn_f32", NI->getName().str().c_str(), nullptr,
                               nullptr));
      lrnParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::TransposeNodeKind: {
      auto *TI = llvm::cast<TransposeNode>(&I);
      std::unique_ptr<synTransposeParams> params =
          makeSynTransposeParams(TI->getShuffle());
      chk(synCreateGenericNode(&tensors[TI->getInput()].get(),
                               &tensors[TI].get(), 1, 1, (void *)params.get(),
                               "Transpose", TI->getName().str().c_str(),
                               nullptr, nullptr));
      transposeParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::SplatNodeKind: {
      auto *SI = llvm::cast<SplatNode>(&I);
      std::unique_ptr<ns_ConstantKernel::Params> params =
          makeConstantParams(SI->getValue());
      chk(synCreateGenericNode(
          &tensors[SI].get(), &tensors[SI].get(), 0, 1, (void *)params.get(),
          getKernelName("constant", SI->getResult().getElementType()).c_str(),
          SI->getName().data(), nullptr, nullptr));
      constantParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::QuantizeNodeKind: {
      auto *QI = llvm::cast<QuantizeNode>(&I);
      std::string kernel = llvm::formatv(
          "cast{0}_to{1}", getKernelSuffix(QI->getInput().getElementType()),
          getKernelSuffix(QI->getResult().getElementType()));
      chk(synCreateGenericNode(&tensors[QI->getInput()].get(),
                               &tensors[QI].get(), 1, 1, nullptr, kernel.data(),
                               QI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::DequantizeNodeKind: {
      auto *DI = llvm::cast<DequantizeNode>(&I);
      std::string kernel = llvm::formatv(
          "cast{0}_to{1}", getKernelSuffix(DI->getInput().getElementType()),
          getKernelSuffix(DI->getResult().getElementType()));
      chk(synCreateGenericNode(&tensors[DI->getInput()].get(),
                               &tensors[DI].get(), 1, 1, nullptr, kernel.data(),
                               DI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::SoftMaxNodeKind: {
      auto *SI = llvm::cast<SoftMaxNode>(&I);
      std::unique_ptr<ns_Softmax::Params> params = makeSoftmaxParams();
      chk(synCreateGenericNode(
          &tensors[SI->getInput()].get(), &tensors[SI].get(), 1, 1,
          (void *)params.get(),
          getKernelName("softmax", SI->getResult().getElementType()).c_str(),
          SI->getName().data(), nullptr, nullptr));
      softmaxParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::SliceNodeKind: {
      auto *ETI = llvm::cast<SliceNode>(&I);

      auto iDims = tensors[ETI->getInput()].dims();
      auto oDims = tensors[ETI].dims();
      auto nDims = iDims.size();

      auto offsets = ETI->getStart();

      for (size_t i = 0, e = iDims.size(); i < e; ++i) {
        if (iDims[i] != oDims[i]) {
          std::unique_ptr<synSliceAxisParams> params =
              makeSynSliceAxisParams(i, nDims, oDims[i], offsets[i]);
          chk(synCreateGenericNode(&tensors[ETI->getInput()].get(),
                                   &tensors[ETI].get(), 1, 1,
                                   (void *)params.get(), "slice_axis",
                                   ETI->getName().data(), nullptr, nullptr));
          sliceParams.emplace_back(std::move(params));
          break;
        }
      }
      break;
    }
    case Kinded::Kind::TileNodeKind: {
      auto *TI = llvm::cast<TileNode>(&I);
      std::unique_ptr<ns_TileKernel::Params> params =
          makeTileParams(TI->getCount(), TI->getAxis());
      chk(synCreateGenericNode(
          &tensors[TI->getInput()].get(), &tensors[TI].get(), 1, 1,
          (void *)params.get(),
          getKernelName("tile", TI->getResult().getElementType()).c_str(),
          TI->getName().data(), nullptr, nullptr));
      tileParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::ConcatNodeKind: {
      auto *CI = llvm::cast<ConcatNode>(&I);
      std::unique_ptr<unsigned> params =
          makeConcatParams(CI->getDim(), tensors[CI].dims().size());
      std::vector<synTensor> inputs;
      for (auto const &N : CI->getInputs()) {
        if (N.getNumUsers() > 1) {
          std::string memcpyNodeName =
              llvm::formatv("{0}_memcpy_{1}", N.getNode()->getName(),
                            inputs.size())
                  .str();
          TensorHandle memcpy(N.getType(), memcpyNodeName);
          chk(synCreateGenericNode(
              &tensors[N].get(), &memcpy.get(), 1, 1, nullptr,
              getKernelName("memcpy", N.getType()->getElementType()).c_str(),
              memcpy.getName().c_str(), nullptr, nullptr));
          inputs.push_back(memcpy.get());
          tempTensors.emplace_back(std::move(memcpy));
        } else {
          inputs.push_back(tensors[N].get());
        }
      }

      chk(synCreateGenericNode(inputs.data(), &tensors[CI].get(), inputs.size(),
                               1, params.get(), "concat", CI->getName().data(),
                               nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      concatParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::RescaleQuantizedNodeKind: {
      auto *RI = llvm::cast<RescaleQuantizedNode>(&I);
      chk(synCreateGenericNode(
          &tensors[RI->getInput()].get(), &tensors[RI].get(), 1, 1, nullptr,
          getKernelName("requant", RI->getResult().getElementType()).c_str(),
          RI->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::SaveNodeKind: {
      auto *CI = llvm::cast<SaveNode>(&I);
      if (tensors.count(CI)) {
        chk(synCreateGenericNode(&tensors[CI->getInput()].get(),
                                 &tensors[CI].get(), 1, 1, nullptr, "memcpy",
                                 CI->getName().data(), nullptr, nullptr));
      }
      break;
    }
    case Kinded::Kind::BatchOneHotNodeKind: {
      auto *BOHNode = llvm::cast<BatchOneHotNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[BOHNode->getData()].get(),
          tensors[BOHNode->getLengths()].get(),
          tensors[BOHNode->getValues()].get(),
      };
      chk(synCreateGenericNode(inputs.data(), &tensors[BOHNode].get(),
                               inputs.size(), 1, nullptr, "batch_one_hot_i32",
                               BOHNode->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::LengthsSumNodeKind: {
      auto *LSNode = llvm::cast<LengthsSumNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[LSNode->getData()].get(),
          tensors[LSNode->getLengths()].get(),
      };
      chk(synCreateGenericNode(inputs.data(), &tensors[LSNode].get(),
                               inputs.size(), 1, nullptr, "lengths_sum_f32",
                               LSNode->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::LengthsRangeFillNodeKind: {
      auto *LRFNode = llvm::cast<LengthsRangeFillNode>(&I);
      chk(synCreateGenericNode(&tensors[LRFNode->getLengths()].get(),
                               &tensors[LRFNode].get(), 1, 1, nullptr,
                               "lengths_range_fill_i32",
                               LRFNode->getName().data(), nullptr, nullptr));
      break;
    }
    case Kinded::Kind::SparseLengthsSumNodeKind: {
      auto *RI = llvm::cast<SparseLengthsSumNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[RI->getData()].get(),
          tensors[RI->getIndices()].get(),
          tensors[RI->getLengths()].get(),
      };
      chk(synCreateGenericNode(inputs.data(), &tensors[RI].get(), inputs.size(),
                               1, nullptr, "sparse_lengths_sum_f32_f32",
                               RI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::SparseLengthsWeightedSumNodeKind: {
      auto *RI = llvm::cast<SparseLengthsWeightedSumNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[RI->getData()].get(),
          tensors[RI->getIndices()].get(),
          tensors[RI->getLengths()].get(),
          tensors[RI->getWeights()].get(),
      };
      chk(synCreateGenericNode(inputs.data(), &tensors[RI].get(), inputs.size(),
                               1, nullptr, "sparse_lengths_weighted_sum_f32",
                               RI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind: {
      auto *RI = llvm::cast<FusedRowwiseQuantizedSparseLengthsSumNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[RI->getData()].get(),
          tensors[RI->getIndices()].get(),
          tensors[RI->getLengths()].get(),
      };
      chk(synCreateGenericNode(inputs.data(), &tensors[RI].get(), inputs.size(),
                               1, nullptr, "sparse_lengths_sum_u8_2D_f32_embed",
                               RI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind: {
      auto *RI =
          llvm::cast<FusedRowwiseQuantizedSparseLengthsWeightedSumNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[RI->getData()].get(),
          tensors[RI->getIndices()].get(),
          tensors[RI->getLengths()].get(),
          tensors[RI->getWeights()].get(),
      };
      chk(synCreateGenericNode(inputs.data(), &tensors[RI].get(), inputs.size(),
                               1, nullptr,
                               "sparse_lengths_weighted_sum_u8_2D_f32_embed",
                               RI->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      break;
    }
    case Kinded::Kind::GatherNodeKind: {
      auto *gather = llvm::cast<GatherNode>(&I);
      std::vector<synTensor> inputs = {tensors[gather->getData()].get(),
                                       tensors[gather->getIndices()].get()};

      auto params = glow::make_unique<ns_GatherKernel::Params>();
      params->axis =
          gather->getData().dims().size() - gather->getBatchDims() - 1;

      chk(synCreateGenericNode(
          inputs.data(), &tensors[gather].get(), inputs.size(), 1, params.get(),
          getKernelName("gather", gather->getResult().getElementType()).c_str(),
          gather->getName().data(), nullptr, nullptr));

      multiInputs.emplace_back(std::move(inputs));
      gatherParams.emplace_back(std::move(params));
      break;
    }
    case Kinded::Kind::BatchBoxCoxNodeKind: {
      auto *BBC = llvm::cast<BatchBoxCoxNode>(&I);
      std::vector<synTensor> inputs = {
          tensors[BBC->getInput()].get(),
          tensors[BBC->getLambda1()].get(),
          tensors[BBC->getLambda2()].get(),
      };
      auto params = glow::make_unique<float>(BBC->getEpsilon());
      chk(synCreateGenericNode(
          inputs.data(), &tensors[BBC].get(), inputs.size(), 1, params.get(),
          "batch_box_cox_f32", BBC->getName().data(), nullptr, nullptr));
      multiInputs.emplace_back(std::move(inputs));
      batchBoxCoxParams.emplace_back(std::move(params));
      break;
    }
    default: {
      return MAKE_ERR(
          strFormat("Unhandled node: %s", I.getDebugDesc().c_str()));
      break;
    }
    }
  }

  // Compile the graph.
  std::string recipeName;
  ASSIGN_VALUE_OR_RETURN_ERR(recipeName, getRecipeFile());
  CompilationAttribute compileParams[1];
  compileParams[0].type = VISUALIZATION;
  compileParams[0].u32 = 1;
  LOG(INFO) << "Compiling " << F->getName().str() << " to " << recipeName;
  auto start = TraceEvent::now();

  // TODO: add a TRACE_EVENT entry
  chk(synCompileGraph(compileParams, 1, recipeName.c_str()));
  auto duration = TraceEvent::now() - start;
  LOG(INFO) << "Compilation took " << duration / 1000.0 << " [ms]";
  chk(synDestroyGraph());

  return Expected<std::unique_ptr<CompiledFunction>>(
      glow::make_unique<HabanaFunction>(runtime::RuntimeBundle::create(*F),
                                        recipeName, F));
}

static bool isQuantizedType(ElemKind kind) {
  return kind == ElemKind::Int8QTy || kind == ElemKind::Int16QTy ||
         kind == ElemKind::Int32QTy;
}

bool HabanaBackend::isOpSupported(const NodeInfo &NI) const {
  if (NI.getKind() == Kinded::Kind::SaveNodeKind) {
    return true;
  }

  if (isQuantizedType(NI.getOutElemTy(0))) {
    switch (NI.getKind()) {
    case Kinded::Kind::AddNodeKind:
    case Kinded::Kind::AvgPoolNodeKind:
    case Kinded::Kind::ConvolutionNodeKind:
    case Kinded::Kind::DivNodeKind:
    case Kinded::Kind::FullyConnectedNodeKind:
    case Kinded::Kind::HabanaConvolutionNodeKind:
    case Kinded::Kind::HabanaConvolutionAddNodeKind:
    case Kinded::Kind::HabanaFullyConnectedNodeKind:
    case Kinded::Kind::MatMulNodeKind:
    case Kinded::Kind::MaxPoolNodeKind:
    case Kinded::Kind::MulNodeKind:
    case Kinded::Kind::QuantizeNodeKind:
    case Kinded::Kind::ReluNodeKind:
    case Kinded::Kind::ReshapeNodeKind:
    case Kinded::Kind::SliceNodeKind:
    case Kinded::Kind::SplatNodeKind:
    case Kinded::Kind::SubNodeKind:
    case Kinded::Kind::TileNodeKind:
    case Kinded::Kind::ConcatNodeKind:
    case Kinded::Kind::TransposeNodeKind:
      return true;
    case Kinded::Kind::RescaleQuantizedNodeKind:
      return NI.allInputsAndOutputsHaveSameElemKind(
          {ElemKind::Int8QTy, ElemKind::Int16QTy});
    default:
      return false;
    }
  }

  switch (NI.getKind()) {
  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::AvgPoolNodeKind:
  case Kinded::Kind::BatchBoxCoxNodeKind:
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::BatchedAddNodeKind:
  case Kinded::Kind::BatchedReduceAddNodeKind:
  case Kinded::Kind::ConcatNodeKind:
  case Kinded::Kind::DequantizeNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::HabanaFullyConnectedNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::MatMulNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MaxPoolNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ReshapeNodeKind:
  case Kinded::Kind::SaveNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::SliceNodeKind:
  case Kinded::Kind::SoftMaxNodeKind:
  case Kinded::Kind::SplatNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::TileNodeKind:
  case Kinded::Kind::TransposeNodeKind:
  case Kinded::Kind::LengthsRangeFillNodeKind:
  case Kinded::Kind::LengthsSumNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
  case Kinded::Kind::SparseLengthsWeightedSumNodeKind:
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind:
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind:
  case Kinded::Kind::LocalResponseNormalizationNodeKind:
    return true;
  case Kinded::Kind::BatchOneHotNodeKind:
    return NI.allInputsAndOutputsHaveSameElemKind({ElemKind::Int32ITy});
  case Kinded::Kind::GatherNodeKind:
    // Gather is technically supported but currently appears to trigger bugs in
    // large models.
  default:
    return false;
  }

  return false;
}

bool HabanaBackend::shouldLower(const Node *N) const {
  switch (N->getKind()) {
  case Kinded::Kind::BatchBoxCoxNodeKind:
  case Kinded::Kind::BatchMatMulNodeKind:
  case Kinded::Kind::ConvolutionNodeKind:
  case Kinded::Kind::FullyConnectedNodeKind:
  case Kinded::Kind::FusedRowwiseQuantizedSparseLengthsSumNodeKind:
  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::SparseLengthsSumNodeKind:
  case Kinded::Kind::TileNodeKind:
    return false;
  default:
    return true;
  }
}

namespace {
/// Separate Slice into several Slice nodes that only slice in one dimension
/// since the Habana slice_axis can only slice in one axis at once. For example,
///
/// Slice with input shape {2, 3, 4}, output shape {1, 1, 1}, offset {0, 1, 2}
///
/// becomes
///
/// Slice with input shape {2, 3, 4}, output shape {1, 3, 4}, offset {0, 0, 0}
/// Slice with input shape {1, 3, 4}, output shape {1, 1, 4}, offset {0, 1, 0}
/// Slice with input shape {1, 1, 4}, output shape {1, 1, 1}, offset {0, 0, 2}
bool separateSlice(Function *F, SliceNode &slice) {
  TypeRef iTy = slice.getInput().getType();
  TypeRef oTy = slice.getResult().getType();

  auto iDims = iTy->dims();
  auto oDims = oTy->dims();

  // Check if the Slice modifies more than one dimension. If so, do nothing.
  // This is not an optimisation; transformPostLowering() will loop infinitely
  // without this by continuously replacing a Slice that modifies only one
  // dimension with an identical copy.
  size_t dimsChanged = 0;

  for (size_t i = 0, e = iDims.size(); i < e; ++i) {
    if (iDims[i] != oDims[i]) {
      dimsChanged++;
    }
  }

  if (dimsChanged <= 1) {
    return false;
  }

  auto start = slice.getStart();

  // This will be modified to create the correct output Type for each new Slice
  // that will be created. It should start out equal to the dims of the input
  // Type and only one element should change between the creation of two
  // successive Slice nodes (since each new Slice only operates in one axis).
  std::vector<size_t> sliceDims(iDims.begin(), iDims.end());
  // This will be modified to create the correct offset/start for each new
  // Slice. Initialize everything to zero because only one element should be
  // non-zero at a time (since each new Slice only operates in one axis).
  std::vector<size_t> sliceStart(start.size(), 0);

  // Set this to the input of the Slice being replaced to start with. This is
  // used to connect all the new Slices together.
  NodeValue prevNode = slice.getInput();

  for (size_t i = 0, e = iDims.size(); i < e; ++i) {
    // If the input and output dimension are equal, do nothing.
    if (iDims[i] != oDims[i]) {
      // Copy the i-th output dimension into sliceDims. sliceDims is then
      // identical to the shape of prevNode except for the i-th dimension (the
      // one being sliced).
      sliceDims[i] = oDims[i];
      // Copy the offset for the i-th dimension for the original Slice; the rest
      // should all be zero since those dimensions aren't being sliced.
      sliceStart[i] = start[i];

      // Create the new Slice node.
      auto newSliceName = slice.getName().str() + strFormat(".%zd", i);
      auto newSliceTy = F->getParent()->uniqueTypeWithNewShape(iTy, sliceDims);
      auto *newSlice =
          new SliceNode(newSliceName, newSliceTy, prevNode, sliceStart);
      F->addNode(newSlice);

      // Reset sliceStart to all zeros for next iteration.
      sliceStart[i] = 0;
      // Set prevNode to the output of the newly created to preserve the loop
      // invariant so that the next Slice receives input from this one.
      prevNode = newSlice->getResult();
    }
  }

  // Replace all uses of the original Slice with the output of the newly created
  // chain of Slices. DCE will remove the original Slice.
  slice.getResult().replaceAllUsesOfWith(prevNode);
  return true;
}

/// Fuse conv followed by relu into a single FusedConvRelu node.
bool fuseConvRelu(Function *F, HabanaConvolutionNode &conv, ReluNode &relu) {
  // Convolution should have only a single use.
  if (!conv.getResult().hasOneUse()) {
    return false;
  }

  auto fusedOpName =
      conv.getName().str() + "." + relu.getName().str() + ".fused";
  auto *fusedNode = new HabanaConvolutionNode(
      fusedOpName, relu.getResult().getType(), conv.getInput(),
      conv.getFilter(), conv.getBias(), conv.getKernels(), conv.getStrides(),
      conv.getPads(), conv.getGroup(), /*doRelu=*/true);
  F->addNode(fusedNode);

  relu.getResult().replaceAllUsesOfWith(fusedNode);
  return true;
}

/// Fuse fully connected followed by relu into a single FusedFCRelu node.
bool fuseFCRelu(Function *F, HabanaFullyConnectedNode &fc, ReluNode &relu) {
  // fully connected should have only a single use.
  if (!fc.getResult().hasOneUse()) {
    return false;
  }

  auto fusedOpName = fc.getName().str() + "." + relu.getName().str() + ".fused";
  auto *fusedNode = new HabanaFullyConnectedNode(
      fusedOpName, relu.getResult().getType(), fc.getInput(), fc.getWeights(),
      fc.getBias(), /*doRelu=*/true);
  F->addNode(fusedNode);

  relu.getResult().replaceAllUsesOfWith(fusedNode);
  return true;
}

/// Fuse any sum of Conv + Nodse into Node -> FusedConvAdd.
bool fuseConvAdd(Function *F, AddNode &add) {
  // Perform this transformation:
  // Conv --> Add
  //           ^     -->    Node --> HabanaConvolutionAdd
  //           |
  // Node ------
  auto *lhs = llvm::dyn_cast<HabanaConvolutionNode>(add.getLHS());
  auto *rhs = llvm::dyn_cast<HabanaConvolutionNode>(add.getRHS());

  // Pick the Conv with only a single use (the Add) to fuse because
  // there is no output of the fused node that can provide only the
  // convolution output to other users.
  HabanaConvolutionNode *singleUseConv;
  NodeValue otherNode;

  if (lhs && lhs->getResult().hasOneUse()) {
    singleUseConv = lhs;
    otherNode = add.getRHS();
  } else if (rhs && rhs->getResult().hasOneUse()) {
    singleUseConv = rhs;
    otherNode = add.getLHS();
  } else {
    // If neither addend is a Convolution with only one use, this
    // transformation cannot be applied, so return false.
    return false;
  }
  // Replace the Conv with a single use and the Add with FusedConvAdd.
  // Feed the inputs of the single use Conv into the FusedConvAdd and feed
  // in the output of the other Node as its addend input.
  auto fusedNodeName =
      singleUseConv->getName().str() + "." + add.getName().str() + ".fused";
  auto *fusedConvAddNode = F->addNode(new HabanaConvolutionAddNode(
      fusedNodeName, add.getResult().getType(), otherNode,
      singleUseConv->getInput(), singleUseConv->getFilter(),
      singleUseConv->getBias(), singleUseConv->getKernels(),
      singleUseConv->getStrides(), singleUseConv->getPads(),
      singleUseConv->getGroup(), /*doRelu=*/false));

  // Replace all uses of the Add with FusedConvAdd. DCE will remove the
  // orphaned Conv -> Add subgraph.
  add.getResult().replaceAllUsesOfWith(fusedConvAddNode);

  return true;
}

/// Fuse HabanaConvolutionAdd with Relu into a HabanaConvolutionAdd with
/// doRelu set to true.
bool fuseConvAddRelu(Function *F, HabanaConvolutionAddNode &convAdd,
                     ReluNode &relu) {
  // If the HabanaConvolutionAdd has more than one use, this transformation
  // cannot be applied because the input to the ReLU node is not exposed by
  // the node when doRelu is set to true.
  if (!convAdd.getResult().hasOneUse()) {
    return false;
  }

  auto fusedNodeName =
      convAdd.getName().str() + "." + relu.getName().str() + ".fused";
  auto *fusedNode = new HabanaConvolutionAddNode(
      fusedNodeName, relu.getResult().getType(), convAdd.getAddend(),
      convAdd.getInput(), convAdd.getFilter(), convAdd.getBias(),
      convAdd.getKernels(), convAdd.getStrides(), convAdd.getPads(),
      convAdd.getGroup(), /*doRelu=*/true);
  F->addNode(fusedNode);

  relu.getResult().replaceAllUsesOfWith(fusedNode);
  return true;
}

/// Surround a Tile node with Reshapes in order to be able to perform the tiling
/// operation (which can only be done with 4D tensors for now). The Reshapes
/// make sure that the new, three-node subgraph has the same input and output
/// types as the Tile that it replaced.
bool surroundTileWithReshapes(Function *F, TileNode &tile) {
  bool changed = false;
  auto iDims = tile.getInput().getType()->dims();
  auto oDims = tile.getResult().getType()->dims();

  unsigned nDims = iDims.size();
  // SYN_MAX_TENSOR_DIM is 5, and the -1 is for the batch dimension.
  unsigned missingDims = SYN_MAX_TENSOR_DIM - 1 - nDims;

  if (missingDims) {
    // Create types identical to the input and output of the Tile node whose
    // shapes are padded with 1s to make sure they have SYN_MAX_TENSOR_DIM - 1
    // dimensions.
    std::vector<size_t> inputShapeWithOnes(missingDims, 1);
    std::vector<size_t> outputShapeWithOnes(missingDims, 1);

    inputShapeWithOnes.insert(inputShapeWithOnes.end(), iDims.begin(),
                              iDims.end());
    outputShapeWithOnes.insert(outputShapeWithOnes.end(), oDims.begin(),
                               oDims.end());

    // Create a Reshape to reshape the input to the new input type
    // (old input type with a shape padded with 1s).
    auto *reshapeIn = F->createReshape(tile.getName().str() + ".reshape.in",
                                       tile.getInput(), inputShapeWithOnes);
    // Create a new Tile that tiles the reshaped input. The axis needs to be
    // shifted by the number of 1s added to the original input shape.
    auto *newTile = F->createTile(tile.getName(), reshapeIn, tile.getCount(),
                                  tile.getAxis() + missingDims);
    // Create a Reshape to reshape the tiled output to the original output type.
    auto *reshapeOut =
        F->createReshape(tile.getName().str() + ".reshape.out", newTile,
                         tile.getResult().getType()->dims());

    // Replace all uses of the old Tile with reshapeOut. The old Tile will be
    // removed by DCE.
    tile.getResult().replaceAllUsesOfWith(reshapeOut);
    changed = true;
  }

  return changed;
}

} // namespace

Expected<bool> HabanaBackend::transformPostLowering(
    Function *F, CompilationContext &cctx,
    const glow::runtime::DeviceInfo *devInfo) const {
  LOG_SCOPE(F->getLogContext(), "HabanaBackend::transformPostLowering")

  bool changed = false;
  for (auto &node : F->getNodes()) {
    // Separate any Slice nodes into several that only slice in one dimension
    // at a time.
    if (auto *slice = llvm::dyn_cast<SliceNode>(&node)) {
      changed |= separateSlice(F, *slice);
    }

    // Surround any Tile nodes with Reshapes since Tiles can only be performed
    // on 4D shapes for now.
    if (auto *tile = llvm::dyn_cast<TileNode>(&node)) {
      changed |= surroundTileWithReshapes(F, *tile);
    }

    // Fuse Convolution followed by relu.
    if (auto *relu = llvm::dyn_cast<ReluNode>(&node)) {
      if (auto *conv = llvm::dyn_cast<HabanaConvolutionNode>(
              relu->getInput().getNode())) {
        changed |= fuseConvRelu(F, *conv, *relu);
        continue;
      } else if (auto *convAdd = llvm::dyn_cast<HabanaConvolutionAddNode>(
                     relu->getInput().getNode())) {
        changed |= fuseConvAddRelu(F, *convAdd, *relu);
      }
    }
    // Fuse any Conv, Node -> Add sequences into Node -> FusedConvAdd.
    if (auto *add = llvm::dyn_cast<AddNode>(&node)) {
      changed |= fuseConvAdd(F, *add);
      continue;
    }

    // Replace FC with HabanaFC that understands weight + bias.
    if (auto *FC = llvm::dyn_cast<FullyConnectedNode>(&node)) {
      auto *weights =
          F->createTranspose("weight_transpose", FC->getWeights(), {1, 0});
      auto *NF = F->addNode(
          new HabanaFullyConnectedNode(FC->getName(), FC->getResult().getType(),
                                       FC->getInput(), weights, FC->getBias(),
                                       /*doRelu=*/false));
      FC->getResult().replaceAllUsesOfWith(NF);
      changed = true;
      continue;
    }

    // Fuse fully connected followed by relu.
    if (auto *relu = llvm::dyn_cast<ReluNode>(&node)) {
      if (auto *fc = llvm::dyn_cast<HabanaFullyConnectedNode>(
              relu->getInput().getNode())) {
        changed |= fuseFCRelu(F, *fc, *relu);
        continue;
      }
    }

    // Replace Conv with HabanaConv for better control over code generation.
    if (auto *conv = llvm::dyn_cast<ConvolutionNode>(&node)) {
      // Transpose filter into order expected by Habana backend (HWCN)
      auto *TF =
          F->createTranspose((conv->getName()).str() + ".filter.transpose",
                             conv->getFilter(), {1, 2, 3, 0});
      auto *NC = F->addNode(new HabanaConvolutionNode(
          conv->getName(), conv->getResult().getType(), conv->getInput(), TF,
          conv->getBias(), conv->getKernels(), conv->getStrides(),
          conv->getPads(), conv->getGroup(), /*doRelu=*/false));
      conv->getResult().replaceAllUsesOfWith(NC);
      changed = true;
      continue;
    }
  }

  return changed;
}

bool HabanaBackend::isVersionBiggerEqualTo(std::string versionToCompare) {
  const char *habanaVersion = synGetVersion();
  int myMajor, myMinor, myPath;
  sscanf(habanaVersion, "%d.%d.%d", &myMajor, &myMinor, &myPath);
  int toMajor, toMinor, toPath;
  sscanf(versionToCompare.c_str(), "%d.%d.%d", &toMajor, &toMinor, &toPath);
  if (toMajor <= myMajor) {
    if (toMinor <= myMinor) {
      if (toPath <= myPath) {
        return true;
      }
    }
  }
  return false;
}

unsigned HabanaBackend::numDevices() {
  std::ifstream devices("/proc/bus/pci/devices");
  std::string device;
  unsigned count = 0;
  while (std::getline(devices, device)) {
    if (device.find("habanalabs") != std::string::npos) {
      count++;
    }
  }
  return count;
}

std::vector<unsigned> HabanaBackend::scanDeviceIDs() {
  std::vector<unsigned> deviceIDs(HabanaBackend::numDevices());
  std::iota(std::begin(deviceIDs), std::end(deviceIDs), 0);
  return deviceIDs;
}
