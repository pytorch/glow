/**
 * Copyright (c) 2017-present, Facebook, Inc.
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
#include "BackendTestUtils.h"

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Partitioner/Partitioner.h"

#include <algorithm>
#include <random>

#include "gtest/gtest.h"

#include "llvm/Support/CommandLine.h"

using namespace glow;

namespace {
llvm::cl::OptionCategory recSysTestCat("RecSys Category");

llvm::cl::opt<unsigned> bottomMLPIntermediateDimOpt(
    "bottom-mlp-intermediate-dim",
    llvm::cl::desc("Intermediate dim for the bottom MLP."), llvm::cl::Optional,
    llvm::cl::init(1024), llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> topMLPIntermediateDimOpt(
    "top-mlp-intermediate-dim",
    llvm::cl::desc("Intermediate dim for the top MLP."), llvm::cl::Optional,
    llvm::cl::init(1024), llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> miniBatchOpt("mini-batch", llvm::cl::desc("Minibatch."),
                                     llvm::cl::Optional, llvm::cl::init(16),
                                     llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> embeddingDimOpt("embedding-dim",
                                        llvm::cl::desc("Embedding dim."),
                                        llvm::cl::Optional, llvm::cl::init(64),
                                        llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> denseDimOpt("dense-dim", llvm::cl::desc("Dense dim."),
                                    llvm::cl::Optional, llvm::cl::init(800),
                                    llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> numBottomMLPLayersOpt(
    "num-bottom-mlp-layers", llvm::cl::desc("Number of bottom MLP layers."),
    llvm::cl::Optional, llvm::cl::init(3), llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> numTopMLPLayersOpt(
    "num-top-mlp-layers", llvm::cl::desc("Number of top MLP layers."),
    llvm::cl::Optional, llvm::cl::init(3), llvm::cl::cat(recSysTestCat));

llvm::cl::list<unsigned> tableSizesOpt(
    "embedding-table-sizes",
    llvm::cl::desc("Comma-separated list of embedding table sizes."),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(recSysTestCat));

llvm::cl::list<unsigned> tableCountsOpt(
    "embedding-table-counts",
    llvm::cl::desc("Comma-separated list of embedding table counts, "
                   "corresponding to a count for each size listed in "
                   "embedding-table-sizes."),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> deviceMemCapacityOpt(
    "device-mem-capacity",
    llvm::cl::desc("Device memory capacity. Default is 6 MB."),
    llvm::cl::Optional, llvm::cl::init(1024 * 1024 * 6),
    llvm::cl::cat(recSysTestCat));

llvm::cl::opt<unsigned> numDevicesOpt(
    "num-devices", llvm::cl::desc("Number of devices to use for partitioning."),
    llvm::cl::Optional, llvm::cl::init(6), llvm::cl::cat(recSysTestCat));
} // namespace

/// Fills the tensor \p H with some stable random data with the seed \p seed
/// and the range [-scale .. scale].
static void fillStableRandomData(Handle<float> H, size_t seed,
                                 float scale = 1) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = scale * (float((int(i * 1921 + seed) % 100) - 50) / 50);
  }
}

/// Fills the tensor \p H with some stable random integers with the seed \p
/// seed and the range [0, scale).
template <typename T>
static void fillStableRandomIndex(Handle<T> H, size_t seed, size_t min = 0,
                                  size_t max = 10) {
  for (size_t i = 0, e = H.size(); i < e; i++) {
    H.raw(i) = min + (int(i * 1921 + seed) % (max - min));
  }
}
template void fillStableRandomIndex(Handle<int64_t> Handle, size_t seed,
                                    size_t min, size_t max);
template void fillStableRandomIndex(Handle<int32_t> Handle, size_t seed,
                                    size_t min, size_t max);

/// Sum of all elements in Tensor.
static size_t sumOfElements(Handle<int32_t> H) {
  size_t sum = 0;
  for (size_t i = 0, e = H.size(); i < e; i++) {
    sum += H.raw(i);
  }
  return sum;
}

/// Tests a simplified Recommendation System model.
///
/// The RecSys model has four components:
///    * An initial Multilayer Perceptron acting in the inputs.
///    * Some number of Sparse Features: SparseLengthSum nodes acting on
///      embedding tables (see https://caffe2.ai/docs/sparse-operations.html).
///    * An interaction layer bringing together the output for hte top MLP and
///      the sparse features.
///    * A final MLP acting on the result of the interaction.
///
/// The final result is a float indicating the strength of the recommendation.
///
///
///              +------+
///              |Output|
///              +--^---+
///                 |
///             +---+---+
///             |  TOP  |
///             |       |
///             |  MLP  |
///             +---^---+
///                 |
///                 |
///         +-------+--------+
///         |   Interaction  <---------+
///   +----->                <---+     |
///   |     +--------^-----^-+   |     |
///   |              |     |     |     |
/// +--+----+      +-+-+ +-+-+ +-+-+ +-+-+
/// | Bottom|      |SLS| |SLS| |SLS| |SLS|
/// |       |      +---+ +---+ +---+ +---+
/// |  MLP  |          Sparse Features
/// +---^---+
///     |
/// +---+---+
/// | Input |
/// +-------+
///
class RecommendationSystemTest : public BackendTest {
public:
protected:
  PlaceholderBindings bindings_;

  // Test Config:
  size_t miniBatch;
  size_t embeddingDim;
  size_t denseDim;
  std::vector<size_t> tableSizes;

  // Partitioner config:
  uint64_t deviceMemCapacity;
  size_t numDevices;

  // Result handle.
  Placeholder *result{nullptr};

  void SetUp() override {
    /// Test configuration, tweak here:
    miniBatch = miniBatchOpt;
    embeddingDim = embeddingDimOpt;
    denseDim = denseDimOpt;

    if (!tableSizesOpt.empty()) {
      if (!tableCountsOpt.empty()) {
        CHECK_EQ(tableSizesOpt.size(), tableCountsOpt.size())
            << "Embedding table sizes and counts must be same length.";
        for (size_t i = 0, e = tableSizesOpt.size(); i < e; i++) {
          for (size_t j = 0, f = tableCountsOpt[i]; j < f; j++) {
            tableSizes.push_back(tableSizesOpt[i]);
          }
        }
      } else {
        tableSizes =
            std::vector<size_t>(tableSizesOpt.begin(), tableSizesOpt.end());
      }
      // Stable randomization of the order of the tables.
      std::shuffle(tableSizes.begin(), tableSizes.end(), std::mt19937());
    } else {
      tableSizes = {8000, 6000, 7000, 9000, 12000,
                    8000, 6000, 7000, 9000, 12000};
    }

    deviceMemCapacity = deviceMemCapacityOpt;

    numDevices = numDevicesOpt;
  }

  void TearDown() override {
    EE_.clear();
    mod_.clear();
    result = nullptr;
    bindings_.clear();
  }

  /// Returns a new Constant, of the provided \p type and \p dims initialized
  /// with random data.
  static Constant *createRandomizedConstant(Module &mod_, TypeRef type,
                                            llvm::ArrayRef<size_t> dims,
                                            llvm::StringRef name,
                                            float min = 0.0, float max = 1.0) {
    auto *c =
        mod_.createConstant(mod_.uniqueTypeWithNewShape(type, dims), name);

    switch (type->getElementType()) {
    case ElemKind::FloatTy: {
      c->getHandle<float>().randomize(min, max, mod_.getPRNG());
      break;
    }
    case ElemKind::Float16Ty: {
      c->getHandle<float16_t>().randomize(min, max, mod_.getPRNG());
      break;
    }
    case ElemKind::Int32QTy: {
      c->getHandle<int32_t>().randomize(-INT32_MAX, INT32_MAX, mod_.getPRNG());
      break;
    }
    case ElemKind::Int8QTy: {
      c->getHandle<int8_t>().randomize(-128, 127, mod_.getPRNG());
      break;
    }
    case ElemKind::UInt8FusedQTy: {
      c->getHandle<uint8_t>().randomize(0, 255, mod_.getPRNG());
      break;
    }
    default:
      LOG(FATAL) << "Unsupported type: " << type->getElementName().str();
    }

    return c;
  }

  /// Returns a new Constant of type UInt8FusedQTy with fused rowwise
  /// quantization scales and offsets (i.e. the last 8 bytes of each row
  /// contains the scale and offset).
  static Constant *createRandomFusedRowwiseQuantizedConstant(
      Module &mod_, llvm::ArrayRef<size_t> dims, llvm::StringRef name) {
    Constant *c = createRandomizedConstant(
        mod_, mod_.uniqueType(ElemKind::UInt8FusedQTy, {1}, 1, 0),
        {dims[0], dims[1] + 8}, name);

    auto *dbP = c->getPayload().getUnsafePtr();
    const size_t outWidth = c->dims()[1];
    for (unsigned j = 0, e = c->dims()[0]; j < e; j++) {
      // Now set the scale/offset at the end of each row.
      char *currRowScaleOffsetPtr =
          dbP + (j + 1) * outWidth - 2 * sizeof(float);

      // range (0, 255) -> (-0.1, 0.1)
      float scale = 1.0f / 1275;
      float offset = -0.1;

      memcpy(currRowScaleOffsetPtr, &scale, sizeof(float));
      memcpy(currRowScaleOffsetPtr + sizeof(float), &offset, sizeof(float));
    }

    return c;
  }

  /// Creates a Multi-layer perceptron network consisting of start & end FCs
  /// with \p intermediateLayers hidden layers.
  ///   * All weights and biases are random.
  ///   * All internal activations are RELU.
  ///   * Parent node \p N_ has output dimension \p inputDim.
  ///   * Hidden layers have dimension of \p intDim * intDim.
  ///   * Output layer has output dimension \p outputDim.
  static NodeValue createMLP(Module &mod_, Function *F_, Node *N_,
                             size_t inputDim, size_t intDim, size_t outputDim,
                             size_t intermediateLayers) {
    assert(intermediateLayers > 0);

    // Type object for the internal layers.
    // Note: dimension argument is a placeholder and will get filled out by each
    // createRandomizedConstant invocation.
    auto internalType = mod_.uniqueType(ElemKind::FloatTy, {1});

    /// Initial
    auto *initial_bias = createRandomizedConstant(mod_, internalType, {intDim},
                                                  "initial_bias", 0, 0.00001);
    auto *initial_weight = createRandomizedConstant(
        mod_, internalType, {inputDim, intDim}, "initial_weight", -0.03, 0.03);

    FullyConnectedNode *initial_layer = F_->createFullyConnected(
        "dense", N_, initial_weight,
        initial_bias); // Output is size {MB, intermediate dim}
    auto *initial_relu = F_->createRELU("relu1", initial_layer);

    auto *last = initial_relu;

    /// Intermediate
    for (unsigned i = 0; i < intermediateLayers; ++i) {

      auto *intermediate_bias = createRandomizedConstant(
          mod_, internalType, {intDim}, "intermediate_bias", 0, 0.00001);
      auto *intermediate_weight =
          createRandomizedConstant(mod_, internalType, {intDim, intDim},
                                   "intermediate_weight", -0.03, 0.03);

      FullyConnectedNode *intermediate_layer = F_->createFullyConnected(
          "dense", last, intermediate_weight,
          intermediate_bias); // Output is size {MB, intDim}
      last = F_->createRELU("relu2", intermediate_layer);
    }

    /// End
    auto *end_bias = createRandomizedConstant(mod_, internalType, {outputDim},
                                              "end_bias", 0, 0.00001);
    auto *end_weight = createRandomizedConstant(
        mod_, internalType, {intDim, outputDim}, "end_weight", -0.001, 0.003);

    FullyConnectedNode *end_layer = F_->createFullyConnected(
        "dense", last, end_weight, end_bias); // Output is size {MB, embDim}

    auto *RN = F_->createRELU("relu3", end_layer);

    return RN->getResult();
  }

  /// Creates a rowwise quantized Multi-layer perceptron network consisting of
  /// start & end FCs with \p intermediateLayers hidden layers.
  ///   * All weights and biases are random. Weights are Int8Q (rowwise), biases
  ///     are Int32.
  ///   * All internal activations are RELU, however the final layer has no
  ///     activation attached.
  ///   * Parent node \p N_ has output dimension \p inputDim int float.
  ///   * Hidden layers have dimension of \p intDim * intDim int Int8Q
  ///     (rowwise).
  ///   * Output layer has output dimension \p outputDim in float.
  ///
  /// Quantized MLPs use RowwiseQuantizedFullyConnected Nodes, which expect:
  ///   * weights to be Float32 and convert to Int8 fused rowwise quantized
  ///     Tensors internally
  ///   * Biases are Int32 quantized.
  static NodeValue createQuantizedMLP(Module &mod_, Function *F_, NodeValue N_,
                                      size_t inputDim, size_t intDim,
                                      size_t outputDim,
                                      size_t intermediateLayers) {
    // Must have intermediate layers.
    assert(intermediateLayers > 0);

    const size_t minibatchSize = N_.dims()[0];

    // Type objects for the internal types.
    // Note: dimension argument is a placeholder and will get filled out by each
    // createRandomizedConstant invocation.
    auto internalTypeF = mod_.uniqueType(ElemKind::FloatTy, {1});
    auto internalTypeQ = mod_.uniqueType(ElemKind::Int8QTy, {1}, 1, 0);
    auto internalBiasType = mod_.uniqueType(ElemKind::Int32QTy, {1}, 1e-11, 0);

    auto *start = F_->createQuantize(
        "mlp_quant", N_, mod_.uniqueTypeWithNewShape(internalTypeQ, N_.dims()));

    /// Initial.
    auto *initial_bias = createRandomizedConstant(mod_, internalBiasType,
                                                  {intDim}, "initial_bias");
    auto *initial_weight = createRandomizedConstant(
        mod_, internalTypeF, {inputDim, intDim}, "initial_weight", -0.03, 0.03);

    // Output is size {MB, intermediatDim}
    Node *initial_layer = F_->createRowwiseQuantizedFullyConnected(
        "dense", start, initial_weight, initial_bias,
        mod_.uniqueTypeWithNewShape(internalTypeQ, {minibatchSize, intDim}),
        quantization::Asymmetric,
        /* transposeWeight */ true);

    Node *initial_relu = F_->createRELU("initial_relu", initial_layer);

    /// Intermediate
    auto *last = initial_relu;
    for (unsigned i = 0; i < intermediateLayers; ++i) {
      auto *intermediate_bias = createRandomizedConstant(
          mod_, internalBiasType, {intDim}, "intermediate_bias");
      auto *intermediate_weight =
          createRandomizedConstant(mod_, internalTypeF, {intDim, intDim},
                                   "intermediate_weight", -0.03, 0.03);

      Node *intermediate_layer = F_->createRowwiseQuantizedFullyConnected(
          "dense", last, intermediate_weight, intermediate_bias,
          mod_.uniqueType(ElemKind::Int8QTy, {minibatchSize, intDim}, 1.0, 0),
          quantization::Asymmetric,
          /* transposeWeight */ true); // Output is size {MB, intDim}
      last = F_->createRELU("intermediate_relu", intermediate_layer);
    }

    /// End
    auto *end_bias = createRandomizedConstant(mod_, internalBiasType,
                                              {outputDim}, "end_bias");
    auto *end_weight = createRandomizedConstant(
        mod_, internalTypeF, {intDim, outputDim}, "end_weight", -0.03, 0.03);

    // Output is size {MB, embDim}
    auto *end_layer = F_->createRowwiseQuantizedFullyConnected(
        "dense", last, end_weight, end_bias,
        mod_.uniqueTypeWithNewShape(internalTypeQ, {minibatchSize, outputDim}),
        quantization::Asymmetric,
        /* transposeWeight */ true);

    auto *RN = F_->createRELU("relu", end_layer);
    auto *DQN = F_->createDequantize("mlp_dequant", RN);

    return DQN->getResult();
  }

  /// Creates a number of Sparse tables (FP32 or Int8Q), the Indices lookup and
  /// the SpareLengthsSum Node tying it together.
  static void createSparseEmbeddings(
      Module &mod_, PlaceholderBindings &bindings_, Function *F_,
      llvm::ArrayRef<Placeholder *> lengths, llvm::ArrayRef<size_t> embSizes,
      size_t embDim, std::vector<NodeValue> &embeddings, bool quantizeSLWS) {
    auto internalTypeF = mod_.uniqueType(ElemKind::FloatTy, {1});

    for (unsigned int i = 0; i < lengths.size(); i++) {
      fillStableRandomIndex(
          bindings_.allocate(lengths[i])->getHandle<int32_t>(), 2011, 90, 111);

      size_t sum =
          sumOfElements(bindings_.get(lengths[i])->getHandle<int32_t>());
      auto *indices = mod_.createPlaceholder(
          ElemKind::Int64ITy, {sum}, "indices" + std::to_string(i), false);
      fillStableRandomIndex(bindings_.allocate(indices)->getHandle<int64_t>(),
                            2001, 0, embSizes[i]);

      // output is size {MB, embDim}
      if (quantizeSLWS) {
        Constant *data = createRandomFusedRowwiseQuantizedConstant(
            mod_, {embSizes[i], embDim}, "data" + std::to_string(i));
        embeddings[i] = F_->createFusedRowwiseQuantizedSparseLengthsSum(
            "RQSLWS" + std::to_string(i), data, indices, lengths[i]);
      } else {
        Constant *data =
            createRandomizedConstant(mod_, internalTypeF, {embSizes[i], embDim},
                                     "data" + std::to_string(i));
        embeddings[i] = F_->createSparseLengthsSum("sls" + std::to_string(i),
                                                   data, indices, lengths[i]);
      }
    }
  }

  /// Creates a number of Sparse tables (FP32 or Int8Q), the Indices lookup and
  /// the SpareLengthsSum Node tying it together.
  static void createSparseWeightedGatherEmbeddings(
      Module &mod_, PlaceholderBindings &bindings_, Function *F_,
      llvm::ArrayRef<Placeholder *> lengths, llvm::ArrayRef<size_t> tableSizes,
      size_t embeddingDim, std::vector<NodeValue> &embeddings,
      bool quantizeSLWS, uint32_t weightsSize = 1000) {
    for (size_t i = 0; i < lengths.size(); i++) {
      fillStableRandomIndex(
          bindings_.allocate(lengths[i])->getHandle<int32_t>(), 2011, 90, 111);

      size_t sum =
          sumOfElements(bindings_.get(lengths[i])->getHandle<int32_t>());
      auto *indices = mod_.createPlaceholder(
          ElemKind::Int64ITy, {sum}, "indices" + std::to_string(i), false);
      fillStableRandomIndex(bindings_.allocate(indices)->getHandle<int64_t>(),
                            2001, 0, tableSizes[i]);

      // Should be able to pass weights - fix later. Currently, just a
      // randomized constant.
      Constant *weightsConst = createRandomizedConstant(
          mod_, mod_.uniqueType(ElemKind::FloatTy, {weightsSize}),
          {weightsSize}, "weights" + std::to_string(i), 1.0f, 1.0000001f);

      auto *weightIndices =
          mod_.createPlaceholder(ElemKind::Int32ITy, {sum},
                                 "weight_indices" + std::to_string(i), false);
      fillStableRandomIndex(
          bindings_.allocate(weightIndices)->getHandle<int32_t>(), 2001, 0,
          weightsSize - 1);

      auto *weights = F_->createGather("weight_gather" + std::to_string(i),
                                       weightsConst, weightIndices, 0);

      // output is size {MB, embeddingDim_}
      if (quantizeSLWS) {
        Constant *data = createRandomFusedRowwiseQuantizedConstant(
            mod_, {tableSizes[i], embeddingDim}, "data" + std::to_string(i));
        embeddings[i] = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            "RQSLWS" + std::to_string(i), data, weights, indices, lengths[i]);

      } else {
        Constant *data = createRandomizedConstant(
            mod_,
            mod_.uniqueType(ElemKind::FloatTy, {tableSizes[i], embeddingDim}),
            {tableSizes[i], embeddingDim}, "data" + std::to_string(i));
        embeddings[i] = F_->createSparseLengthsWeightedSum(
            "slws" + std::to_string(i), data, weights, indices, lengths[i]);
      }
    }
  }

  /// Builds a simple graph, returns back input var and save node through refs.
  static void createSimpleRecSysGraph(
      Module &mod_, PlaceholderBindings &bindings_, Function *F_,
      llvm::ArrayRef<Placeholder *> lengths, Placeholder *denseData,
      llvm::ArrayRef<size_t> embSizes, size_t embDim, llvm::StringRef funcName,
      bool quantizeSLWS, bool quantizeFC, bool gatherWeights) {
    EXPECT_EQ(lengths.size(), embSizes.size());

    // First Dense embedding
    fillStableRandomData(bindings_.allocate(denseData)->getHandle(), 2001,
                         0.001);
    NodeValue bottomMLP;
    if (quantizeFC) {
      bottomMLP = createQuantizedMLP(mod_, F_, denseData, denseData->dims()[1],
                                     bottomMLPIntermediateDimOpt, embDim,
                                     numBottomMLPLayersOpt);
    } else {
      bottomMLP =
          createMLP(mod_, F_, denseData, denseData->dims()[1],
                    bottomMLPIntermediateDimOpt, embDim, numBottomMLPLayersOpt);
    }

    // Sparse Embeddings
    std::vector<NodeValue> embeddings(lengths.size());
    if (gatherWeights) {
      createSparseWeightedGatherEmbeddings(mod_, bindings_, F_, lengths,
                                           embSizes, embDim, embeddings,
                                           quantizeSLWS);
    } else {
      createSparseEmbeddings(mod_, bindings_, F_, lengths, embSizes, embDim,
                             embeddings, quantizeSLWS);
    }

    // Interacting sparse and dense
    embeddings.push_back(bottomMLP);
    std::cout << "Number of embeddings concatenated: " << embeddings.size()
              << std::endl;
    auto *CN = F_->createConcat("concat", embeddings,
                                1); // Output is size {MB, embDim*n}
    auto *reshaped = F_->createReshape(
        "reshape", CN,
        {bottomMLP.dims()[0], embeddings.size(), embDim}); // {MB, n, embDim}
    auto *transposed = F_->createTranspose("transpose", reshaped,
                                           {0, 2, 1}); // {MB, embDim, n}
    auto *dot = F_->createBatchMatMul("dot_products", reshaped,
                                      transposed); // {MB, n, n}
    auto *reshapeDot =
        F_->createReshape("reshapeDot", dot,
                          {bottomMLP.dims()[0],
                           embeddings.size() * embeddings.size()}); // {MB, n^2}
    NodeValue interact = F_->createConcat("interact", {reshapeDot, bottomMLP},
                                          1); // {MB, n^2 + embDim}

    // MLP at the top
    Node *topMLP;
    if (quantizeFC) {
      topMLP = createQuantizedMLP(mod_, F_, interact, interact.dims()[1],
                                  topMLPIntermediateDimOpt, /* outputDim */ 1,
                                  numTopMLPLayersOpt);
    } else {
      topMLP = createMLP(mod_, F_, interact, interact.dims()[1],
                         topMLPIntermediateDimOpt, /* outputDim */ 1,
                         numTopMLPLayersOpt);
    }

    // Output
    auto *save = F_->createSave("save", topMLP);
    bindings_.allocate(save->getPlaceholder());

    return;
  }

  void testRecSys(bool quantizeSLWS, bool quantizeFC, bool convertToFP16,
                  bool gatherWeights, bool checkConcat = false) {
    // Create the tables.
    std::vector<Placeholder *> sparseLengths(tableSizes.size());
    for (unsigned int i = 0; i < sparseLengths.size(); i++) {
      sparseLengths[i] = mod_.createPlaceholder(
          ElemKind::Int32ITy, {miniBatch}, "SL" + std::to_string(i), false);
    }

    auto *denseData = mod_.createPlaceholder(
        ElemKind::FloatTy, {miniBatch, denseDim}, "denseData",
        false); // Dense dim can be anything

    // Generate the network.
    createSimpleRecSysGraph(mod_, bindings_, F_, sparseLengths, denseData,
                            tableSizes, embeddingDim, "main", quantizeSLWS,
                            quantizeFC, gatherWeights);
    SaveNode *result1 = llvm::cast<SaveNode>(F_->getNodeByName("save"));
    result = result1->getPlaceholder();

    Placeholder *concatPH = nullptr;
    if (checkConcat) {
      // Add an observer node after concat.
      auto *CN = F_->getNodeByName("concat");
      auto *saveConcat = F_->createSave("after_concat_data", CN);
      concatPH = saveConcat->getPlaceholder();
    }

    CompilationContext cctx;
    if (convertToFP16) {
      PrecisionConfiguration &precConfig = cctx.precisionConfig;
      precConfig.convertToFP16 = convertToFP16;
      precConfig.precisionModeKindSet.insert(
          Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind);
      precConfig.precisionModeKindSet.insert(
          Kinded::Kind::RowwiseQuantizedFullyConnectedNodeKind);
    }

    // Compile.
    EE_.compile(F_, cctx);

    // Run graph
    EE_.run(bindings_);

    if (checkConcat) {
      // Get result and verify.
      auto *resultTensor = bindings_.get(result);
      auto resultHandle = resultTensor->getHandle();

      EXPECT_EQ(resultTensor->size(), miniBatch);

      auto *concatT = bindings_.get(concatPH);
      auto concatH = concatT->getHandle();
      // Check that intermediate concat results didn't overflow.
      std::cout << "Intermediate concats" << std::endl;
      concatH.dump();
      for (int i = 0, e = concatH.size(); i < e; ++i) {
        EXPECT_LE(fabs(concatH.raw(i)), 100);
      }

      std::cout << "Result of prediction" << std::endl;
      std::cout << resultHandle.size() << std::endl;
      resultHandle.dump();
      for (int i = 0, e = resultHandle.size(); i < e; ++i) {
        EXPECT_GE(resultHandle.raw(i), 0.0);
      }
    }
  }

  /// Execute a graph of functions based on the given DAG.
  void executeDAG(DAGNode *G, Module &mod, PlaceholderBindings &bindings,
                  llvm::ArrayRef<Placeholder *> vars,
                  llvm::ArrayRef<Tensor *> inputs) {
    std::unordered_map<std::string, Function *> name2func;

    for (auto *F : mod.getFunctions()) {
      name2func[F->getName()] = F;
    }

    std::vector<DAGNode *> exeList;
    int endPt = 0;
    int curPt = 0;
    // The first node is always the dummy node.
    exeList.push_back(G);
    endPt++;
    while (curPt < endPt) {
      DAGNode *dag = exeList.at(curPt);
      // The root in a G is always a dummy function.
      if (curPt > 0) {
        ExecutionEngine EE{EE_.getBackend()->getBackendName()};
        Function *func = name2func[dag->name];
        EE.compile(CompilationMode::Infer, func);
        updateInputPlaceholders(bindings, vars, inputs);
        EE.run(bindings);
      }
      for (unsigned int i = 0, e = dag->children.size(); i < e; i++) {
        exeList.push_back(dag->children.at(i));
        endPt++;
      }
      curPt++;
    }
  }

  /// Create partitions to run and compare results.
  void runPartitionedGraph(size_t numDevices, size_t memSize,
                           Tensor referenceResult,
                           PlaceholderBindings &bindings) {

    assert(memSize > 0 && "Must set partitionerPerDeviceMemCapacity > 0.");
    assert(numDevices > 0 && "Must set partitionerNumDevices > 0.");
    auto backendName = EE_.getBackend()->getBackendName();
    std::cout << numDevices << " devices of size " << memSize << "\n";
    std::vector<DeviceInfo> devices(numDevices, {memSize, backendName});
    Partitioner myPartitioner(&mod_, devices);
    CompilationContext cctx;
    EXIT_ON_ERR(myPartitioner.Partition(cctx));

    DAGListTy myList = std::move(myPartitioner.getPartitionResult());
    std::cout << "Partitions = " << mod_.getFunctions().size() << std::endl;
    ASSERT_LE(mod_.getFunctions().size(), numDevices);
    ASSERT_EQ(myList.size(), 1);
    DAG &dag = myList.front();

    // Run the paritioned graph and compare the results.
    bindings.allocate(mod_.getPlaceholders());
    bindings.allocate(mod_.getPlaceholders());
    executeDAG(dag.root.get(), mod_, bindings, {}, {});

    Tensor *resultTensor = bindings.get(result);
    EXPECT_TRUE(referenceResult.isEqual(*resultTensor));
  }

  /// Test SparseLengthsSum independently.
  void testSLSQuant() {
    std::vector<Placeholder *> sparseLengths(1);
    sparseLengths[0] =
        mod_.createPlaceholder(ElemKind::Int32ITy, {miniBatch}, "SL0", false);

    std::vector<NodeValue> embeddings(sparseLengths.size());
    createSparseEmbeddings(mod_, bindings_, F_, sparseLengths, tableSizes,
                           embeddingDim, embeddings, true);

    auto *save = F_->createSave("save", embeddings[0]);
    bindings_.allocate(save->getPlaceholder());

    SaveNode *result1 = llvm::cast<SaveNode>(F_->getNodeByName("save"));
    result = result1->getPlaceholder();

    EE_.compile(CompilationMode::Infer, F_);

    // Run graph
    EE_.run(bindings_);
    auto *resultTensor = bindings_.get(result);

    // TODO: for now we only check the output dimension, contents are ignored
    EXPECT_EQ(resultTensor->size(), miniBatch * embeddingDim);
    resultTensor->getHandle().dump();
  }
};

INSTANTIATE_TEST_CASE_P_FOR_BACKEND_TEST(RecSys, RecommendationSystemTest);

/// Standard Tests
/// These tests have three options:
///   * quantizeSLWS  enables Int8 Fused Rowwise Quantization for the Sparse
///     Embeddings (Int8 quantized values with float scale and offset).
///   * quantizeFC    enables Int8 Fused Rowwise Quantization for FC weights and
///     activations inside the MLPs.
///   * convertToFP16 walks the graph at the end of constructing the graph and
///     converts all FP32 nodes & tensors to FP16, meaning the graph will use
///     FP16 for internal weights, biases and activations (when not already Int8
///     quantized). Inputs and outputs are still FP32 but are immediately
///     dropped to FP16 precision at the beginning of the graph.

TEST_P(RecommendationSystemTest, RecSys_FP32) {
  ENABLED_BACKENDS(CPU, Habana);

  testRecSys(/* quantizeSLWS */ false,
             /* quantizeFC */ false,
             /* convertToFP16 */ false,
             /* gatherWeights */ false);
}

TEST_P(RecommendationSystemTest, RecSys_RWQuantized_SLWS_FC) {
  ENABLED_BACKENDS(CPU);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ true,
             /* convertToFP16 */ false,
             /* gatherWeights */ false);
}

TEST_P(RecommendationSystemTest, RecSys_RWQuantized_SLWS) {
  ENABLED_BACKENDS(CPU, Habana);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ false,
             /* convertToFP16 */ false,
             /* gatherWeights */ false);
}

TEST_P(RecommendationSystemTest, RecSys_RWQuantized_SLWS_FC_FP16) {
  ENABLED_BACKENDS(Interpreter);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ true,
             /* convertToFP16 */ true,
             /* gatherWeights */ false);
}

TEST_P(RecommendationSystemTest, RecSys_RWQuantized_SLWS_FP16) {
  ENABLED_BACKENDS(Interpreter);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ false,
             /* convertToFP16 */ true,
             /* gatherWeights */ false);
}

/// Partitioning Tests
/// These tests have the same options as the above, but also partition the
/// created graph into segments and walk the dag. The test then compares output
/// for the partitioned and unpartitioned runs.

TEST_P(RecommendationSystemTest, RecSys_FP32_Partitioned) {
  ENABLED_BACKENDS(CPU, Habana);

  testRecSys(/* quantizeSLWS */ false,
             /* quantizeFC */ false,
             /* convertToFP16 */ false,
             /* gatherWeights */ false);

  deviceMemCapacity *= 2; // Double memory for this test

  runPartitionedGraph(numDevices, deviceMemCapacity,
                      bindings_.get(result)->clone(), bindings_);
}

TEST_P(RecommendationSystemTest, RecSys_Partitioned_RWQuantized_SLWS) {
  ENABLED_BACKENDS(CPU, Habana);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ false,
             /* convertToFP16 */ false,
             /* gatherWeights */ false);

  deviceMemCapacity *= 2; // Double memory for this test

  runPartitionedGraph(numDevices, deviceMemCapacity,
                      bindings_.get(result)->clone(), bindings_);
}

TEST_P(RecommendationSystemTest, RecSys_Partitioned_RWQuantized_SLWS_FC) {
  ENABLED_BACKENDS(CPU);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ true,
             /* convertToFP16 */ false,
             /* gatherWeights */ false);

  runPartitionedGraph(numDevices, deviceMemCapacity,
                      bindings_.get(result)->clone(), bindings_);
}

TEST_P(RecommendationSystemTest, RecSys_Partitioned_RWQuantized_SLWS_FP16) {
  ENABLED_BACKENDS(Interpreter);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ false,
             /* convertToFP16 */ true,
             /* gatherWeights */ false);

  runPartitionedGraph(numDevices, deviceMemCapacity,
                      bindings_.get(result)->clone(), bindings_);
}

TEST_P(RecommendationSystemTest, RecSys_Partitioned_RWQuantized_SLWS_FC_FP16) {
  ENABLED_BACKENDS(Interpreter);

  testRecSys(/* quantizeSLWS */ true,
             /* quantizeFC */ true,
             /* convertToFP16 */ true,
             /* gatherWeights */ false);

  runPartitionedGraph(numDevices, deviceMemCapacity,
                      bindings_.get(result)->clone(), bindings_);
}

/// Test SLS independently, with no other layers being run.
TEST_P(RecommendationSystemTest, RecSys_SLS_Only) {
  ENABLED_BACKENDS(CPU, Habana);

  testSLSQuant();
}

/// Test gathering weights for SLWS.
TEST_P(RecommendationSystemTest, RecSys_FP32_Gather_Weights) {
  ENABLED_BACKENDS(CPU, Habana);

  testRecSys(/* quantizeSLWS */ false,
             /* quantizeFC */ false,
             /* convertToFP16 */ false,
             /* gatherWeights */ true);
}

/// Test gathering weights for SLWS.
TEST_P(RecommendationSystemTest, RecSys_FP32_Medium_Gather_Weights) {
  ENABLED_BACKENDS(CPU, Habana);

  // Note that this overrides the parameters provided by command line options if
  // provided, as this comes after SetUp().
  tableSizes = {800000, 600000, 700000, 900000, 1200000,
                800000, 600000, 700000, 900000, 1200000,
                800000, 600000, 700000, 900000, 1200000};
  deviceMemCapacity = 1024 * 1024 * 512; // 512M.
  testRecSys(/* quantizeSLWS */ false,
             /* quantizeFC */ false,
             /* convertToFP16 */ false,
             /* gatherWeights */ true);
}
