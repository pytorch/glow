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
#include <algorithm>
#include <cstdlib>
#include <future>
#include <random>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

namespace {
llvm::cl::OptionCategory category("ResNetBench Options");

llvm::cl::opt<std::string> backend("backend", llvm::cl::desc("Backend to use"),
                                   llvm::cl::Optional,
                                   llvm::cl::init("Interpreter"),
                                   llvm::cl::cat(category));
llvm::cl::opt<unsigned>
    numReps("numReps", llvm::cl::desc("Number of benchmark repititions"),
            llvm::cl::init(1), llvm::cl::value_desc("N"),
            llvm::cl::cat(category));

llvm::cl::opt<unsigned> batchSize("batchSize",
                                  llvm::cl::desc("Image batch size"),
                                  llvm::cl::init(1), llvm::cl::value_desc("N"),
                                  llvm::cl::cat(category));

llvm::cl::opt<unsigned> baseSize("baseSize",
                                 llvm::cl::desc("Image H and W initial size"),
                                 llvm::cl::init(224), llvm::cl::value_desc("N"),
                                 llvm::cl::cat(category));

llvm::cl::opt<unsigned>
    numBins("numBins", llvm::cl::desc("Number of image sizes to create"),
            llvm::cl::init(1), llvm::cl::value_desc("N"),
            llvm::cl::cat(category));

llvm::cl::opt<unsigned> stepSize(
    "stepSize", llvm::cl::desc("Difference between each dimension in bins"),
    llvm::cl::init(10), llvm::cl::value_desc("N"), llvm::cl::cat(category));

llvm::cl::opt<unsigned> replicationCount("replicationCount",
                                         llvm::cl::desc("replicationCount"),
                                         llvm::cl::init(1),
                                         llvm::cl::value_desc("N"),
                                         llvm::cl::cat(category));

llvm::cl::opt<bool> saturateHost("saturateHost", llvm::cl::desc("saturateHost"),
                                 llvm::cl::init(true), llvm::cl::cat(category));

llvm::cl::opt<bool> convertToFP16("convertToFP16",
                                  llvm::cl::desc("convertToFP16"),
                                  llvm::cl::init(true),
                                  llvm::cl::cat(category));

llvm::cl::opt<bool>
    fpEverywhere("fpEverywhere",
                 llvm::cl::desc("Run model in fp instead quantized"),
                 llvm::cl::init(false), llvm::cl::cat(category));

llvm::cl::opt<bool> dumpDAG("dumpDAG",
                            llvm::cl::desc("Dump the final glow graph"),
                            llvm::cl::init(true), llvm::cl::cat(category));

llvm::cl::opt<unsigned> numDevices("numDevices",
                                   llvm::cl::desc("Number of backend devices"),
                                   llvm::cl::init(1), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));

llvm::cl::opt<unsigned> maxActiveRequests(
    "maxActiveRequests", llvm::cl::desc("Maximum active Glow requests"),
    llvm::cl::init(250), llvm::cl::value_desc("N"), llvm::cl::cat(category));

llvm::cl::opt<unsigned> numBatches("numBatches",
                                   llvm::cl::desc("Number of batches to run"),
                                   llvm::cl::init(10),
                                   llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));

llvm::cl::opt<unsigned>
    numRequesters("numRequesters", llvm::cl::desc("Number of request threads"),
                  llvm::cl::init(1), llvm::cl::value_desc("N"),
                  llvm::cl::cat(category));

llvm::cl::opt<int>
    logEvery("logEvery", llvm::cl::desc("Log every N requests on first thead"),
             llvm::cl::init(1000), llvm::cl::value_desc("N"),
             llvm::cl::cat(category));

llvm::cl::opt<unsigned> numCompileThreads(
    "numCompileThreads",
    llvm::cl::desc("Number of threads to use for compilation"),
    llvm::cl::init(1), llvm::cl::value_desc("N"), llvm::cl::cat(category));

llvm::cl::opt<bool>
    avgPool("avgPool",
            llvm::cl::desc("Add quantized AdaptiveAvgPool node to the graph. "
                           "If fpEverywhere then the node will also be fp."),
            llvm::cl::init(true), llvm::cl::cat(category));

llvm::cl::opt<bool>
    avgPoolFP("avgPoolFP",
              llvm::cl::desc("Add fp AdaptiveAvgPool node to the graph."),
              llvm::cl::init(false), llvm::cl::cat(category));

enum class Block {
  Bottleneck,
  BasicBlock,
};

unsigned_t getExpansion(Block block) {
  switch (block) {
  case Block::Bottleneck:
    return 4;
  case Block::BasicBlock:
    return 1;
  }
  LOG(FATAL) << "Unsupported block";
}

class ResNetBuilder {
private:
  Function *F_ = nullptr;
  // Hack to distinguish weights of each convolution to prevent constant weight
  // sharing within graph but enable weight sharing across graphs
  unsigned_t nextFilterValue_ = 1;

  const Block block_;
  const unsigned_t groups_;
  const unsigned_t widthPerGroup_;
  const unsigned_t dilation_ = 1;
  const unsigned_t inPlanes_ = 64;
  const std::vector<unsigned_t> layers_;

  NodeValue createConv(NodeValue input, unsigned_t outChannels,
                       unsigned_t kernel, unsigned_t stride = 1,
                       unsigned_t pad = 0, unsigned_t dilation = 1,
                       unsigned_t groups = 1, bool fp = false) {

    if (fpEverywhere) {
      fp = true;
    }

    ShapeNHWC inputShape(input.dims());

    assert(inputShape.c % groups == 0);
    assert(outChannels % groups == 0);

    auto *filter = F_->getParent()->createConstant(
        ElemKind::FloatTy, {outChannels, kernel, kernel, inputShape.c / groups},
        "filter");

    size_t fanIn = kernel * kernel * inputShape.c;
    filter->getPayloadMutable().init(Tensor::InitKind::Xavier, fanIn,
                                     F_->getParent()->getPRNG());

    // Need to be constant so that all networks have the same weights
    filter->getPayloadMutable().init(Tensor::InitKind::Broadcast,
                                     float(20 + nextFilterValue_++),
                                     F_->getParent()->getPRNG());

    auto bias = F_->getParent()->createConstant(ElemKind::FloatTy,
                                                {outChannels}, "bias");
    bias->getPayloadMutable().init(Tensor::InitKind::Broadcast,
                                   float(20 + nextFilterValue_++),
                                   F_->getParent()->getPRNG());

    std::vector<unsigned_t> kernels = {kernel, kernel};
    std::vector<unsigned_t> strides = {stride, stride};
    std::vector<unsigned_t> pads = {pad, pad, pad, pad};
    std::vector<unsigned_t> dilations = {dilation, dilation};

    auto outSz = calculateConvPoolOutputDims(inputShape.h, inputShape.w,
                                             kernels, strides, pads, dilations);
    std::array<dim_t, 4> outDims = {
        {inputShape.n, outSz.first, outSz.second, outChannels}};

    if (fp) {
      auto *outTy = F_->getParent()->uniqueType(ElemKind::FloatTy, outDims);
      return F_
          ->createConv("conv", input, filter, bias, outTy, kernels, strides,
                       pads, groups, dilations)
          ->getResult();
    } else {
      auto *outTy =
          F_->getParent()->uniqueType(ElemKind::Int8QTy, outDims, 1.0, 0);

      return F_
          ->createChannelwiseQuantizedConv(
              "conv", input, filter, bias, /*filterScales*/ nullptr,
              /*filterOffsets*/ nullptr, /*biasScales*/ nullptr,
              /*biasOffsets*/ nullptr, outTy, kernels, strides, pads, groups,
              dilations,
              /*quantizeFilter*/ true, /*quantizeBias*/ false,
              /*schema*/ quantization::Schema::Symmetric)
          ->getResult();
    }
  }

  NodeValue conv3x3(NodeValue input, unsigned_t outPlanes,
                    unsigned_t stride = 1, unsigned_t groups = 1,
                    unsigned_t dilation = 1) {
    return createConv(input, outPlanes, /*kernel*/ 3, stride, /*pad*/ dilation,
                      dilation, groups);
  }

  NodeValue conv1x1(NodeValue input, unsigned_t outPlanes,
                    unsigned_t stride = 1) {
    return createConv(input, outPlanes, /*kernel*/ 1, stride);
  }

  NodeValue createRelu(NodeValue input) {
    return F_->createRELU("relu", input)->getResult();
  }

  NodeValue createAdd(NodeValue lhs, NodeValue rhs) {
    if (isQuantizedElemKind(lhs.getElementType())) {
      return F_->createAdd("qadd", lhs.getType(), lhs, rhs);
    } else {
      return F_->createAdd("add", lhs, rhs);
    }
  }

  NodeValue createBN(NodeValue input) {
    // Emulate fused Conv + BN
    auto inputKind = input.getNode()->getKind();
    if (inputKind == Kinded::Kind::ConvolutionNodeKind ||
        inputKind == Kinded::Kind::ChannelwiseQuantizedConvolutionNodeKind) {
      return input;
    }
    LOG(FATAL)
        << "Fake batchnorm op has to be after a convolution to emulate fusion";
  }

  NodeValue makeAvgPool(NodeValue input) {
    auto inputDims = input.dims();
    auto *outTy = F_->getParent()->uniqueTypeWithNewShape(
        input.getType(), {inputDims[0], 1, 1, inputDims[3]});
    return F_->createAdaptiveAvgPool("adaptive_avg_pool", input, outTy)
        ->getResult();
  }

  NodeValue makeBlock(NodeValue input, NodeValue residual, unsigned_t planes,
                      unsigned_t stride = 1, unsigned_t groups = 1,
                      unsigned_t baseWidth = 64, unsigned_t dilation = 1) {
    unsigned_t expansion = getExpansion(block_);
    NodeValue next = input;
    if (block_ == Block::Bottleneck) {
      auto width = unsigned_t(planes * (baseWidth / 64.0)) * groups;
      next = conv1x1(next, width);
      next = createBN(next);
      next = createRelu(next);

      next = conv3x3(next, width, stride, groups, dilation);
      next = createBN(next);
      next = createRelu(next);

      next = conv1x1(next, planes * expansion);
      next = createBN(next);
      next = createAdd(next, residual);
      next = createRelu(next);

    } else if (block_ == Block::BasicBlock) {
      next = conv3x3(next, planes, stride);
      next = createBN(next);
      next = createRelu(next);

      next = conv3x3(next, planes);
      next = createBN(next);
      next = createAdd(next, residual);
      next = createRelu(next);
    } else {
      LOG(FATAL) << "Unknown block";
    }
    return next;
  }

  NodeValue makeLayer(NodeValue input, unsigned_t planes, unsigned_t blocks,
                      unsigned_t stride) {
    NodeValue next = input;
    NodeValue residual = next;
    auto blockExpansion = getExpansion(block_);
    if (stride != 1 || inPlanes_ != planes * blockExpansion) {
      residual = conv1x1(next, planes * blockExpansion, stride);
    }
    next = makeBlock(next, residual, planes, stride, groups_, widthPerGroup_,
                     dilation_);

    for (unsigned_t i = 1; i < blocks; ++i) {
      residual = next;
      next = makeBlock(next, residual, planes, /*stride*/ 1, groups_,
                       widthPerGroup_, dilation_);
    }

    return next;
  }

public:
  ResNetBuilder(Block block, llvm::ArrayRef<unsigned_t> layers,
                unsigned_t groups = 1, unsigned_t widthPerGroup = 64)
      : block_(block), groups_(groups), widthPerGroup_(widthPerGroup),
        layers_(layers.vec()) {}

  Placeholder *build(Placeholder *input, Function *F) {
    F_ = F;
    nextFilterValue_ = 1;
    NodeValue next = input->getOutput();
    next = F_->createTranspose("NCHW2NHWC", next, NCHW2NHWC);
    next =
        createConv(next, /*outChannels*/ inPlanes_, /*kernel*/ 7, /*stride*/ 2,
                   /*pad*/ 3, /*dilation*/ 1, /*groups*/ 1, /*fp*/ true);
    next = createBN(next);
    next = createRelu(next);
    next = F_->createMaxPool("maxpool", next, /*kernel*/ 3, /*sride*/ 2,
                             /*pad*/
                             1)
               ->getResult();
    if (!fpEverywhere) {
      next = F_->createQuantize("quant", next, ElemKind::Int8QTy, 1.0, 0);
    }
    next = makeLayer(next, /*planes*/ 64, /*blocks*/ layers_[0],
                     /*stride*/ 1);

    next = makeLayer(next, /*planes*/ 128, /*blocks*/ layers_[1],
                     /*stride*/ 2);

    next = makeLayer(next, /*planes*/ 256, /*blocks*/ layers_[2],
                     /*stride*/ 2);

    next = makeLayer(next, /*planes*/ 512, /*blocks*/ layers_[3],
                     /*stride*/ 2);
    if (avgPool) {
      next = makeAvgPool(next);
    }
    next = makeAvgPool(next);
    if (!fpEverywhere) {
      next =
          F_->createDequantize("dequant", next, ElemKind::FloatTy)->getResult();
    }
    if (avgPoolFP) {
      next = makeAvgPool(next);
    }
    next = F_->createTranspose("NHWC2NCHW", next, NHWC2NCHW);
    Placeholder *output = F_->createSave("save", next)->getPlaceholder();
    F_ = nullptr;
    return output;
  }
};

struct FunctionBundle {
  std::string name;
  Placeholder *input;
  Placeholder *output;
};

ResNetBuilder resnext101_32x4d() {
  return ResNetBuilder(Block::Bottleneck, {3, 4, 23, 3},
                       /*groups*/ 32,
                       /*widthPerGroup*/ 4);
}

// ResNetBuilder resnet50() {
//   return ResNetBuilder(Block::Bottleneck, {3, 4, 6, 3});
// }

/*
 * This class implements a performance proxy for ResNet-like models
 */
class ResNetBench : public Benchmark {
private:
  ResNetBuilder builder_;
  std::vector<ShapeNCHW> shapes_;
  std::string backendName_;
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<FunctionBundle> bundles_;
  int64_t compilationTime_;

  std::vector<FunctionBundle> makeNetworks(ResNetBuilder builder, Module &mod,
                                           llvm::ArrayRef<ShapeNCHW> shapes) {
    std::vector<FunctionBundle> res;
    for (const auto &shape : shapes) {
      std::string shapeStr =
          strFormat("%dx%dx%dx%d", int(shape.n), int(shape.c), int(shape.h),
                    int(shape.w));
      auto *F = mod.createFunction(strFormat("F_%s", shapeStr.c_str()));
      Placeholder *input = mod.createPlaceholder(
          ElemKind::FloatTy, {shape.n, shape.c, shape.h, shape.w}, "input",
          false);
      Placeholder *output = builder.build(input, F);
      FunctionBundle bundle;
      bundle.name = F->getName().str();
      bundle.input = input;
      bundle.output = output;
      res.push_back(std::move(bundle));
    }
    return res;
  }

public:
  ResNetBench(ResNetBuilder builder, llvm::ArrayRef<ShapeNCHW> shapes,
              std::string backendName)
      : builder_(builder), shapes_(shapes.vec()), backendName_(backendName) {}

  void setup() override {
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    for (unsigned_t i = 0; i < numDevices; ++i) {
      auto config = std::make_unique<runtime::DeviceConfig>(backendName_);
      config->deviceID = i;
      configs.push_back(std::move(config));
    }

    glow::runtime::HostConfig hostConfig;
    hostConfig.maxActiveRequests = maxActiveRequests;

    hostManager_ =
        std::make_unique<runtime::HostManager>(std::move(configs), hostConfig);

    const auto numCompileThreadsToUse =
        std::min(size_t(numCompileThreads), shapes_.size());

    // Divide Functions up for compilation threads
    LOG(INFO) << "Building networks";
    std::vector<std::unique_ptr<Module>> modules;
    for (size_t i = 0; i < numCompileThreadsToUse; ++i) {
      auto mod = std::make_unique<Module>();
      const auto beginIt =
          shapes_.begin() + ((shapes_.size() / numCompileThreadsToUse) * i);
      const auto endIt =
          i == numCompileThreadsToUse - 1
              ? shapes_.end()
              : shapes_.begin() +
                    ((shapes_.size() / numCompileThreadsToUse) * (i + 1));
      std::vector<ShapeNCHW> threadShapes{beginIt, endIt};
      auto bundles = makeNetworks(builder_, *mod, threadShapes);
      for (auto &bundle : bundles) {
        bundles_.push_back(std::move(bundle));
      }
      modules.push_back(std::move(mod));
    }

    auto compileFn = [this](std::unique_ptr<Module> mod) {
      glow::CompilationContext cctx;
      cctx.replicationCount = replicationCount;
      cctx.saturateHost = saturateHost;
      cctx.precisionConfig.convertToFP16 = convertToFP16;
      cctx.dumpFinalGraph = dumpDAG;
      hostManager_->addNetwork(std::move(mod), cctx);
    };

    // Compile modules in parallel
    LOG(INFO) << "Compiling networks";
    int64_t compilationStartTime = TraceEvent::now();
    std::vector<std::thread> threads;
    for (size_t i = 1; i < numCompileThreadsToUse; ++i) {
      auto mod = std::move(modules[i]);
      std::thread t(compileFn, std::move(mod));
      threads.push_back(std::move(t));
    }

    compileFn(std::move(modules[0]));

    for (auto &t : threads) {
      t.join();
    }

    int64_t compilationEndTime = TraceEvent::now();
    compilationTime_ = compilationEndTime - compilationStartTime;

    // Run a few warmups
    LOG(INFO) << "Running warmups";
    runImpl(2 * bundles_.size());
  }

  void runImpl(unsigned_t numRuns, int32_t threadNum = -1) {
    std::unique_ptr<ExecutionContext> ctx =
        glow::make_unique<ExecutionContext>();

    auto *bindings = ctx->getPlaceholderBindings();

    for (unsigned_t i = 0; i < numRuns; i++) {
      if (logEvery > 0 && i > 0 && threadNum == 0 && i % logEvery == 0) {
        LOG(INFO) << "Thread 0 reached request " << i;
      }
      // Add threadNum to offset the theads
      auto nextBundleNum = (std::max(threadNum, 0) + i) % bundles_.size();
      const auto &bundle = bundles_[nextBundleNum];
      bindings->allocate(bundle.input);
      bindings->allocate(bundle.output);
      auto err = hostManager_->runNetworkBlocking(bundle.name, ctx);
    }
  }

  void run() override {
    std::vector<std::thread> threads;
    unsigned_t reqsPerThread = numBatches / numRequesters;
    unsigned_t numReqs = numRequesters * reqsPerThread;

    LOG(INFO) << "Running";
    int64_t startTime = TraceEvent::now();
    for (unsigned_t i = 0; i < numRequesters; ++i) {
      threads.push_back(std::thread(
          [this, reqsPerThread, i]() { runImpl(reqsPerThread, i); }));
    }

    for (auto &thread : threads) {
      thread.join();
    }
    int64_t endTime = TraceEvent::now();
    int64_t totatTimeMs = (endTime - startTime) / 1000;

    std::cout << "Total runtime: " << totatTimeMs << "ms" << std::endl;
    if (totatTimeMs > 0) {
      std::cout << "Avg requests/second: "
                << numReqs / (double(totatTimeMs) / 1000) << std::endl;
      std::cout << "Avg images/second: "
                << (batchSize * numReqs) / (double(totatTimeMs) / 1000)
                << std::endl;
      std::cout << "Avg runtime per request " << double(totatTimeMs) / numReqs
                << "ms" << std::endl;
    }
    std::cout << "numBins: " << numBins << std::endl;
    std::cout << "baseSize: " << baseSize << "x" << baseSize << std::endl;
    std::cout << "batchSize: " << batchSize << std::endl;
    std::cout << "stepSize: " << stepSize << std::endl;
    std::cout << "replicationCount: " << replicationCount << std::endl;
    std::cout << "numDevices: " << numDevices << std::endl;
    std::cout << "numRequesters: " << numRequesters << std::endl;
    std::cout << "compilation time: " << compilationTime_ / 1000 << "ms"
              << std::endl;
  }

  void teardown() override { LOG(INFO) << "Teardown"; }
};

std::vector<ShapeNCHW> generateShapes(dim_t batchSize, dim_t baseSize,
                                      dim_t numBins, dim_t stepSize) {
  assert(numBins > 0);
  std::vector<ShapeNCHW> shapes;
  shapes.emplace_back(batchSize, 3, baseSize, baseSize);

  ShapeNCHW hStepped(batchSize, 3, baseSize, baseSize);
  ShapeNCHW wStepped(batchSize, 3, baseSize, baseSize);
  for (dim_t i = 1; i < numBins; ++i) {
    if (i % 2 == 0) {
      hStepped.h += stepSize;
      shapes.push_back(hStepped);
    } else {
      wStepped.w += stepSize;
      shapes.push_back(wStepped);
    }
  }
  return shapes;
}
} // namespace

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "ResNet benchmark");

  CHECK(!avgPool || !avgPoolFP) << "avgPool and avgPoolFP can't be true or "
                                   "pooling will occur two times";

  std::vector<ShapeNCHW> shapes =
      generateShapes(batchSize, baseSize, numBins, stepSize);
  auto builder = resnext101_32x4d();
  ResNetBench b(builder, shapes, backend);

  bench(&b, numReps);
}
