// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "glow/lib/Backends/NNPI/CustomKernels/DSPInjectors/DSPInjectorUtils.h"

#include <glog/logging.h>

#define FBGEMM_FP16_MAX 65504.0f
#define FBGEMM_FP16_MIN -65504.0f
#define FBGEMM_DSP_VECTOR_WIDTH_F16 32

namespace glow {
namespace {

NNPITileParams getWalkDim1RowwiseTile(std::vector<size_t> dims) {
  size_t numDims = (size_t)dims.size();
  DCHECK_GT(dims.size(), 0);
  size_t buffering = 1;

  std::vector<uint32_t> tensorDims(numDims);
  std::vector<uint32_t> tileDims(numDims);
  std::vector<uint32_t> alignment(numDims);

  size_t desiredSize = 16384;
  size_t rowSize = 1;
  bool doubleBufferingNeeded = false;
  for (int dim = 0; dim < numDims; dim++) {
    size_t reverseDims = numDims - dim - 1;
    DCHECK_GE(reverseDims, 0);
    DCHECK_LT(reverseDims, numDims);
    uint32_t dimval = dims[numDims - dim - 1];
    tensorDims[dim] = (uint32_t)dimval;
    alignment[dim] =
        (dim > 0) ? 1 : FBGEMM_DSP_VECTOR_WIDTH_F16 * sizeof(float16_t);

    // Set tile size for this dim
    size_t numRowsNeeded = desiredSize / rowSize;
    if (numRowsNeeded >= tensorDims[dim]) {
      tileDims[dim] = tensorDims[dim];
    } else {
      doubleBufferingNeeded = true;
      tileDims[dim] = (numRowsNeeded > 1) ? numRowsNeeded : 1;
    }

    // Pad dim0 to multiple of 32
    if (dim == 0) {
      tileDims[dim] = FBGEMM_DSP_VECTOR_WIDTH_F16 *
                      ((tileDims[dim] + FBGEMM_DSP_VECTOR_WIDTH_F16 - 1) /
                       FBGEMM_DSP_VECTOR_WIDTH_F16);
    }
    rowSize = rowSize * tileDims[dim];
  }
  if (doubleBufferingNeeded) {
    buffering = 2;
  }

  // setup tile params to be used for all tensors
  NNPITileParams tileParams;
  tileParams.numDims = numDims;
  tileParams.numTilesPerBuffer = buffering;

  // Setup walk according to tile size
  std::vector<uint32_t> numTiles(numDims);
  uint32_t numIterations = 1;
  for (int i = 0; i < numDims; i++) {
    uint32_t tensorSize = (uint32_t)tensorDims[i];
    uint32_t tileStep = (uint32_t)tileDims[i];
    numTiles[i] = std::ceil((float)tensorSize / tileStep);
    tileParams.walkParams[i] = {numIterations, 1, 1, numTiles[i], 1, 1};
    numIterations *= numTiles[i];
  }

  tileParams.numIterations = numIterations;
  std::copy(std::begin(tileDims), std::end(tileDims),
            std::begin(tileParams.dims));
  std::copy(std::begin(alignment), std::end(alignment),
            std::begin(tileParams.alignment));
  std::copy(std::begin(tileDims), std::end(tileDims),
            std::begin(tileParams.tileStep));

  // Dump summary of choices ( change to LOG(INFO) )
  // print_vector("dims", dims);
  // print_vector("tensorDims", tensorDims);
  // print_vector("tileDims", tileDims);
  // print_vector("alignment", alignment);
  LOG(INFO) << "numIterations: " << numIterations << std::endl;
  return tileParams;
}

struct EltwiseParams {
  uint16_t numOfVectors;
};
} // namespace

int DSPInjectorUtils::GetNumElements(uint32_t *dims, int numDims) {
  int numOfElm = 1;
  for (int i = 0; i < numDims; i++) {
    numOfElm *= dims[i];
  }
  return numOfElm;
}

/* Reusable elementwise creation functions */
NNPICustomDSPNode *DSPInjectorUtils::createCustomEltwiseFP16_configurable(
    Function *F_, const std::string &name, const std::string &kernel_name,
    std::vector<NodeValue> input_nodes, int64_t IceRefCallback,
    NNPITileParams tileParams) {
  // Set the kernel params (loop variable)
  Constant *kernelParams = F_->getParent()->createConstant(
      F_->getParent()->uniqueType(ElemKind::UInt8QTy, {sizeof(EltwiseParams)},
                                  1.0, 0),
      std::string(name) + std::string("kernelParams"));
  auto kpH = kernelParams->getPayload().getUnsafePtr();
  EltwiseParams params;

  int32_t numTileElements = GetNumElements(tileParams.dims, tileParams.numDims);
  DCHECK_EQ(numTileElements % 32, 0)
      << "Tile num elements must be multiple of 32 for this op";
  params.numOfVectors = numTileElements / 32;
  memcpy(kpH, &params, sizeof(EltwiseParams));

  // Setup walk config
  NNPIWalkConfig walkConfig;
  walkConfig.numInputTensors = input_nodes.size();
  walkConfig.numOutputTensors = 1;

  // copy same tile params for all tensors
  for (int i = 0; i < input_nodes.size(); i++) {
    walkConfig.inputTileWalkParams[i] = tileParams;
  }
  walkConfig.outputTileWalkParams[0] = tileParams;

  Constant *walkConfigConstant = F_->getParent()->createConstant(
      F_->getParent()->uniqueType(ElemKind::UInt8QTy, {sizeof(NNPIWalkConfig)},
                                  1.0, 0),
      std::string(name) + std::string("walkConfig"));
  auto wcH = walkConfigConstant->getPayload().getUnsafePtr();
  memcpy(wcH, &walkConfig, sizeof(NNPIWalkConfig));
  DCHECK_GT(input_nodes.size(), 0);

  // Create the DSP node and add to the network
  NNPICustomDSPNode *dspNode = new NNPICustomDSPNode(
      name,                     // name
      input_nodes[0].getType(), // result type
      kernelParams,             // kernel params tensor
      walkConfigConstant,
      input_nodes,                              // input nodes
      10,                                       // private area size
      kernel_name,                              // kernel name
      reinterpret_cast<int64_t>(IceRefCallback) // ice ref callback
  );

  auto *addq = F_->addNode(dspNode);
  return addq;
}

NNPICustomDSPNode *DSPInjectorUtils::createEltwiseFP16(
    Function *F_, const std::string &name, const std::string &kernel_name,
    std::vector<NodeValue> input_nodes, int64_t IceRefCallback) {
  DCHECK_GT(input_nodes.size(), 0);
  std::vector<size_t> dims = input_nodes[0].getType()->dims();

  // Choose tile size, amount of buffering, and alignment
  NNPITileParams tileParams;
  tileParams = getWalkDim1RowwiseTile(dims);

  // Call configurable kernel
  return DSPInjectorUtils::createCustomEltwiseFP16_configurable(
      F_, name, kernel_name, input_nodes, IceRefCallback, tileParams);
}

} // namespace glow

#undef FBGEMM_FP16_MAX
#undef FBGEMM_FP16_MIN
#undef FBGEMM_DSP_VECTOR_WIDTH_F16
