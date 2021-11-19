#include "BackendTestUtils.h"

#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Flags/Flags.h"
#include "glow/lib/Backends/NNPI/InferenceContext.h"
#include "glow/lib/Backends/NNPI/NNPIOptions.h"

#include "gtest/gtest.h"

using namespace glow;

TEST(NNPIInferenceContextTest, SanitizationFailure) {
  runtime::flags::SanitizeInputsPercent = 100;

  ExecutionEngine loadedEE{"NNPI"};
  Module &loadedMod = loadedEE.getModule();
  Function *loadedF = loadedMod.createFunction("main");

  const std::vector<dim_t> indicesShape{20};
  const std::vector<dim_t> lengthsShape{20};
  const size_t indicesUnpaddedSize = 1000;

  auto phIndices = loadedMod.createPlaceholder(ElemKind::Int32ITy, indicesShape,
                                               "dummyIndices", false);
  auto phLengths = loadedMod.createPlaceholder(ElemKind::Int32ITy, lengthsShape,
                                               "dummyLengths", false);

  const size_t tableHeight = 300;
  std::vector<ValidateSLSInfo> validateSLSInputs;
  validateSLSInputs.emplace_back(true, tableHeight, phIndices, nullptr,
                                 phLengths, nullptr);

  // Dummy initialization of NNPI inference context

  runtime::InferenceContext infCtx;

  NNPICompiledFunction nnpiF{loadedF};
  size_t deviceNetwork = 10001;
  NNPIAdapterContainer *adapter{nullptr};
  size_t device = 10002;
  runtime::StaticPlaceholderMap phMap;
  llvm::StringMap<std::string> params;
  auto deviceOptions = std::make_shared<NNPIDeviceOptions>(params);
  unsigned int deviceId = 3;
  const auto initResult = infCtx.init(
      {}, {}, nnpiF.getCompiledNetworkHandle(), nnpiF.getCompilationConfig(),
      deviceNetwork, adapter, device, nnpiF.getPartialInputs(),
      validateSLSInputs, nnpiF.getPaddedInputs(), nnpiF.getStaticInputs(),
      &phMap, deviceOptions, "main", deviceId);
  ASSERT_TRUE(initResult);

  // Prepare a request with bad SLS inputs.

  auto bindings = std::make_unique<PlaceholderBindings>();
  auto tIndices = bindings->allocate(phIndices);
  // Making sure the tensor looks like partial by updating unpadded size.
  tIndices->reset(tIndices->getType(), indicesUnpaddedSize);
  // Populating indices with values that are out of range.
  tIndices->getHandle<int32_t>().randomize(tableHeight, 2 * tableHeight - 1,
                                           loadedMod.getPRNG());
  auto tLengths = bindings->allocate(phLengths);
  // Populating lengths with proper values.
  tLengths->getHandle<int32_t>().clear(1);

  const size_t requestId = 22222;

  auto errMsg = std::make_shared<std::string>();
  infCtx.execute(requestId,
                 std::make_unique<glow::ExecutionContext>(std::move(bindings)),
                 [errMsg](glow::runtime::RunIdentifierTy, Error err,
                          std::unique_ptr<glow::ExecutionContext>) {
                   auto errVal = takeErrorValue(std::move(err));
                   *errMsg = errVal->logToString();
                 });

  auto foundPos = errMsg->find("Error message: Failed santization.");
  ASSERT_TRUE(foundPos != std::string::npos);
  LOG(ERROR) << *errMsg;
}
