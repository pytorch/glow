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
#include "glow/Runtime/InputSanitizer.h"
#include "glow/Flags/Flags.h"
#include "glow/glow/tests/unittests/BackendTestUtils.h"

#include "gtest/gtest.h"

using namespace glow;
using namespace glow::runtime;

namespace {

using CreateOperatorFn = std::function<Placeholder *(
    Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
    Placeholder *phWeights, Placeholder *phLengths, Placeholder *phOffsets)>;

using PopulateTensorsFn =
    std::function<void(Tensor *indices, Tensor *lengths, Tensor *offsets)>;

void runTest(Module *m, Function *f, dim_t embeddingRows, dim_t indicesNumMax,
             dim_t indicesNum, dim_t lengthsNum, dim_t offsetsNum,
             CreateOperatorFn createOperatorFn,
             PopulateTensorsFn populateTensorsFn, bool shouldSucceed,
             const std::string &expectedErrorMessage) {
  glow::runtime::flags::SanitizeInputsPercent = 100;

  auto phData = m->createPlaceholder(ElemKind::FloatTy, {embeddingRows, 1},
                                     "data", false);
  auto phFQData = m->createPlaceholder(
      ElemKind::UInt8FusedQTy, {embeddingRows, 1 + 2 * (dim_t)sizeof(float)},
      1.0, 0, "fqData", false);
  auto phIndices = m->createPlaceholder(ElemKind::Int32ITy, {indicesNumMax},
                                        "indices", false);
  auto phWeights = m->createPlaceholder(ElemKind::FloatTy, {indicesNumMax},
                                        "weights", false);
  auto phLengths =
      m->createPlaceholder(ElemKind::Int32ITy, {lengthsNum}, "lengths", false);
  auto phOffsets =
      m->createPlaceholder(ElemKind::Int32ITy, {offsetsNum}, "offsets", false);

  auto phOut = createOperatorFn(phData, phFQData, phIndices, phWeights,
                                phLengths, phOffsets);

  auto sanitizers = getInputSanitizers(*f);
  ASSERT_EQ(1, sanitizers.size());

  Tensor indicesReal(ElemKind::Int32ITy, {indicesNum});
  Tensor lengthsReal(ElemKind::Int32ITy, {lengthsNum});
  Tensor offsetsReal(ElemKind::Int32ITy, {offsetsNum});

  populateTensorsFn(&indicesReal, &lengthsReal, &offsetsReal);

  Tensor indicesPartial(indicesReal.getUnsafePtr(), phIndices->getType(),
                        indicesReal.getSizeInBytes());
  Tensor lengthsPartial(lengthsReal.getUnsafePtr(), phLengths->getType(),
                        lengthsReal.getSizeInBytes());
  Tensor offsetsPartial(offsetsReal.getUnsafePtr(), phOffsets->getType(),
                        offsetsReal.getSizeInBytes());

  auto bindings = std::make_unique<PlaceholderBindings>();
  bindings->insert(phIndices, std::move(indicesPartial));
  bindings->insert(phLengths, std::move(lengthsPartial));
  bindings->insert(phOffsets, std::move(offsetsPartial));
  bindings->allocate(phOut);

  auto sanitizeError = sanitizeInputs(sanitizers, *bindings);
  bool failed = static_cast<bool>(sanitizeError);

  if (shouldSucceed) {
    if (failed) {
      auto errMsg = takeErrorValue(std::move(sanitizeError))->logToString();
      ASSERT_TRUE(false)
          << "Sanitization unexpectedly failed with this message: " << errMsg;
    }
  } else {
    ASSERT_TRUE(failed) << "Sanitization unexpectedly succeeded";
    auto errMsg = takeErrorValue(std::move(sanitizeError))->logToString();
    auto foundPos = errMsg.find(expectedErrorMessage);
    ASSERT_TRUE(foundPos != std::string::npos)
        << "Sanitization failed with a different message: " << errMsg;
  }
}

} // namespace

TEST(InputSanitizerTest, CheckSLS_HappyPath) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto SLS =
            f->createSparseLengthsSum("SLS", phData, phIndices, phLengths);
        auto saveSLS = f->createSave("saveSLS", SLS);
        return saveSLS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // Making sum of the lengths equal to the number of indices
        lengths->getHandle<int32_t>().clear(5);
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}

TEST(InputSanitizerTest, CheckSLS_NegativeIndex) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto SLS =
            f->createSparseLengthsSum("SLS", phData, phIndices, phLengths);
        auto saveSLS = f->createSave("saveSLS", SLS);
        return saveSLS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().raw(0) = -1;
        lengths->getHandle<int32_t>().clear(5);
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: Indices sanitization failed on tensor indices: index -1 "
      "at pos 0 is out of range [0, 1275)");
}

TEST(InputSanitizerTest, CheckSLS_VeryLargeIndex) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto SLS =
            f->createSparseLengthsSum("SLS", phData, phIndices, phLengths);
        auto saveSLS = f->createSave("saveSLS", SLS);
        return saveSLS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().raw(0) = 10 * embeddingRows;
        lengths->getHandle<int32_t>().clear(5);
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: Indices sanitization failed on tensor indices: index "
      "12750 at pos 0 is out of range [0, 1275)");
}

TEST(InputSanitizerTest, CheckSLS_NegativeLength) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto SLS =
            f->createSparseLengthsSum("SLS", phData, phIndices, phLengths);
        auto saveSLS = f->createSave("saveSLS", SLS);
        return saveSLS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        lengths->getHandle<int32_t>().raw(0) = -1;
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: SLS lengths sanitization failed on tensor lengths: "
      "length -1 at pos 0 is negative");
}

TEST(InputSanitizerTest, CheckSLS_BadSumOfLengths) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto SLS =
            f->createSparseLengthsSum("SLS", phData, phIndices, phLengths);
        auto saveSLS = f->createSave("saveSLS", SLS);
        return saveSLS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        lengths->getHandle<int32_t>().clear(100);
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: SLS lengths sanitization failed on tensor lengths: "
      "indices length 50 is not equal to sum of lengths 1000");
}

TEST(InputSanitizerTest, CheckSLWS_HappyPath) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto SLWS = f->createSparseLengthsWeightedSum("SLWS", phData, phWeights,
                                                      phIndices, phLengths);
        auto saveSLWS = f->createSave("saveSLWS", SLWS);
        return saveSLWS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // Making sum of the lengths equal to the number of indices
        lengths->getHandle<int32_t>().clear(5);
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}

TEST(InputSanitizerTest, CheckFRQSLS_HappyPath) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto FRQSLS = f->createFusedRowwiseQuantizedSparseLengthsSum(
            "FRQSLS", phFQData, phIndices, phLengths);
        auto saveFRQSLS = f->createSave("saveFRQSLS", FRQSLS);
        return saveFRQSLS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // Making sum of the lengths equal to the number of indices
        lengths->getHandle<int32_t>().clear(5);
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}

TEST(InputSanitizerTest, CheckFRQSLWS_HappyPath) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto FRQSLWS = f->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            "FRQSLWS", phFQData, phWeights, phIndices, phLengths);
        auto saveFRQSLWS = f->createSave("saveFRQSLWS", FRQSLWS);
        return saveFRQSLWS->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // Making sum of the lengths equal to the number of indices
        lengths->getHandle<int32_t>().clear(5);
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}

TEST(InputSanitizerTest, CheckEB_HappyPath) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // All zeros is a valid tensor for EBB.
        offsets->getHandle<int32_t>().clear(0);
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}

TEST(InputSanitizerTest, CheckEB_HappyPath_ProperLastOffset) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // Making sure offsets are non-decreasing and the last offset is equal
        // to the number of indices.
        offsets->getHandle<int32_t>().clear(0);
        offsets->getHandle<int32_t>().raw(offsetsNum - 1) = indicesNum;
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}

TEST(InputSanitizerTest, CheckEB_NegativeIndex) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().raw(0) = -1;
        offsets->getHandle<int32_t>().clear(0);
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: Indices sanitization failed on tensor indices: index -1 "
      "at pos 0 is out of range [0, 1275)");
}

TEST(InputSanitizerTest, CheckEB_VeryLargeIndex) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().raw(0) = 10 * embeddingRows;
        offsets->getHandle<int32_t>().clear(0);
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: Indices sanitization failed on tensor indices: index "
      "12750 at pos 0 is out of range [0, 1275)");
}

TEST(InputSanitizerTest, CheckEB_FirstOffsetNotZero) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        offsets->getHandle<int32_t>().raw(0) = 1;
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: EBB offsets sanitization failed on tensor offsets: the "
      "first offset is not zero 1");
}

TEST(InputSanitizerTest, CheckEB_DecreasingOffsets) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        offsets->getHandle<int32_t>().raw(0) = 0;
        offsets->getHandle<int32_t>().raw(1) = 5;
        offsets->getHandle<int32_t>().raw(2) = 3;
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: EBB offsets sanitization failed on tensor offsets: "
      "decreasing offsets 5 and 3 at pos 1");
}

TEST(InputSanitizerTest, CheckEB_BadLastOffset) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EB = f->createEmbeddingBag("EB", phData, phWeights, phIndices,
                                        phOffsets);
        auto saveEB = f->createSave("saveEB", EB);
        return saveEB->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        offsets->getHandle<int32_t>().clear(0);
        offsets->getHandle<int32_t>().raw(offsetsNum - 1) = 3 * indicesNum;
      },
      /* shouldSucceed */ false,
      /* expectedErrorMessage */
      "Error message: EBB offsets sanitization failed on tensor offsets: the "
      "last offset 150 is not equal to the number of indices 50");
}

TEST(InputSanitizerTest, CheckEBBRO_HappyPath) {
  const dim_t embeddingRows = 1275;
  const dim_t indicesNumMax = 20000;
  const dim_t indicesNum = 50;
  const dim_t lengthsNum = 10;
  const dim_t offsetsNum = 11;

  auto m = std::make_shared<Module>();
  auto f = m->createFunction("testFunction");

  runTest(
      m.get(), f, embeddingRows, indicesNumMax, indicesNum, lengthsNum,
      offsetsNum,
      [f](Placeholder *phData, Placeholder *phFQData, Placeholder *phIndices,
          Placeholder *phWeights, Placeholder *phLengths,
          Placeholder *phOffsets) {
        auto EBBRO = f->createEmbeddingBagByteRowwiseOffsets(
            "EBBRO", phFQData, phWeights, phIndices, phOffsets);
        auto saveEBBRO = f->createSave("saveEBBRO", EBBRO);
        return saveEBBRO->getPlaceholder();
      },
      [m](Tensor *indices, Tensor *lengths, Tensor *offsets) {
        indices->getHandle<int32_t>().randomize(0, embeddingRows - 1,
                                                m->getPRNG());
        // All zeros is a valid tensor for EBB.
        offsets->getHandle<int32_t>().clear(0);
      },
      /* shouldSucceed */ true,
      /* expectedErrorMessage */ "");
}
