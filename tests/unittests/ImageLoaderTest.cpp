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

#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "ImporterTestUtils.h"
#include "glow/Base/Image.h"
#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "tools/loader/Loader.h"

#include "gtest/gtest.h"

#include "llvm/ADT/StringMap.h"

#include "tools/loader/ExecutorCore.h"
#include "tools/loader/ExecutorCoreHelperFunctions.h"

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;

class ImageLoaderTest : public ::testing::TestWithParam<bool> {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

template <class PP> static void registerPP(glow::Executor &core) {
  auto pp = []() -> std::unique_ptr<PostProcessOutputDataExtension> {
    return std::make_unique<PP>();
  };
  core.registerPostProcessOutputExtension(pp);
}

/// Load an ONNX model w/ two inputs. Provide two input lists, each
/// with a single 4D numpy file, inputs normalized to (0,1), (-1,1).
TEST_P(ImageLoaderTest, Numpy2InputsNormModeS1U1tTest) {
  const char *argv[6];
  size_t idx = 0;
  argv[idx++] = "test";
  argv[idx++] =
      "-m=" GLOW_DATA_PATH "tests/models/onnxModels/add_2inputs_4D.onnx";
  argv[idx++] =
      "-input-image-list-file=" GLOW_DATA_PATH
      "tests/images/npy/input1List_4D.txt,tests/images/npy/input2List_4D.txt";
  argv[idx++] = "-model-input-name=X";
  argv[idx++] = "-model-input-name=Y";
  argv[idx++] = "-image-mode=neg1to1,0to1";

  glow::Executor core("test", idx, (char **)argv);

  class PP : public PostProcessOutputDataExtension {
  public:
    int processOutputs(const llvm::StringMap<glow::Placeholder *> &PHM,
                       PlaceholderBindings &b,
                       VecVecRef<std::string> inputImageBatchFilenames) {
      return 0;
    }
  };
  registerPP<PP>(core);
  core.executeNetwork();
}

/// Load an ONNX model w/ two inputs. Provide two input lists, each
/// with a single 4D numpy file, inputs normalized to (-128,127), (0,255).
TEST_P(ImageLoaderTest, Numpy2InputsNormModeS8U8Test) {
  const char *argv[6];
  size_t idx = 0;
  argv[idx++] = "test";
  argv[idx++] =
      "-m=" GLOW_DATA_PATH "tests/models/onnxModels/add_2inputs_4D.onnx";
  argv[idx++] =
      "-input-image-list-file=" GLOW_DATA_PATH
      "tests/images/npy/input1List_4D.txt,tests/images/npy/input2List_4D.txt";
  argv[idx++] = "-model-input-name=X";
  argv[idx++] = "-model-input-name=Y";
  argv[idx++] = "-image-mode=neg128to127,0to255";

  glow::Executor core("test", idx, (char **)argv);

  class PP : public PostProcessOutputDataExtension {
  public:
    int processOutputs(const llvm::StringMap<glow::Placeholder *> &PHM,
                       PlaceholderBindings &b,
                       VecVecRef<std::string> inputImageBatchFilenames) {
      doChecks(PHM, b, inputImageBatchFilenames);
      return 0;
    }
    void doChecks(const llvm::StringMap<glow::Placeholder *> &PHM,
                  PlaceholderBindings &b,
                  VecVecRef<std::string> inputImageBatchFilenames) {
      Placeholder *outPH = getOutputForPostProcessing(PHM);
      CHECK(outPH) << "Missing placeholder";

      auto *S = getSaveNodeFromDest(outPH);
      ASSERT_TRUE(S);
      auto *add = llvm::dyn_cast<AddNode>(S->getInput().getNode());
      ASSERT_TRUE(add);

      std::vector<float> exp = {-108, -106, -104, -102, -100, -98, -96, -94,
                                -92,  -90,  -88,  -86,  -84,  -82, -80, -78,
                                -76,  -74,  -72,  -70,  -68,  -66, -64, -62,
                                -60,  -58,  -56,  -54,  -52,  -50, -48, -46};
      Tensor *outT = b.get(outPH);
      CHECK(outT) << "Missing output tensor";
      auto H = outT->getHandle();
      for (size_t i = 0; i < H.size(); i++) {
        float expected = exp[i];
        float value = H.raw(i);
        EXPECT_NEAR(expected, value, 0.001);
      }
    }
  };

  registerPP<PP>(core);
  core.executeNetwork();
}

/// Load an ONNX model w/ two inputs. Provide two input lists, each
/// with a single 4D numpy file, Each channel within each input has it's
/// own mean/stddev value.
TEST_P(ImageLoaderTest, Numpy2InputsMeanStddevTest) {
  const char *argv[8];
  size_t idx = 0;
  argv[idx++] = "test";
  argv[idx++] =
      "-m=" GLOW_DATA_PATH "tests/models/onnxModels/add_2inputs_4D.onnx";
  argv[idx++] =
      "-input-image-list-file=" GLOW_DATA_PATH
      "tests/images/npy/input1List_4D.txt,tests/images/npy/input2List_4D.txt";
  argv[idx++] = "-model-input-name=X";
  argv[idx++] = "-model-input-name=Y";
  argv[idx++] = "-mean=0,1:2,3";
  argv[idx++] = "-stddev=4,5:6,7";

  glow::Executor core("test", idx, (char **)argv);

  class PP : public PostProcessOutputDataExtension {
  public:
    int processOutputs(const llvm::StringMap<glow::Placeholder *> &PHM,
                       PlaceholderBindings &b,
                       VecVecRef<std::string> inputImageBatchFilenames) {
      doChecks(PHM, b, inputImageBatchFilenames);
      return 0;
    }
    void doChecks(const llvm::StringMap<glow::Placeholder *> &PHM,
                  PlaceholderBindings &b,
                  VecVecRef<std::string> inputImageBatchFilenames) {
      Placeholder *outPH = getOutputForPostProcessing(PHM);
      CHECK(outPH) << "Missing placeholder";

      auto *S = getSaveNodeFromDest(outPH);
      ASSERT_TRUE(S);
      auto *add = llvm::dyn_cast<AddNode>(S->getInput().getNode());
      ASSERT_TRUE(add);

      std::vector<float> exp = {
          3.833,  4.250,  4.667,  5.083,  5.500,  5.917,  6.333,  6.750,
          7.167,  7.583,  8.000,  8.417,  8.833,  9.250,  9.667,  10.083,
          8.286,  8.629,  8.971,  9.314,  9.657,  10.000, 10.343, 10.686,
          11.029, 11.371, 11.714, 12.057, 12.400, 12.743, 13.086, 13.429};

      Tensor *outT = b.get(outPH);
      CHECK(outT) << "Missing output tensor";
      auto H = outT->getHandle();
      for (size_t i = 0; i < H.size(); i++) {
        float expected = exp[i];
        float value = H.raw(i);
        EXPECT_NEAR(expected, value, 0.001);
      }
    }
  };

  registerPP<PP>(core);
  core.executeNetwork();
}

/// Load an ONNX model w/ two inputs. Provide two input lists, one with
/// a single 4D numpy file other with a single 3D file (which gets expanded to
/// 4D) both with NCHW layout. Each channel within each input has it's own
/// mean/stddev value.
TEST_P(ImageLoaderTest, Numpy2InputsMeanStddev4D3DTest) {
  const char *argv[7];
  size_t idx = 0;
  argv[idx++] = "test";
  argv[idx++] =
      "-m=" GLOW_DATA_PATH "tests/models/onnxModels/add_2inputs_4D.onnx";
  argv[idx++] =
      "-input-image-list-file=" GLOW_DATA_PATH
      "tests/images/npy/input1List_4D.txt,tests/images/npy/input2List_3D_2.txt";
  argv[idx++] = "-model-input-name=X";
  argv[idx++] = "-model-input-name=Y";
  argv[idx++] = "-mean=0,1:2,3";
  argv[idx++] = "-stddev=4,5:6,7";

  glow::Executor core("test", idx, (char **)argv);

  class PP : public PostProcessOutputDataExtension {
  public:
    int processOutputs(const llvm::StringMap<glow::Placeholder *> &PHM,
                       PlaceholderBindings &b,
                       VecVecRef<std::string> inputImageBatchFilenames) {
      doChecks(PHM, b, inputImageBatchFilenames);
      return 0;
    }
    void doChecks(const llvm::StringMap<glow::Placeholder *> &PHM,
                  PlaceholderBindings &b,
                  VecVecRef<std::string> inputImageBatchFilenames) {
      Placeholder *outPH = getOutputForPostProcessing(PHM);
      CHECK(outPH) << "Missing placeholder";

      auto *S = getSaveNodeFromDest(outPH);
      ASSERT_TRUE(S);
      auto *add = llvm::dyn_cast<AddNode>(S->getInput().getNode());
      ASSERT_TRUE(add);

      std::vector<float> exp = {
          3.833,  4.250,  4.667,  5.083,  5.500,  5.917,  6.333,  6.750,
          7.167,  7.583,  8.000,  8.417,  8.833,  9.250,  9.667,  10.083,
          8.286,  8.629,  8.971,  9.314,  9.657,  10.000, 10.343, 10.686,
          11.029, 11.371, 11.714, 12.057, 12.400, 12.743, 13.086, 13.429};

      Tensor *outT = b.get(outPH);
      CHECK(outT) << "Missing output tensor";
      auto H = outT->getHandle();
      for (size_t i = 0; i < H.size(); i++) {
        float expected = exp[i];
        float value = H.raw(i);
        EXPECT_NEAR(expected, value, 0.001);
      }
    }
  };

  registerPP<PP>(core);
  core.executeNetwork();
}

/// Load an ONNX model w/ two inputs. Provide two input lists, each
/// with a single 3D numpy file w/ no layout, Each input has it's
/// own mean/stddev value.
TEST_P(ImageLoaderTest, Numpy2InputsMeanStddev3DNoLayoutTest) {
  const char *argv[8];
  size_t idx = 0;
  argv[idx++] = "test";
  argv[idx++] =
      "-m=" GLOW_DATA_PATH "tests/models/onnxModels/add_2inputs_3D.onnx";
  argv[idx++] =
      "-input-image-list-file=" GLOW_DATA_PATH
      "tests/images/npy/input1List_3D.txt,tests/images/npy/input2List_3D.txt";
  argv[idx++] = "-model-input-name=X";
  argv[idx++] = "-model-input-name=Y";
  argv[idx++] = "-mean=1:2";
  argv[idx++] = "-stddev=4:5";
  argv[idx++] = "-image-layout=NonImage,NonImage";

  glow::Executor core("test", idx, (char **)argv);

  class PP : public PostProcessOutputDataExtension {
  public:
    int processOutputs(const llvm::StringMap<glow::Placeholder *> &PHM,
                       PlaceholderBindings &b,
                       VecVecRef<std::string> inputImageBatchFilenames) {
      doChecks(PHM, b, inputImageBatchFilenames);
      return 0;
    }

    void doChecks(const llvm::StringMap<glow::Placeholder *> &PHM,
                  PlaceholderBindings &b,
                  VecVecRef<std::string> inputImageBatchFilenames) {
      Placeholder *outPH = getOutputForPostProcessing(PHM);
      CHECK(outPH) << "Missing placeholder";

      auto *S = getSaveNodeFromDest(outPH);
      ASSERT_TRUE(S);
      auto *add = llvm::dyn_cast<AddNode>(S->getInput().getNode());
      ASSERT_TRUE(add);

      std::vector<float> exp = {3.85, 4.3, 4.75, 5.2, 5.65, 6.1, 6.55, 7.};

      Tensor *outT = b.get(outPH);
      CHECK(outT) << "Missing output tensor";
      auto H = outT->getHandle();
      for (size_t i = 0; i < H.size(); i++) {
        float expected = exp[i];
        float value = H.raw(i);
        EXPECT_NEAR(expected, value, 0.001);
      }
    }
  };

  registerPP<PP>(core);
  core.executeNetwork();
}

INSTANTIATE_TEST_SUITE_P(BN, ImageLoaderTest, ::testing::Values(true));
