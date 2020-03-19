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

#include "tools/loader/Loader.h"
#include "ImporterTestUtils.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"

#include "gtest/gtest.h"

#include "llvm/ADT/StringMap.h"

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

class LoaderTest : public ::testing::Test {
protected:
  // By default constant folding at load time is enabled in general, but we do
  // many tests here loading Constants, so keep it false during these tests by
  // default.
  void SetUp() override { glow::setConstantFoldLoaderOpsFlag(false); }
  void TearDown() override { glow::setConstantFoldLoaderOpsFlag(true); }
};

using namespace glow;

namespace {
const dim_t BATCH_SIZE = 8;
const size_t MINI_BATCH_SIZE = 2;
} // namespace

// A Loader extension class for testing purpose.
class testLoaderExtension : public LoaderExtension {
public:
  static int stage_;
  static size_t index_;
  static Loader *loader_;
  static PlaceholderBindings *bindings_;
  static ProtobufLoader *protobufLoader_;
  static bool destructed_;

  testLoaderExtension() {
    stage_ = 0;
    index_ = 0;
    loader_ = nullptr;
    bindings_ = nullptr;
    protobufLoader_ = nullptr;
    destructed_ = false;
  }

  /// Called once after ONNX or Caffe2 model loading.
  virtual void postModelLoad(Loader &loader, PlaceholderBindings &bindings,
                             ProtobufLoader &protobufLoader,
                             llvm::StringMap<Placeholder *> &outputMap,
                             size_t compilationBatchSize) {
    // To check the method was executed.
    stage_ = 1;

    // To check params are correctly set.
    loader_ = &loader;
    bindings_ = &bindings;
    protobufLoader_ = &protobufLoader;
    EXPECT_EQ(BATCH_SIZE, compilationBatchSize);
  }
  /// Called once at the beginning of the mini-batch inference.
  virtual void inferInitMiniBatch(Loader &loader, PlaceholderBindings &bindings,
                                  size_t minibatchIndex, size_t minibatchSize) {
    // To check the method was executed.
    stage_ = 2;

    // To check params are correctly set.
    loader_ = &loader;
    bindings_ = &bindings;
    index_ = minibatchIndex;
    EXPECT_EQ(MINI_BATCH_SIZE, minibatchSize);
  }
  /// Called once after the completion of the mini-batch inference.
  virtual void inferEndMiniBatch(Loader &loader, PlaceholderBindings &bindings,
                                 size_t minibatchIndex, size_t minibatchSize) {
    // To check the method was executed.
    stage_ = 3;

    // To check params are correctly set.
    loader_ = &loader;
    bindings_ = &bindings;
    index_ = minibatchIndex;
    EXPECT_EQ(MINI_BATCH_SIZE, minibatchSize);
  }
  virtual ~testLoaderExtension() { destructed_ = true; }
};

// A simple Loader second extension class.
class secondTestLoaderExtension : public LoaderExtension {
public:
  static int stage_;
  static bool destructed_;

  secondTestLoaderExtension() {
    stage_ = 0;
    destructed_ = false;
  }

  /// Called once after ONNX or Caffe2 model loading.
  virtual void postModelLoad(Loader &, PlaceholderBindings &, ProtobufLoader &,
                             llvm::StringMap<Placeholder *> &, size_t) {
    stage_ = 1;
  }
  /// Called once at the beginning of the mini-batch inference.
  virtual void inferInitMiniBatch(Loader &, PlaceholderBindings &, size_t,
                                  size_t) {
    stage_ = 2;
  }
  /// Called once after the completion of the mini-batch inference.
  virtual void inferEndMiniBatch(Loader &, PlaceholderBindings &, size_t,
                                 size_t) {
    stage_ = 3;
  }

  virtual ~secondTestLoaderExtension() { destructed_ = true; }
};

// Static class members.
int testLoaderExtension::stage_;
size_t testLoaderExtension::index_;
Loader *testLoaderExtension::loader_;
PlaceholderBindings *testLoaderExtension::bindings_;
ProtobufLoader *testLoaderExtension::protobufLoader_;
bool testLoaderExtension::destructed_;
int secondTestLoaderExtension::stage_;
bool secondTestLoaderExtension::destructed_;

/// This test simulates what can be a Glow applciation (like image_classifier).
TEST_F(LoaderTest, LoaderExtension) {
  {
    std::unique_ptr<ExecutionContext> exContext =
        glow::make_unique<ExecutionContext>();
    PlaceholderBindings &bindings = *exContext->getPlaceholderBindings();
    llvm::StringMap<Placeholder *> outputMap;

    // Create a loader object.
    Loader loader;

    // Register Loader extensions.
    loader.registerExtension(
        std::unique_ptr<LoaderExtension>(new testLoaderExtension()));
    loader.registerExtension(
        std::unique_ptr<LoaderExtension>(new secondTestLoaderExtension()));

    // Load a model
    std::string NetDescFilename(
        GLOW_DATA_PATH "tests/models/caffe2Models/sqr_predict_net.pbtxt");
    std::string NetWeightFilename(
        GLOW_DATA_PATH "tests/models/caffe2Models/empty_init_net.pbtxt");

    Placeholder *output;
    Tensor inputData(ElemKind::FloatTy, {BATCH_SIZE, 2});
    Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {"input"},
                               {&inputData.getType()}, *loader.getFunction());
    output = EXIT_ON_ERR(caffe2LD.getSingleOutput());

    // Check the model was loaded.
    EXPECT_EQ(loader.getFunction()->getNodes().size(), 3);
    auto *save = getSaveNodeFromDest(output);
    ASSERT_TRUE(save);
    auto *pow = llvm::dyn_cast<PowNode>(save->getInput().getNode());
    ASSERT_TRUE(pow);
    auto *input = llvm::dyn_cast<Placeholder>(pow->getLHS().getNode());
    ASSERT_TRUE(input);
    auto *splat = llvm::dyn_cast<SplatNode>(pow->getRHS().getNode());
    ASSERT_TRUE(splat);

    // Get bindings and call post model load extensions.
    ASSERT_EQ(testLoaderExtension::stage_, 0);
    loader.postModelLoad(bindings, caffe2LD, outputMap,
                         inputData.getType().dims()[0]);
    ASSERT_EQ(testLoaderExtension::stage_, 1);
    ASSERT_EQ(testLoaderExtension::loader_, &loader);
    ASSERT_EQ(testLoaderExtension::bindings_, &bindings);
    ASSERT_EQ(testLoaderExtension::protobufLoader_, &caffe2LD);
    ASSERT_EQ(secondTestLoaderExtension::stage_, 1);

    // Allocate tensors to back all inputs and outputs.
    bindings.allocate(loader.getModule()->getPlaceholders());

    // Compile the model.
    CompilationContext cctx = loader.getCompilationContext();
    cctx.bindings = &bindings;
    loader.compile(cctx);

    // Load data to input placeholders.
    updateInputPlaceholdersByName(bindings, loader.getModule(), {"input"},
                                  {&inputData});

    // Run mini-batches.
    for (size_t miniBatchIndex = 0; miniBatchIndex < BATCH_SIZE;
         miniBatchIndex += MINI_BATCH_SIZE) {
      // Minibatch inference initialization of loader extensions.
      loader.inferInitMiniBatch(bindings, miniBatchIndex, MINI_BATCH_SIZE);
      ASSERT_EQ(testLoaderExtension::stage_, 2);
      ASSERT_EQ(testLoaderExtension::index_, miniBatchIndex);
      ASSERT_EQ(testLoaderExtension::loader_, &loader);
      ASSERT_EQ(testLoaderExtension::bindings_, &bindings);
      ASSERT_EQ(testLoaderExtension::protobufLoader_, &caffe2LD);
      ASSERT_EQ(secondTestLoaderExtension::stage_, 2);

      // Perform the inference execution for a minibatch.
      loader.runInference(exContext.get(), BATCH_SIZE);

      // Minibatch inference initialization of loader extensions.
      loader.inferEndMiniBatch(bindings, miniBatchIndex, MINI_BATCH_SIZE);
      ASSERT_EQ(testLoaderExtension::stage_, 3);
      ASSERT_EQ(testLoaderExtension::index_, miniBatchIndex);
      ASSERT_EQ(testLoaderExtension::loader_, &loader);
      ASSERT_EQ(testLoaderExtension::bindings_, &bindings);
      ASSERT_EQ(testLoaderExtension::protobufLoader_, &caffe2LD);
      ASSERT_EQ(secondTestLoaderExtension::stage_, 3);
    }

    // Extension object not destructed yet.
    ASSERT_EQ(testLoaderExtension::destructed_, false);
    ASSERT_EQ(secondTestLoaderExtension::destructed_, false);
  } // End of the loader scope.

  // Check that extensions were properly destructed by the Loader destruction.
  ASSERT_EQ(testLoaderExtension::destructed_, true);
  ASSERT_EQ(secondTestLoaderExtension::destructed_, true);
}
