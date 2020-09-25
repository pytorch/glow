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

#include "BackendTestUtils.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"
#include "glow/IR/Instrs.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Quantization/Base/Base.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

using namespace glow;

class OperatorStatelessTest : public BackendStatelessTest {};

class OperatorTest : public BackendTest {
protected:
  PlaceholderBindings bindings_;
  /// Use this for storing tensors that are unowned, i.e. if they would normally
  /// be stack local and so they cannot be read in TearDown.
  std::vector<Tensor> unownedTensors_;
  virtual void SetUp() override {
    // Skip stripping the module so that we can inspect Constants after
    // compilation.
    EE_.setSkipModuleStrip(true);
  }

  virtual void TearDown() override {
    if (::testing::Test::IsSkipped()) {
      return;
    }

    EXPECT_TRUE(F_->getNodes().size() != 0)
        << "Functions should have nodes at the end of the test.";

    ASSERT_TRUE(F_->verify(&EE_.getBackend()))
        << "Function must pass verification.";

    // Now export the model to later import it back in.
    llvm::SmallString<64> path;
    auto tempFileRes =
        llvm::sys::fs::createTemporaryFile("exporter", "output.onnxtxt", path);
    ASSERT_EQ(tempFileRes.value(), 0)
        << "Failed to create temp file to write into.";
    std::string pathToModel(path.c_str());

    Error err = Error::empty();
    ONNXModelWriter onnxWR(pathToModel, *F_, 7, 9, &err,
                           /* textMode */ true, /* zipMode */ false,
                           /* useGlowCustomOps */ true);
    ASSERT_FALSE(ERR_TO_BOOL(std::move(err))) << "Error exporting model";

    // Now that we've exported, load it back into a new module/function, run it,
    // and compare results from the original run.
    PlaceholderBindings loadedBindings;
    ExecutionEngine loadedEE{getBackendName()};
    Module &loadedMod = loadedEE.getModule();
    Function *loadedF = loadedMod.createFunction(F_->getName());
    {
      Error err = Error::empty();
      // Note: We disable constant folding here because we only need it to
      // calculate shapes that are the result of constant compute in the proto,
      // but this won't be the case when using useGlowCustomOps exporting.
      ONNXModelLoader onnxLD(pathToModel, {}, {}, *loadedF, &err,
                             /* zipMode */ false, /* perNodeOpts */ nullptr,
                             /* disableConstFoldInLoader */ true,
                             /* loadIntoExistingModule */ false,
                             &loadedEE.getBackend());
      if (ERR_TO_BOOL(std::move(err))) {
        llvm::sys::fs::remove(pathToModel);
        FAIL() << "Error loading exported model";
      }
    }

    // Note that we use the backend for verification here, because the function
    // is post optimization pipeline and so has backend-specific requirements
    // built in, e.g. for required layout.
    ASSERT_TRUE(loadedF->verify(&loadedEE.getBackend()))
        << "Loaded Function must pass verification";

    // String representations of original and loaded functions must be the same.
    // Note that we skip printing users for Storage because some tests have
    // other Functions sharing Storage for testing purposes.
    EXPECT_EQ(F_->toString(/* skipUsersForStorage */ true),
              loadedF->toString(/* skipUsersForStorage */ true));

    // Copy over inputs from previous bindings to newly loaded bindings. We have
    // new Placeholders so can't reuse the bindings from before.
    for (const auto &p : bindings_.pairs()) {
      if (!isInput(p.first, *F_)) {
        continue;
      }

      // Look for an input PH by the same name as the original Function.
      Placeholder *inputPH =
          loadedMod.getPlaceholderByNameSlow(p.first->getName());
      ASSERT_TRUE(inputPH);
      loadedBindings.insert(inputPH, p.second.getUnowned(inputPH->dims()));
    }

    // Allocate all other PHs/tensors that need it (i.e. result PHs/tensors).
    loadedBindings.allocate(loadedF->findPlaceholders());

    // Skip the optimization pipeline for loadedF (via onlyLowerFuns), as we
    // already passed it through the optimization pipeline before exporting it.
    CompilationContext cctx;
    cctx.optimizationOpts.onlyLowerFuns.insert(loadedF);
    loadedEE.compile(cctx);
    loadedEE.run(loadedBindings);

    // Now bitwise-equal compare result tensors from before and after.
    for (const auto &p : bindings_.pairs()) {
      const Placeholder *resultPH = p.first;
      if (!isOutput(resultPH, *F_)) {
        continue;
      }
      const Tensor &resultT = p.second;

      // Find the result PH by the same name in the loaded Function.
      Placeholder *loadedResultPH =
          loadedMod.getPlaceholderByNameSlow(resultPH->getName());
      ASSERT_TRUE(loadedResultPH);
      const Tensor *loadedResultT = loadedBindings.get(loadedResultPH);

      EXPECT_TRUE(resultT.isBitwiseEqual(*loadedResultT, /* verbose */ true));
    }

    llvm::sys::fs::remove(pathToModel);
  }
};

/// Helper to create a Placeholder; if \p T is quantized, then it will include a
/// dummy scale and offset, otherwise it will not.
static Placeholder *createPlaceholderConditionallyQuantized(
    Module &mod, ElemKind T, llvm::ArrayRef<dim_t> dims, llvm::StringRef name,
    bool isTrainable, llvm::StringRef layout = ANY_LAYOUT) {
  return isQuantizedElemKind(T)
             ? mod.createPlaceholder(T, dims, 1.0, 0, name, isTrainable, layout)
             : mod.createPlaceholder(T, dims, name, isTrainable, layout);
}

/// Helper to get a unique Type; if \p T is quantized, then it will include a
/// dummy scale and offset, otherwise it will not.
static TypeRef uniqueTypeConditionallyQuantized(Module &mod, ElemKind T,
                                                llvm::ArrayRef<dim_t> dims) {
  return isQuantizedElemKind(T) ? mod.uniqueType(T, dims, 1.0, 0)
                                : mod.uniqueType(T, dims);
}

/// Helper to create a Tensor; if \p T is quantized, then it will include a
/// dummy scale and offset, otherwise it will not.
static Tensor createTensorConditionallyQuantized(ElemKind T,
                                                 llvm::ArrayRef<dim_t> dims) {
  return isQuantizedElemKind(T) ? Tensor(T, dims, 1.0, 0) : Tensor(T, dims);
}

template <typename DataType>
glow::Handle<bool>
lessHelper(glow::PlaceholderBindings &bindings, glow::Module &mod,
           glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
           llvm::ArrayRef<DataType> xValues, llvm::ArrayRef<DataType> yValues,
           llvm::ArrayRef<dim_t> xDims, llvm::ArrayRef<dim_t> yDims) {
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, xDims, "X",
                                                    /* isTrainable */ false);

  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, yDims, "Y",
                                                    /* isTrainable */ false);

  bindings.allocate(llvm::dyn_cast<Placeholder>(X))->getHandle<DataType>() =
      xValues;

  bindings.allocate(llvm::dyn_cast<Placeholder>(Y))->getHandle<DataType>() =
      yValues;

  auto *cmpr =
      F->createNodeWithBroadcast<CmpLTNode>("cmpLT", /* axis */ -1, X, Y);

  auto *save = F->createSave("save", cmpr);
  auto *saveAlloc = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  return saveAlloc->getHandle<bool>();
}

TEST_P(OperatorTest, less_int8) {
  CHECK_IF_ENABLED();

  int8_t xValues[] = {3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

                      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

                      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

                      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  int8_t yValues[] = {3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

                      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

                      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

                      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  dim_t xDims[] = {2, 2, 4, 4};
  dim_t yDims[] = {2, 2, 4, 4};

  Handle<bool> saveH =
      lessHelper<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, xValues,
                         yValues, xDims, yDims);

  bool refResults[] = {
      false, true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,
  };

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, less_floatCases) {
  CHECK_IF_ENABLED();

  float xValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  float yValues[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                        yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_float16Cases) {
  CHECK_IF_ENABLED();

  float16 xValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  float16 yValues[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<float16>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                          xValues, yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_bfloat16Cases) {
  CHECK_IF_ENABLED();

  bfloat16 xValues[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  bfloat16 yValues[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<bfloat16>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                           xValues, yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_int64Cases) {
  CHECK_IF_ENABLED();

  int64_t xValues[] = {1, 2, 3, 4, 5};

  int64_t yValues[] = {5, 4, 3, 2, 1};

  dim_t xDims[] = {5};
  dim_t yDims[] = {5};

  Handle<bool> saveH =
      lessHelper<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy, xValues,
                          yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i}));
  }
}

TEST_P(OperatorTest, less_float) {
  CHECK_IF_ENABLED();

  float xValues[] = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  float yValues[] = {3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  dim_t xDims[] = {2, 2, 4, 4};
  dim_t yDims[] = {2, 2, 4, 4};

  Handle<bool> saveH =
      lessHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                        yValues, xDims, yDims);

  bool refResults[] = {
      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,

      true,  true,  true,  true, false, false, false, true,
      false, false, false, true, true,  true,  false, true,
  };

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, less_broadcast_float) {
  CHECK_IF_ENABLED();

  float xValues[] = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

                     1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
                     7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  float yValues[] = {3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

                     3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
                     4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  dim_t xDims[] = {2, 2, 4, 4};
  dim_t yDims[] = {1, 2, 4, 4};

  Handle<bool> saveH =
      lessHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                        yValues, xDims, yDims);

  bool refResults[] = {true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true,

                       true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true,

                       true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true,

                       true,  true,  true,  true, false, false, false, true,
                       false, false, false, true, true,  true,  false, true};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, less_int32Cases) {
  CHECK_IF_ENABLED();

  int32_t xValues[] = {1, 2, 3, 4, 5};
  int32_t yValues[] = {5, 4, 3, 2, 1};

  dim_t xDims[] = {1, 1, 1, 5};
  dim_t yDims[] = {1, 1, 1, 5};

  Handle<bool> saveH =
      lessHelper<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy, xValues,
                          yValues, xDims, yDims);

  bool refResults[] = {true, true, false, false, false};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

template <typename DataType>
glow::Handle<DataType>
whereHelper(glow::PlaceholderBindings &bindings, glow::Module &mod,
            glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
            llvm::ArrayRef<DataType> xValues, llvm::ArrayRef<DataType> yValues,
            llvm::ArrayRef<bool> cValues, llvm::ArrayRef<dim_t> xDims,
            llvm::ArrayRef<dim_t> yDims, llvm::ArrayRef<dim_t> cDims) {
  auto *cond = createPlaceholderConditionallyQuantized(mod, ElemKind::BoolTy,
                                                       cDims, "cond", false);
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, xDims, "X",
                                                    DTy != ElemKind::FloatTy);

  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, yDims, "Y",
                                                    DTy != ElemKind::FloatTy);

  bindings.allocate(llvm::dyn_cast<Placeholder>(cond))->getHandle<bool>() =
      cValues;

  bindings.allocate(llvm::dyn_cast<Placeholder>(X))->getHandle<DataType>() =
      xValues;

  bindings.allocate(llvm::dyn_cast<Placeholder>(Y))->getHandle<DataType>() =
      yValues;

  auto *whr = F->createNodeWithBroadcast<SelectNode>("Select", /* axis */ -1,
                                                     cond, X, Y);

  auto *save = F->createSave("save", whr);
  auto *saveAlloc = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  return saveAlloc->getHandle<DataType>();
}

TEST_P(OperatorTest, where_2d_broadcast_x_y_i8) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<int8_t, 16> xValues = {3, 5, 7};

  llvm::SmallVector<int8_t, 16> yValues = {2, 4, 6};

  llvm::SmallVector<bool, 4> cValues = {1, 0, 1};

  llvm::SmallVector<dim_t, 4> condDims = {3, 1, 1};

  llvm::SmallVector<dim_t, 4> xDims = {1, 3, 1};
  llvm::SmallVector<dim_t, 4> yDims = {3, 1, 1};

  Handle<int8_t> saveH =
      whereHelper<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, xValues,
                          yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<int8_t, 16> refResults = {3, 5, 7, 4, 4, 4, 3, 5, 7};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        EXPECT_EQ(refResults[counter++], saveH.at({i, j, k}));
      }
    }
  }
}

TEST_P(OperatorTest, where_2d_wise_i8) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<int8_t, 16> xValues = {
      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<int8_t, 16> yValues = {
      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<bool, 4> cValues = {1, 0, 1, 0};

  llvm::SmallVector<dim_t, 4> condDims = {2, 2, 1, 1};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<int8_t> saveH =
      whereHelper<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, xValues,
                          yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<int8_t, 16> refResults = {
      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, where_2d_wise_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<float, 16> xValues = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  llvm::SmallVector<float, 16> yValues = {
      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  llvm::SmallVector<bool, 4> cValues = {1, 0, 1, 0};

  llvm::SmallVector<dim_t, 4> condDims = {2, 2, 1, 1};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<float> saveH =
      whereHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                         yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<float, 16> refResults = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k, f}));
        }
      }
    }
  }
}

TEST_P(OperatorTest, where_row_wise_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<bool, 4> cValues = {1, 1, 1, 0, 0, 1, 0, 0};

  llvm::SmallVector<dim_t, 4> condDims = {2, 4, 1};

  llvm::SmallVector<dim_t, 4> xDims = {2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 4, 4};

  llvm::SmallVector<float, 16> xValues = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f,

      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 3.0f, 5.0f, 7.0f, 1.0f};

  llvm::SmallVector<float, 16> yValues = {
      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f, 0.0f, 6.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f};

  Handle<float> saveH =
      whereHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                         yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<float, 16> refResults = {
      1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      7.0f, 8.0f, 9.0f, 2.0f, 5.0f, 9.0f, 2.0f, 6.0f,

      3.0f, 4.0f, 5.0f, 7.0f, 4.0f, 5.0f, 6.0f, 3.0f,
      4.0f, 2.0f, 1.0f, 8.0f, 5.0f, 9.0f, 2.0f, 6.0f,
  };

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k}));
      }
    }
  }
}

TEST_P(OperatorTest, where_element_wise_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<dim_t, 4> condDims = {1, 4, 4};

  llvm::SmallVector<dim_t, 4> xDims = {1, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {1, 4, 4};

  llvm::SmallVector<bool, 4> cValues = {1, 1, 1, 0, 0, 1, 0, 0,
                                        0, 1, 0, 1, 1, 0, 1, 0};

  llvm::SmallVector<float, 16> xValues = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f,
                                          6.0f, 3.0f, 7.0f, 8.0f, 9.0f, 2.0f,
                                          3.0f, 5.0f, 7.0f, 1.0f};

  llvm::SmallVector<float, 16> yValues = {3.0f, 4.0f, 5.0f, 7.0f, 2.0f, 1.0f,
                                          0.0f, 6.0f, 4.0f, 2.0f, 1.0f, 8.0f,
                                          5.0f, 9.0f, 2.0f, 6.0f};

  Handle<float> saveH =
      whereHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                         yValues, cValues, xDims, yDims, condDims);

  llvm::SmallVector<float, 16> refResults = {1.0f, 2.0f, 3.0f, 7.0f, 2.0f, 5.0f,
                                             0.0f, 6.0f, 4.0f, 8.0f, 1.0f, 2.0f,
                                             3.0f, 9.0f, 7.0f, 6.0f};

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        EXPECT_FLOAT_EQ(refResults[counter++], saveH.at({i, j, k}));
      }
    }
  }
}

struct NMSMetaData {
  int centerPoint{0};
  size_t maxOutputPerClass{0};
  float iouThreshold{0.0};
  float scoreThreshold{0.0};
};

struct SelectedBox {
  int batchIndex{0};
  int classIndex{0};
  int boxIndex{0};
};

struct Box {
  float x;
  float y;
  float h;
  float w;
};

template <typename DataType, typename outType = int64_t>
static Handle<outType> testNonMaxSuppression(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind DTy, llvm::ArrayRef<dim_t> boxesDims,
    llvm::ArrayRef<dim_t> scoresDims, llvm::ArrayRef<DataType> boxesData,
    llvm::ArrayRef<DataType> classes, llvm::ArrayRef<SelectedBox> refResults,
    llvm::ArrayRef<int32_t> refNumSelected, const NMSMetaData &metaData,
    bool isV4) {

  // NHW
  auto *boxes = createPlaceholderConditionallyQuantized(mod, DTy, boxesDims,
                                                        "boxes", false);

  auto *scores = createPlaceholderConditionallyQuantized(mod, DTy, scoresDims,
                                                         "scores", false);

  NonMaxSuppressionNode *nms = nullptr;

  if (isV4) {
    nms = F->createNonMaxSuppressionV4(
        "NMS", boxes, scores, metaData.centerPoint, metaData.maxOutputPerClass,
        metaData.iouThreshold, metaData.scoreThreshold);
  } else {
    nms = F->createNonMaxSuppressionONNX(
        "NMS", boxes, scores, metaData.centerPoint, metaData.maxOutputPerClass,
        metaData.iouThreshold, metaData.scoreThreshold);
  }

  auto *saveIndices = F->createSave("save", nms->getIndices());
  auto *saveNumSelected =
      F->createSave("numSelected", nms->getNumberOfSelectedIndices());
  auto *result = bindings.allocate(saveIndices->getPlaceholder());
  auto *result2 = bindings.allocate(saveNumSelected->getPlaceholder());

  bindings.allocate(boxes)->getHandle<DataType>() = boxesData;
  bindings.allocate(scores)->getHandle<DataType>() = classes;

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE.compile(cctx);
  EE.run(bindings);

  Handle<outType> result2H = result2->getHandle<outType>();
  for (dim_t i = 0; i < (dim_t)refNumSelected.size(); ++i) {
    EXPECT_EQ(result2H.at({i}), refNumSelected[i]);
  }

  Handle<outType> resultH = result->getHandle<outType>();

  if (isV4) {
    for (dim_t i = 0; i < (dim_t)metaData.maxOutputPerClass; ++i) {
      EXPECT_EQ(refResults[i].boxIndex, resultH.at({i}));
    }
  } else {
    for (dim_t i = 0; i < (dim_t)metaData.maxOutputPerClass; ++i) {
      EXPECT_EQ(refResults[i].batchIndex, resultH.at({i, (dim_t)0}));
      EXPECT_EQ(refResults[i].classIndex, resultH.at({i, (dim_t)1}));
      EXPECT_EQ(refResults[i].boxIndex, resultH.at({i, (dim_t)2}));
    }
  }

  return resultH;
}

template <typename DataType, typename outType = int64_t>
static Handle<float> testNonMaxSuppressionWithGather(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind DTy, llvm::ArrayRef<dim_t> boxesDims,
    llvm::ArrayRef<dim_t> scoresDims, llvm::ArrayRef<dim_t> boxIndicesDim,
    llvm::ArrayRef<DataType> boxesData, llvm::ArrayRef<DataType> classes,
    llvm::ArrayRef<int32_t> boxIndicesData, llvm::ArrayRef<Box> refBoxResults,
    llvm::ArrayRef<int32_t> refNumSelected, const NMSMetaData &metaData,
    bool isV4) {
  // NHW
  auto *boxes = createPlaceholderConditionallyQuantized(mod, DTy, boxesDims,
                                                        "boxes", false);

  auto *scores = createPlaceholderConditionallyQuantized(mod, DTy, scoresDims,
                                                         "scores", false);

  auto *boxIndices = createPlaceholderConditionallyQuantized(
      mod, ElemKind::Int32ITy, boxIndicesDim, "boxIndices", false);

  NonMaxSuppressionNode *nms = nullptr;

  unsigned axis = 1;
  if (isV4) {
    nms = F->createNonMaxSuppressionV4(
        "NMS", boxes, scores, metaData.centerPoint, metaData.maxOutputPerClass,
        metaData.iouThreshold, metaData.scoreThreshold);
    axis = 0;
  } else {

    nms = F->createNonMaxSuppressionONNX(
        "NMS", boxes, scores, metaData.centerPoint, metaData.maxOutputPerClass,
        metaData.iouThreshold, metaData.scoreThreshold);
  }

  // extract all the box indices
  auto *gthI =
      F->createGather("gatherBoxIndices", nms->getIndices(), boxIndices, axis);
  auto *gthB = F->createGather("gatherClassIndices", boxes, gthI, axis);
  Node *fltn2 = nullptr;

  if (isV4) {
    fltn2 = gthB;
  } else {
    fltn2 = F->createFlatten("flatten", gthB, 2);
  }

  auto *saveBoxes = F->createSave("saveBoxes", fltn2);
  auto saveNumSelected =
      F->createSave("numSelected", nms->getNumberOfSelectedIndices());

  auto *result = bindings.allocate(saveBoxes->getPlaceholder());
  auto *result2 = bindings.allocate(saveNumSelected->getPlaceholder());

  bindings.allocate(boxes)->getHandle<DataType>() = boxesData;
  bindings.allocate(scores)->getHandle<DataType>() = classes;
  bindings.allocate(boxIndices)->getHandle<int32_t>() = boxIndicesData;

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  EE.compile(cctx);
  EE.run(bindings);

  Handle<outType> result2H = result2->getHandle<outType>();
  for (dim_t i = 0; i < (dim_t)refNumSelected.size(); ++i) {
    EXPECT_EQ(result2H.at({i}), refNumSelected[i]);
  }

  Handle<float> resultH = result->getHandle<float>();

  for (dim_t i = 0; i < (dim_t)refBoxResults.size(); ++i) {
    EXPECT_EQ(refBoxResults[i].x, resultH.at({i, (dim_t)0}));
    EXPECT_EQ(refBoxResults[i].y, resultH.at({i, (dim_t)1}));
    EXPECT_EQ(refBoxResults[i].h, resultH.at({i, (dim_t)2}));
    EXPECT_EQ(refBoxResults[i].w, resultH.at({i, (dim_t)3}));
  }

  return resultH;
}

TEST_P(OperatorTest, nms_center_point_box_with_gather_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 6};
  llvm::SmallVector<dim_t, 1> boxIndexesDms = {1};

  llvm::SmallVector<float, 24> boxes = {
      0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
      0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<int32_t, 1> boxIndices = {2};
  llvm::SmallVector<Box, 3> refResults = {
      {0.5, 10.5, 1.0, 1.0}, {0.5, 0.5, 1.0, 1.0}, {0.5, 0.5, 1.0, 1.0}};
  NMSMetaData metaData = {1, 3, 0.5, 0.4};
  llvm::SmallVector<int32_t, 1> refNumSelected = {2};

  testNonMaxSuppressionWithGather<float>(
      bindings_, mod_, F_, EE_, ElemKind::FloatTy, boxesDims, scoresDims,
      boxIndexesDms, boxes, classes, boxIndices, refResults, refNumSelected,
      metaData, false);
}

TEST_P(OperatorTest, nms_v4_center_point_box_with_gather_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {6, 4};
  llvm::SmallVector<dim_t, 1> scoresDims = {6};
  llvm::SmallVector<dim_t, 1> boxIndexesDims = {3};

  llvm::SmallVector<float, 24> boxes = {
      0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
      0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<int32_t, 3> boxIndices = {0, 1, 2};
  llvm::SmallVector<Box, 3> refResults = {
      {0.5, 10.5, 1.0, 1.0}, {0.5, 0.5, 1.0, 1.0}, {0.5, 0.5, 1.0, 1.0}};
  NMSMetaData metaData = {1, 3, 0.5, 0.4};
  llvm::SmallVector<int32_t, 1> refNumSelected{2};

  testNonMaxSuppressionWithGather<float>(
      bindings_, mod_, F_, EE_, ElemKind::FloatTy, boxesDims, scoresDims,
      boxIndexesDims, boxes, classes, boxIndices, refResults, refNumSelected,
      metaData, true);
}

TEST_P(OperatorTest, nms_center_point_box_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 6};
  llvm::SmallVector<float, 24> boxes = {
      0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
      0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 3> refResults = {
      {0, 0, 3}, {0, 0, 0}, {0, 0, 5}};
  NMSMetaData metaData = {1, 3, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{3};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_v4_center_point_box_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {6, 4};
  llvm::SmallVector<dim_t, 1> scoresDims = {6};
  llvm::SmallVector<float, 24> boxes = {
      0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
      0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 3> refResults = {
      {0, 0, 3}, {0, 0, 0}, {0, 0, 5}};
  NMSMetaData metaData = {1, 3, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{3};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, true);
}

TEST_P(OperatorTest, nms_flipped_coordinates_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 6};
  llvm::SmallVector<float, 24> boxes = {
      1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
      0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};
  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 3> refResults = {
      {0, 0, 3}, {0, 0, 0}, {0, 0, 5}};
  NMSMetaData metaData = {0, 3, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{3};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_identical_boxes_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 10, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 10};
  llvm::SmallVector<float, 40> boxes = {
      0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
      1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
      0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
  llvm::SmallVector<float, 10> classes = {0.9, 0.9, 0.9, 0.9, 0.9,
                                          0.9, 0.9, 0.9, 0.9, 0.9};
  llvm::SmallVector<SelectedBox, 3> refResults = {{0, 0, 0}};
  NMSMetaData metaData = {0, 1, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{1};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_limit_output_size_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 6};
  llvm::SmallVector<float, 24> boxes = {
      0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
      0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 2> refResults = {{0, 0, 3}, {0, 0, 0}};
  NMSMetaData metaData = {0, 2, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{2};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_single_box_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 1, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 1};
  llvm::SmallVector<float, 4> boxes = {0.0, 0.0, 1.0, 1.0};
  llvm::SmallVector<float, 1> classes = {0.9};
  llvm::SmallVector<SelectedBox, 1> refResults = {
      {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  NMSMetaData metaData = {0, 3, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{1};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_by_iou_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 6};
  llvm::SmallVector<float, 24> boxes = {
      0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
      0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 2> refResults = {
      {0, 0, 3}, {0, 0, 0}, {0, 0, 5}};
  NMSMetaData metaData = {0, 3, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{3};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_by_iou_and_scores_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 6};
  llvm::SmallVector<float, 24> boxes = {
      0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
      0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
  llvm::SmallVector<float, 6> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 2> refResults = {{0, 0, 3}, {0, 0, 0}};
  NMSMetaData metaData = {0, 2, 0.5, 0.4};
  llvm::SmallVector<int32_t, 1> refNumSelected{2};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_two_batches_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {2, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {2, 1, 6};
  llvm::SmallVector<float, 48> boxes = {
      0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
      0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
      0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
      0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
  llvm::SmallVector<float, 12> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
                                          0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 4> refResults = {
      {0, 0, 3}, {0, 0, 0}, {1, 0, 3}, {1, 0, 0}};
  NMSMetaData metaData = {0, 2, 0.5, 0.0};
  llvm::SmallVector<int32_t, 2> refNumSelected{2, 2};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_two_classes_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 6, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 2, 6};
  llvm::SmallVector<float, 24> boxes = {
      0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
      0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
  llvm::SmallVector<float, 12> classes = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
                                          0.9, 0.75, 0.6, 0.95, 0.5, 0.3};
  llvm::SmallVector<SelectedBox, 4> refResults = {
      {0, 0, 3}, {0, 0, 0}, {0, 1, 3}, {0, 1, 0}};
  NMSMetaData metaData = {0, 2, 0.5, 0.4};
  llvm::SmallVector<int32_t, 1> refNumSelected{4};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

TEST_P(OperatorTest, nms_two_boxes_float) {
  CHECK_IF_ENABLED();
  llvm::SmallVector<dim_t, 3> boxesDims = {1, 2, 4};
  llvm::SmallVector<dim_t, 3> scoresDims = {1, 1, 2};
  llvm::SmallVector<float, 4> boxes = {0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 0.9};
  llvm::SmallVector<float, 2> classes = {0.8, 0.9};
  llvm::SmallVector<SelectedBox, 1> refResults = {{0, 0, 1}};
  NMSMetaData metaData = {0, 1, 0.5, 0.0};
  llvm::SmallVector<int32_t, 1> refNumSelected{1};

  testNonMaxSuppression<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                               boxesDims, scoresDims, boxes, classes,
                               refResults, refNumSelected, metaData, false);
}

/// Helper function to test AudioSpectrogram node.
template <size_t windowCount, size_t windowSize, bool magnitudeSquared>
static FunctionTensorPair
createAndInitBasicAudioSpectrogramTest(glow::PlaceholderBindings &bindings,
                                       glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Create random input audio signal.
  dim_t windowStride = 320;
  dim_t inputLength = windowSize + (windowCount - 1) * windowStride;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {inputLength}, "input",
                                      false /* isTrainable */);
  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

  // Create AudioSpectrogram node.
  auto *audioSpec = F->createAudioSpectrogram(
      "audio_spectrogram", input, windowSize, windowStride, magnitudeSquared);
  auto *res = F->createSave("save", audioSpec);
  auto *resultTensor = bindings.allocate(res->getPlaceholder());
  return std::make_pair(F, resultTensor);
}

#define TEST_AUDIO_SPECTROGRAM(WCOUNT, WSIZE, MSQUARED, TOL)                   \
  TEST_P(OperatorStatelessTest,                                                \
         AudioSpectrogram_##WCOUNT##x##WSIZE##_##MSQUARED##_Float) {           \
    ENABLED_BACKENDS("Interpreter", "CPU");                                    \
    compareAgainstInterpreter(                                                 \
        getBackendName(),                                                      \
        createAndInitBasicAudioSpectrogramTest<WCOUNT, WSIZE, MSQUARED>,       \
        ElemKind::FloatTy, ElemKind::FloatTy, TOL);                            \
  }

/// Test one window magnitude spectrograms.
TEST_AUDIO_SPECTROGRAM(1, 2, false, 1e-6)
TEST_AUDIO_SPECTROGRAM(1, 4, false, 1e-6)
TEST_AUDIO_SPECTROGRAM(1, 8, false, 1e-6)
TEST_AUDIO_SPECTROGRAM(1, 16, false, 1e-6)
TEST_AUDIO_SPECTROGRAM(1, 32, false, 1e-6)
TEST_AUDIO_SPECTROGRAM(1, 64, false, 5e-6)
TEST_AUDIO_SPECTROGRAM(1, 128, false, 5e-6)
TEST_AUDIO_SPECTROGRAM(1, 256, false, 1e-5)
TEST_AUDIO_SPECTROGRAM(1, 512, false, 5e-5)
TEST_AUDIO_SPECTROGRAM(1, 1024, false, 5e-5)

/// Test multiple window magnitude spectrograms.
TEST_AUDIO_SPECTROGRAM(2, 256, false, 1e-5)
TEST_AUDIO_SPECTROGRAM(3, 320, false, 1e-5)
TEST_AUDIO_SPECTROGRAM(4, 640, false, 5e-5)

/// Test multiple window power spectrograms.
TEST_AUDIO_SPECTROGRAM(2, 256, true, 5e-4)
TEST_AUDIO_SPECTROGRAM(3, 320, true, 5e-4)
TEST_AUDIO_SPECTROGRAM(4, 640, true, 1e-3)

/// Helper function to test MFCC node.
template <size_t winNum, size_t specLen>
static FunctionTensorPair
createAndInitBasicMFCCTest(glow::PlaceholderBindings &bindings,
                           glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // Create random input spectrogram.
  auto *spectrogram =
      mod.createPlaceholder(ElemKind::FloatTy, {winNum, specLen}, "spectrogram",
                            false /* isTrainable */);
  bindings.allocate(spectrogram)
      ->getHandle()
      .randomize(10.0, 100.0, mod.getPRNG());

  // Create MFCC node.
  float sampleRate = 16000.0;
  float lowerFrequency = 20.0;
  float upperFrequency = 4000.0;
  size_t filterBankCount = 40;
  size_t numCoefficients = 13;
  auto *mfcc = F->createMFCC("mfcc", spectrogram, sampleRate, lowerFrequency,
                             upperFrequency, filterBankCount, numCoefficients);
  auto *res = F->createSave("save", mfcc);
  auto *resultTensor = bindings.allocate(res->getPlaceholder());
  return std::make_pair(F, resultTensor);
}

#define TEST_MFCC(WNUM, SLEN, TOL)                                             \
  TEST_P(OperatorStatelessTest, MFCC_##WNUM##x##SLEN##_Float) {                \
    ENABLED_BACKENDS("Interpreter", "CPU");                                    \
    compareAgainstInterpreter(getBackendName(),                                \
                              createAndInitBasicMFCCTest<WNUM, SLEN>,          \
                              ElemKind::FloatTy, ElemKind::FloatTy, TOL);      \
  }

TEST_MFCC(1, 17, 5e-5)
TEST_MFCC(1, 33, 5e-5)
TEST_MFCC(1, 65, 2e-5)
TEST_MFCC(1, 129, 1e-5)
TEST_MFCC(2, 257, 1e-5)
TEST_MFCC(3, 513, 1e-5)
TEST_MFCC(3, 1025, 1e-5)

template <typename DataType>
static void testRoiAlign(
    PlaceholderBindings &bindings, Module &mod, Function &F,
    ExecutionEngine &EE, ElemKind ElemTy, llvm::ArrayRef<dim_t> featureMapDims,
    llvm::ArrayRef<DataType> featureMap, llvm::ArrayRef<dim_t> boxesDims,
    llvm::ArrayRef<DataType> boxes, llvm::ArrayRef<dim_t> batchIndicesDims,
    llvm::ArrayRef<int64_t> batchIndices, PoolingMode mode, dim_t outputHeight,
    dim_t outputWidth, uint32_t samplingRatio, float spatialScale, bool aligned,
    llvm::ArrayRef<DataType> expectedValues, float comparisonThreshold,
    bool rotated) {
  auto *featureMapT =
      mod.createPlaceholder(ElemTy, featureMapDims, "featureMap", false);
  bindings.allocate(featureMapT)->getHandle<DataType>() = featureMap;

  auto *boxesT = mod.createPlaceholder(ElemTy, boxesDims, "boxes", false);
  bindings.allocate(boxesT)->getHandle<DataType>() = boxes;

  auto *batchIndicesT = mod.createPlaceholder(
      ElemKind::Int64ITy, batchIndicesDims, "batchIndices", false);
  bindings.allocate(batchIndicesT)->getHandle<int64_t>() = batchIndices;

  auto *LN = F.createROIAlign("ROIAlign", featureMapT, boxesT, batchIndicesT,
                              outputHeight, outputWidth, samplingRatio,
                              spatialScale, aligned, rotated, mode);
  auto *save = F.createSave("save", LN);
  auto *savePlaceholder = save->getPlaceholder();
  bindings.allocate(savePlaceholder);

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  auto saveH = bindings.get(savePlaceholder)->getHandle<DataType>();

  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_NEAR(saveH.raw(i), expectedValues[i], comparisonThreshold);
  }
}

template <typename DataType>
static void roiAlignBasicTest(PlaceholderBindings &bindings, Module &mod,
                              Function &F, ExecutionEngine &EE, ElemKind ElemTy,
                              float comparisonThreshold) {
  llvm::SmallVector<dim_t, 4> featureMapDims = {2, 5, 5, 2};
  llvm::SmallVector<DataType, 100> featureMap = {
      1.,  0.,  1.,  1.,  1.,  2.,  1.,  3.,  1.,  4.,  1.,  5.,  1.,  6.,  1.,
      7.,  1.,  8.,  1.,  9.,  1.,  10., 1.,  11., 1.,  12., 1.,  13., 1.,  14.,
      1.,  15., 1.,  16., 1.,  17., 1.,  18., 1.,  19., 1.,  20., 1.,  21., 1.,
      22., 1.,  23., 1.,  24., 0.,  1.,  1.,  1.,  2.,  1.,  3.,  1.,  4.,  1.,
      5.,  1.,  6.,  1.,  7.,  1.,  8.,  1.,  9.,  1.,  10., 1.,  11., 1.,  12.,
      1.,  13., 1.,  14., 1.,  15., 1.,  16., 1.,  17., 1.,  18., 1.,  19., 1.,
      20., 1.,  21., 1.,  22., 1.,  23., 1.,  24., 1.};

  llvm::SmallVector<dim_t, 2> boxesDims = {2, 4};
  llvm::SmallVector<DataType, 8> boxes = {1., 1., 3., 3., 1., 1., 3., 3.};

  llvm::SmallVector<dim_t, 1> batchIndicesDims = {2};
  llvm::SmallVector<int64_t, 2> batchIndices = {1, 0};

  llvm::SmallVector<DataType, 12> expectedValues = {
      9, 1, 10, 1, 14, 1, 15, 1, 1, 9, 1, 10, 1, 14, 1, 15.};

  testRoiAlign<DataType>(
      bindings, mod, F, EE, ElemTy, featureMapDims, featureMap, boxesDims,
      boxes, batchIndicesDims, batchIndices, PoolingMode::AVG, 2, 2, 2, 1,
      false, expectedValues, comparisonThreshold, /*rotated*/ false);
}

TEST_P(OperatorTest, RoiAlign) {
  CHECK_IF_ENABLED();
  roiAlignBasicTest<float>(bindings_, mod_, *F_, EE_, ElemKind::FloatTy, 1E-4);
}

TEST_P(OperatorTest, FP16RoiAlign) {
  CHECK_IF_ENABLED();
  roiAlignBasicTest<float16_t>(bindings_, mod_, *F_, EE_, ElemKind::Float16Ty,
                               1E-3);
}

template <typename DataType>
static void
roiAlignWithAlignedCoordinatesTest(PlaceholderBindings &bindings, Module &mod,
                                   Function &F, ExecutionEngine &EE,
                                   ElemKind ElemTy, float comparisonThreshold) {
  llvm::SmallVector<dim_t, 4> featureMapDims = {1, 5, 5, 1};
  llvm::SmallVector<DataType, 25> featureMap = {
      0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3,
      0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5};

  llvm::SmallVector<dim_t, 2> boxesDims = {1, 4};
  llvm::SmallVector<DataType, 5> boxes = {0.0, 0.4, 4.3, 2.9};

  llvm::SmallVector<dim_t, 1> batchIndicesDims = {1};
  llvm::SmallVector<int64_t, 1> batchIndices = {0};

  llvm::SmallVector<DataType, 9> expectedValues = {
      0.1287, 0.2650, 0.4083, 0.1288, 0.2650, 0.4083, 0.1287, 0.2650, 0.4083};

  testRoiAlign<DataType>(
      bindings, mod, F, EE, ElemTy, featureMapDims, featureMap, boxesDims,
      boxes, batchIndicesDims, batchIndices, PoolingMode::AVG, 3, 3, 2, 1, true,
      expectedValues, comparisonThreshold, /*rotated*/ false);
}

TEST_P(OperatorTest, RoiAlignWithAlignedCoordinates) {
  CHECK_IF_ENABLED();
  roiAlignWithAlignedCoordinatesTest<float>(bindings_, mod_, *F_, EE_,
                                            ElemKind::FloatTy, 1E-4);
}

TEST_P(OperatorTest, FP16RoiAlignWithAlignedCoordinates) {
  CHECK_IF_ENABLED();
  roiAlignWithAlignedCoordinatesTest<float16_t>(bindings_, mod_, *F_, EE_,
                                                ElemKind::Float16Ty, 1E-3);
}

template <typename DataType>
static void roiAlignBatchIndexInBoxesTensorTest(PlaceholderBindings &bindings,
                                                Module &mod, Function &F,
                                                ExecutionEngine &EE,
                                                ElemKind ElemTy,
                                                float comparisonThreshold) {
  llvm::SmallVector<dim_t, 4> featureMapDims = {1, 5, 5, 1};
  llvm::SmallVector<DataType, 25> featureMap = {
      -1.2428743,  -0.9784467,  0.33036363,  0.47368783,  -0.81611377,
      -1.1874917,  -1.6208626,  -0.04190686, -0.5767553,  1.1949452,
      -2.1838918,  1.0099407,   0.6925469,   0.37020323,  -0.3799704,
      -0.10355259, -0.64257944, -1.3108171,  -1.5346326,  -1.4158413,
      0.65036285,  -0.59222955, -1.560379,   -0.33371264, 0.37395215,
  };

  llvm::SmallVector<dim_t, 2> boxesDims = {2, 5};
  llvm::SmallVector<DataType, 5> boxes = {
      0., 1.1889961, 0.53260314, 3.1794803, 3.5056353,
      0., 1.4748696, 2.4069107,  4.1870456, 4.6166725};

  llvm::SmallVector<dim_t, 1> batchIndicesDims = {1};
  llvm::SmallVector<int64_t, 1> batchIndices = {1};

  llvm::SmallVector<DataType, 18> expectedValues = {
      -1.1747, -0.3246, 0.0591,  -0.3049, 0.1516,  0.1917,
      0.0270,  -0.1727, -0.4240, 0.3784,  0.0435,  -0.2741,
      -0.7801, -1.1925, -1.2289, -0.9860, -1.2124, -0.5044};

  testRoiAlign<DataType>(
      bindings, mod, F, EE, ElemTy, featureMapDims, featureMap, boxesDims,
      boxes, batchIndicesDims, batchIndices, PoolingMode::AVG, 3, 3, 2, 1, true,
      expectedValues, comparisonThreshold, /*rotated*/ false);
}

TEST_P(OperatorTest, RoiAlignBatchIndexInBoxesTensor) {
  CHECK_IF_ENABLED();
  roiAlignBatchIndexInBoxesTensorTest<float>(bindings_, mod_, *F_, EE_,
                                             ElemKind::FloatTy, 1E-4);
}

TEST_P(OperatorTest, FP16RoiAlignBatchIndexInBoxesTensor) {
  CHECK_IF_ENABLED();

  // 1E-2 threshold is required because fp16 occasionally causes sampling
  // points to be shifted due to rounding which results in large maximum
  // difference from reference.
  roiAlignBatchIndexInBoxesTensorTest<float16_t>(bindings_, mod_, *F_, EE_,
                                                 ElemKind::Float16Ty, 1E-2);
}

template <typename DataType>
static void roiAlignRotatedBatchIndexInBoxesTensorTest(
    PlaceholderBindings &bindings, Module &mod, Function &F,
    ExecutionEngine &EE, ElemKind ElemTy, float comparisonThreshold) {
  llvm::SmallVector<dim_t, 4> featureMapDims = {1, 5, 5, 3};
  llvm::SmallVector<DataType, 25> featureMap = {
      -8.6497840881,  -5.0528664589, -5.1990814209,  -10.8463373184,
      -14.9225864410, 4.0806860924,  14.7214040756,  -11.9505138397,
      16.7156505585,  -9.7665214539, -13.4883165359, 1.3252578974,
      -1.6687428951,  10.5697870255, -4.4617910385,  16.9429378510,
      9.5267467499,   5.9925584793,  5.6118640900,   1.5372716188,
      2.4355530739,   -3.0808238983, 2.6959202290,   -9.9537315369,
      -1.1652010679,  15.3153333664, 11.4361877441,  8.7219638824,
      6.0323386192,   -3.3185434341, -5.8790159225,  -7.0839004517,
      11.3739776611,  -7.1884007454, 10.0514144897,  -7.9980802536,
      15.8880462646,  -2.3542327881, -9.3197269440,  -4.7869114876,
      15.6589784622,  -1.5917046070, -1.2245910168,  0.0595506988,
      3.6575553417,   14.7897586823, 11.4384317398,  -5.1155147552,
      0.7425209880,   1.1070071459,  4.2300715446,   -17.3323173523,
      -2.9571244717,  -3.6389255524, -8.8692741394,  19.7417812347,
      7.1416730881,   25.0613708496, 3.8868305683,   -1.4834585190,
      0.3542223871,   14.2146720886, -7.8964066505,  7.7495927811,
      3.6963310242,   9.0857019424,  -3.4129979610,  -3.1457190514,
      -15.2861795425, 10.1850719452, -0.2935675085,  9.8417263031,
      1.1156638861,   -8.5692892075, -1.8766889572};

  llvm::SmallVector<dim_t, 2> boxesDims = {4, 6};
  llvm::SmallVector<DataType, 5> boxes = {
      0.0000000000e+00, 3.7350432873e+00, 1.8349769115e+00,
      2.2127370536e-01, 1.7214350700e+00, 6.7396400452e+01,

      0.0000000000e+00, 2.5810198784e+00, 2.7632935047e+00,
      4.5813250542e-01, 1.0615788698e+00, 5.9284824371e+01,

      0.0000000000e+00, 1.4992059469e+00, 3.3264288902e+00,
      5.8828938752e-02, 1.2860099971e-01, 1.7042655945e+02,

      0.0000000000e+00, 1.6475434303e+00, 1.1158514023e+00,
      6.0969877243e-01, 1.6949450970e+00, 5.7489040375e+01};

  // Unused
  llvm::SmallVector<dim_t, 1> batchIndicesDims = {4};
  llvm::SmallVector<int64_t, 1> batchIndices = {42, 42, 42, 42};

  llvm::SmallVector<DataType, 18> expectedValues = {
      -1.2753072977e+00, 1.1022174835e+01,  2.8559112549e+00,
      -1.5445901155e+00, 1.1492666245e+01,  4.0045604706e+00,
      -1.6816796064e+00, 1.1780773163e+01,  5.1841292381e+00,
      -1.1537375450e+00, 1.2963508606e+01,  4.9455566406e+00,
      -5.9787964821e-01, 1.2705860138e+01,  5.3227939606e+00,
      -1.7603963614e-02, 1.2472600937e+01,  5.6228017807e+00,
      9.0734308958e-01,  5.7471928596e+00,  1.7764383554e+00,
      1.6517986059e+00,  5.6778922081e+00,  1.5571854115e+00,
      2.4206719398e+00,  5.6329779625e+00,  1.2607033253e+00,
      4.3689918518e+00,  7.3948031664e-01,  -7.3034667969e+00,
      8.6381378174e+00,  2.3455758393e-01,  -8.4534435272e+00,
      1.1591947556e+01,  -4.2240649462e-01, -9.0010957718e+00,
      2.1553003788e+00,  -1.4560343027e+00, -6.5866470337e+00,
      6.0744242668e+00,  -8.3328241110e-01, -7.0825934410e+00,
      8.9081802368e+00,  5.4210889339e-01,  -7.2683048248e+00,
      -4.3445730209e+00, 3.6746215820e+00,  -3.1289699078e+00,
      -1.7619293928e+00, 5.1320915222e+00,  -3.2894101143e+00,
      2.2393733263e-01,  6.4935913086e+00,  -3.5123698711e+00,
      -4.1849538684e-01, 2.1935482025e+00,  2.5842363834e+00,
      -2.0057903230e-01, 2.3351111412e+00,  2.5799021721e+00,
      1.1708274484e-02,  2.4918785095e+00,  2.4347753525e+00,
      -8.1277823448e-01, 2.5285022259e+00,  2.0223598480e+00,
      -6.2242215872e-01, 2.6641519070e+00,  2.0815916061e+00,
      -4.2229780555e-01, 2.7992806435e+00,  1.9814052582e+00,
      -1.1822580099e+00, 2.8584275246e+00,  1.4644309282e+00,
      -1.0267944336e+00, 3.0002222061e+00,  1.5492329597e+00,
      -8.4862631559e-01, 3.1232376099e+00,  1.5107266903e+00,
      -6.1683624983e-02, -3.0222876072e+00, 1.8380764723e+00,
      -4.1196775436e+00, -6.7160081863e+00, 1.7320134640e+00,
      -7.8356714249e+00, -1.0480127335e+01, 2.1065652370e+00,
      2.1715850830e+00,  -1.5094176531e+00, 2.0960900784e+00,
      -3.2094952464e-01, -4.5263018608e+00, 2.4609162807e+00,
      -1.1458464861e+00, -7.0648045540e+00, 3.5408535004e+00,
      1.7010095119e+00,  2.0761563778e+00,  -4.2401647568e+00,
      4.8630356789e-01,  8.5567343235e-01,  -3.9398088455e+00,
      1.1503255367e+00,  -1.4384213686e+00, -1.8096057177e+00};

  testRoiAlign<DataType>(bindings, mod, F, EE, ElemTy, featureMapDims,
                         featureMap, boxesDims, boxes, batchIndicesDims,
                         batchIndices, PoolingMode::AVG, 3, 3, 2, 1, true,
                         expectedValues, comparisonThreshold, /*rotated*/ true);
}

TEST_P(OperatorTest, RoiAlignRotatedBatchIndexInBoxesTensor) {
  CHECK_IF_ENABLED();
  roiAlignRotatedBatchIndexInBoxesTensorTest<float>(bindings_, mod_, *F_, EE_,
                                                    ElemKind::FloatTy, 1E-4);
}

TEST_P(OperatorTest, FP16RoiAlignRotatedBatchIndexInBoxesTensor) {
  CHECK_IF_ENABLED();

  // 1E-1 threshold is required because fp16 occasionally causes sampling
  // points to be shifted due to rounding which results in large maximum
  // difference from reference.
  roiAlignRotatedBatchIndexInBoxesTensorTest<float16_t>(
      bindings_, mod_, *F_, EE_, ElemKind::Float16Ty, 1E-1);
}

TEST_P(OperatorTest, BBoxTransform) {
  CHECK_IF_ENABLED();

  auto *rois = mod_.createPlaceholder(ElemKind::FloatTy, {5, 5}, "rois", false);
  bindings_.allocate(rois)->getHandle<float>() = {
      0., 22.113754, 10.269318, 77.57481,   117.23254,
      0., 89.73806,  46.060974, 125.824005, 96.2649,
      1., 11.121593, 78.21209,  75.711426,  254.73167,
      3., 0.9983631, 352.86606, 248.86679,  367.66916,
      3., 221.1072,  136.93027, 413.82764,  211.13977};

  auto *deltas =
      mod_.createPlaceholder(ElemKind::FloatTy, {5, 8}, "deltas", false);
  bindings_.allocate(deltas)->getHandle<float>() = {
      -0.30892685, -0.44120562, 1.7046866,   -0.62745374, 1.1726723,
      -0.52569604, -0.14308402, 0.48242334,  -1.3132329,  -1.5958056,
      -0.81750935, 2.2151427,   -0.73521894, -0.00737088, 2.3750482,
      -1.5794574,  -0.48789233, 1.7873235,   0.6119284,   -0.7701755,
      -0.41762614, -0.9074146,  -0.7296619,  -0.30050594, 0.58725464,
      0.71989095,  -0.8755994,  -1.2122285,  -0.5378105,  -0.90247065,
      1.3996177,   -1.3575566,  0.6860114,   -0.4028068,  0.15296046,
      -0.22815527, -2.4161322,  -1.8008438,  -0.92949533, 0.19269551};

  auto *imInfo =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 3}, "imInfo", false);
  bindings_.allocate(imInfo)->getHandle<float>() = {
      159., 159., 1., 328., 328., 1., 466., 466., 1., 414., 414., 1.};

  std::vector<float> weights = {1.0, 1.0, 1.0, 1.0};
  auto *BBTN =
      F_->createBBoxTransform("bboxTransform", rois, deltas, imInfo, weights,
                              false, false, false, 0, 0, 0, true);
  auto *save = F_->createSave("save", BBTN->getBoxOut());
  auto *savePlaceholder = save->getPlaceholder();
  bindings_.allocate(savePlaceholder);

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = bindings_.get(savePlaceholder)->getHandle();

  std::vector<float> expectedValues = {
      0.0000,   0.0000,   158.0000, 44.4404,  92.0877,  0.0000,   140.0215,
      93.9451,  51.3913,  0.0000,   66.7658,  158.0000, 0.0000,   66.0093,
      158.0000, 75.5617,  0.0000,   327.0000, 71.3890,  327.0000, 0.7150,
      0.0000,   31.3340,  70.6096,  219.7409, 369.7931, 322.4225, 373.4951,
      0.0000,   344.4728, 413.0000, 347.5388, 337.9926, 114.3067, 413.0000,
      173.1735, 0.0000,   0.0000,   0.0000,   83.6907};
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_NEAR(saveH.raw(i), expectedValues[i], 1E-4);
  }
}

// Helper to test SpaceToDepth using \p DTy.
template <typename DataType>
static void testSpaceToDepthBlock3(glow::PlaceholderBindings &bindings,
                                   glow::Module &mod, glow::Function *F,
                                   glow::ExecutionEngine &EE, ElemKind DTy) {
  unsigned blockSize = 3;
  auto *in = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 6, 6},
                                                     "in", false, "NHWC");
  auto *tri = F->createTranspose("sptdTransposeIn", in, {0, 2, 3, 1}, "NHWC");
  auto *stdn = F->createSpaceToDepth("spacetodepth", tri, blockSize);
  auto *tro =
      F->createTranspose("sptdTransposeOut", stdn, {0, 3, 1, 2}, "NCHW");
  auto *save = F->createSave("save", tro);
  auto *result = bindings.allocate(save->getPlaceholder());

  /*
    Example for first batch.
  FROM:
  C0:             C1:
  [0  1  2  3  16 17]    [ 0  -1  -2  -3  -16 -17]
  [4  5  6  7  18 19]    [-4  -5  -6  -7  -18 -19]
  [8  9  10 11 20 21]    [-8  -9  -10 -11 -20 -21]
  [12 13 14 15 22 23]    [-12 -13 -14 -15 -22 -23]
  [24 25 26 27 28 29]    [-24 -25 -26 -27 -28 -29]
  [30 31 32 33 34 35]    [-30 -31 -32 -33 -34 -35]

  TO:
  C = 0
  [0,3]
  [12,15]

  C = 1
  [0,-3]
  [-12,-15]

  C = 2
  [1,16]
  [13,22]

  C = 3
  [-1,-16]
  [-13,-22]

  C = 4
  [2,17]
  [14,23]

  C = 5
  [-2,-17]
  [-14,-23]

  C = 6
  [4,7]
  [24,27]

  C = 7
  [-4,-7]
  [-24,-27]

  C = 8
  [5,18]
  [25,28]

  C = 9
  [-5,-18]
  [-25,-28]

  C = 10
  [6,19]
  [26,29]

  C = 11
  [-6,-19]
  [-26,-29]

  C = 12
  [8,11]
  [30,33]

  C = 13
  [-8,-11]
  [-30,-33]

  C = 14
  [9,20]
  [31,34]

  C = 15
  [-9,-20]
  [-31,-34]

  C = 16
  [10,21]
  [32,35]

  C = 17
  [-10,-21]
  [-32,-35]
  */

  bindings.allocate(in)->getHandle<DataType>() = {
      0,   1,   2,   3,   16,  17,  4,   5,   6,   7,   18,  19,  8,   9,   10,
      11,  20,  21,  12,  13,  14,  15,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  0,   -1,  -2,  -3,  -16, -17, -4,  -5,  -6,
      -7,  -18, -19, -8,  -9,  -10, -11, -20, -21, -12, -13, -14, -15, -22, -23,
      -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34, -35};

  std::vector<DataType> refResult = {
      0,   3,   12,  15,  0,  -3, -12, -15, 1,   16,  13,  22, -1, -16, -13,
      -22, 2,   17,  14,  23, -2, -17, -14, -23, 4,   7,   24, 27, -4,  -7,
      -24, -27, 5,   18,  25, 28, -5,  -18, -25, -28, 6,   19, 26, 29,  -6,
      -19, -26, -29, 8,   11, 30, 33,  -8,  -11, -30, -33, 9,  20, 31,  34,
      -9,  -20, -31, -34, 10, 21, 32,  35,  -10, -21, -32, -35};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Handle<DataType> resultH = result->getHandle<DataType>();

  auto iDims = in->dims();
  auto oDims = resultH.dims();
  EXPECT_EQ(iDims[0], oDims[0]);
  EXPECT_EQ(iDims[1] * blockSize * blockSize, oDims[1]);
  EXPECT_EQ(iDims[2], oDims[2] * blockSize);
  EXPECT_EQ(iDims[3], oDims[3] * blockSize);

  // NCHW format
  dim_t resIndex = 0;
  for (dim_t on = 0; on < oDims[0]; ++on) {
    for (dim_t oc = 0; oc < oDims[1]; ++oc) {
      for (dim_t oh = 0; oh < oDims[2]; ++oh) {
        for (dim_t ow = 0; ow < oDims[3]; ++ow) {
          DataType resultVal = resultH.at({on, oc, oh, ow});
          DataType refVal = refResult[resIndex++];
          EXPECT_EQ(resultVal, refVal);
        }
      }
    }
  }
}

/// Verify that the SpaceToDepth operator works correctly for int8. Block
/// Size 3.
TEST_P(OperatorTest, spaceToDepth_block3_int8) {
  CHECK_IF_ENABLED();
  testSpaceToDepthBlock3<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the SpaceToDepth operator works correctly for Float. Block
/// Size 3.
TEST_P(OperatorTest, spaceToDepth_block3_Float) {
  CHECK_IF_ENABLED();
  testSpaceToDepthBlock3<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

// Helper to test SpaceToDepth using \p DTy.
template <typename DataType>
static void testSpaceToDepth(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  unsigned blockSize = 2;
  auto *in = createPlaceholderConditionallyQuantized(mod, DTy, {2, 2, 4, 4},
                                                     "in", false, "NHWC");
  auto *tri = F->createTranspose("sptdTransposeIn", in, {0, 2, 3, 1}, "NHWC");
  auto *stdn = F->createSpaceToDepth("spacetodepth", tri, blockSize);
  auto *tro =
      F->createTranspose("sptdTransposeOut", stdn, {0, 3, 1, 2}, "NCHW");
  auto *save = F->createSave("save", tro);
  auto *result = bindings.allocate(save->getPlaceholder());

  /*
    Example for first batch.
  FROM:
  C0:             C1:
  [0  1  2  3]    [ 0  -1  -2  -3]
  [4  5  6  7]    [-4  -5  -6  -7]
  [8  9  10 11]   [-8  -9  -10 -11]
  [12 13 14 15]   [-12 -13 -14 -15]

  TO:
  C0:
  [0,  2]
  [8,  10]

  C1:
  [ 0,  -2]
  [-8, -10]

  C2:
  [1, 3]
  [9, 11]

  C3:
  [-1, -3]
  [-9, -11]

  C4:
  [4,  6]
  [12, 14]

  C5:
  [-4,  -6]
  [-12, -14]

  C6:
  [5, 7]
  [13, 15]

  C7:
  [-5,  -7]
  [-13, -15]
  */

  bindings.allocate(in)->getHandle<DataType>() = {
      0, 1,   2,   3,   4,  5,  6,   7,   8,  9,  10,  11,  12,  13,  14,  15,
      0, -1,  -2,  -3,  -4, -5, -6,  -7,  -8, -9, -10, -11, -12, -13, -14, -15,
      0, 7,   9,   23,  24, 25, 26,  27,  8,  9,  10,  33,  12,  13,  14,  15,
      0, -21, -22, -23, -4, -5, -26, -27, -8, -9, -10, -11, -12, -13, -14, -15};

  std::vector<DataType> refResult = {
      0,  2,  8,  10, 0,  -2,  -8,  -10, 1,  3,  9,  11, -1,  -3,  -9,  -11,
      4,  6,  12, 14, -4, -6,  -12, -14, 5,  7,  13, 15, -5,  -7,  -13, -15,
      0,  9,  8,  10, 0,  -22, -8,  -10, 7,  23, 9,  33, -21, -23, -9,  -11,
      24, 26, 12, 14, -4, -26, -12, -14, 25, 27, 13, 15, -5,  -27, -13, -15};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Handle<DataType> resultH = result->getHandle<DataType>();

  auto iDims = in->dims();
  auto oDims = resultH.dims();
  EXPECT_EQ(iDims[0], oDims[0]);
  EXPECT_EQ(iDims[1] * blockSize * blockSize, oDims[1]);
  EXPECT_EQ(iDims[2], oDims[2] * blockSize);
  EXPECT_EQ(iDims[3], oDims[3] * blockSize);

  // NCHW format
  dim_t resIndex = 0;
  for (dim_t on = 0; on < oDims[0]; ++on) {
    for (dim_t oc = 0; oc < oDims[1]; ++oc) {
      for (dim_t oh = 0; oh < oDims[2]; ++oh) {
        for (dim_t ow = 0; ow < oDims[3]; ++ow) {
          DataType resultVal = resultH.at({on, oc, oh, ow});
          DataType refVal = refResult[resIndex++];
          EXPECT_EQ(resultVal, refVal);
        }
      }
    }
  }
}

/// Verify that the SpaceToDepth operator works correctly for int8. Block
/// Size 2.
TEST_P(OperatorTest, spaceToDepth_block2_int8) {
  CHECK_IF_ENABLED();
  testSpaceToDepth<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the SpaceToDepth operator works correctly for Float. Block
/// Size 2.
TEST_P(OperatorTest, spaceToDepth_block2_Float) {
  CHECK_IF_ENABLED();
  testSpaceToDepth<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Helper to test ResizeNearest using \p DTy.
template <typename DataType>
static void testResizeNearest(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy,
                              bool v11 = false) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 2, 1},
                                                        "input", false, "NHWC");
  bindings.allocate(input)->getHandle<DataType>() = {2, 4, 8, 16};

  ResizeNearestNode *resizeUp = nullptr;
  ResizeNearestNode *resizeDown = nullptr;

  std::vector<float> scaleUp = {1, 2.0f, 1.5f, 1};

  if (v11) {
    dim_t newH = std::floor(2 * 2.0f);
    dim_t newW = std::floor(2 * 1.5f);
    auto outTy =
        mod.uniqueTypeWithNewShape(input->getType(), {1, newH, newW, 1});
    resizeUp = F->createResizeNearest("resizeUp", input, outTy);
  } else {
    resizeUp = F->createResizeNearest("resizeUp", input, scaleUp);
  }
  auto *saveUp = F->createSave("saveUp", resizeUp);
  auto *resultUp = bindings.allocate(saveUp->getPlaceholder());

  std::vector<float> scaleDown = {1, 0.9f, 0.6f, 1};

  if (v11) {
    dim_t newH = std::floor(2 * 0.9f);
    dim_t newW = std::floor(2 * 0.6f);
    auto outTy =
        mod.uniqueTypeWithNewShape(input->getType(), {1, newH, newW, 1});
    resizeDown = F->createResizeNearest("resizeDown", input, outTy);
  } else {
    resizeDown = F->createResizeNearest("resizeDown", input, scaleDown);
  }

  auto *saveDown = F->createSave("saveDown", resizeDown);
  auto *resultDown = bindings.allocate(saveDown->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(
      F, bindings,
      {input, saveUp->getPlaceholder(), saveDown->getPlaceholder()});

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultUpH = resultUp->getHandle<DataType>();
  std::vector<dim_t> expectedDimsUp = {1, 4, 3, 1};
  ASSERT_TRUE(resultUpH.dims().vec() == expectedDimsUp);

  EXPECT_EQ(resultUpH.at({0, 0, 0, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 0, 1, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 0, 2, 0}), static_cast<DataType>(4));

  EXPECT_EQ(resultUpH.at({0, 1, 0, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 1, 1, 0}), static_cast<DataType>(2));
  EXPECT_EQ(resultUpH.at({0, 1, 2, 0}), static_cast<DataType>(4));

  EXPECT_EQ(resultUpH.at({0, 2, 0, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 2, 1, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 2, 2, 0}), static_cast<DataType>(16));

  EXPECT_EQ(resultUpH.at({0, 3, 0, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 3, 1, 0}), static_cast<DataType>(8));
  EXPECT_EQ(resultUpH.at({0, 3, 2, 0}), static_cast<DataType>(16));

  auto resultDownH = resultDown->getHandle<DataType>();
  std::vector<dim_t> expectedDimsDown = {1, 1, 1, 1};
  ASSERT_TRUE(resultDownH.dims().vec() == expectedDimsDown);
  EXPECT_EQ(resultDownH.at({0, 0, 0, 0}), static_cast<DataType>(2));
}

/// Verify that the ResizeNearest operator works correctly for Float.
TEST_P(OperatorTest, ResizeNearest_Float) {
  CHECK_IF_ENABLED();
  testResizeNearest<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the ResizeNearest operator works correctly for Float16.
TEST_P(OperatorTest, ResizeNearest_Float16) {
  CHECK_IF_ENABLED();
  testResizeNearest<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Verify that the ResizeNearest operator works correctly for BFloat16.
TEST_P(OperatorTest, ResizeNearest_BFloat16) {
  CHECK_IF_ENABLED();
  testResizeNearest<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Verify that the ResizeNearest operator works correctly for Int8Q.
TEST_P(OperatorTest, ResizeNearest_Int8) {
  CHECK_IF_ENABLED();
  testResizeNearest<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the ResizeNearest operator works correctly for Int16Q.
TEST_P(OperatorTest, ResizeNearest_Int16) {
  CHECK_IF_ENABLED();
  testResizeNearest<int16_t>(bindings_, mod_, F_, EE_, ElemKind::Int16QTy);
}

/// Verify that the ResizeNearest operator works correctly for Int32Q.
TEST_P(OperatorTest, ResizeNearest_Int32) {
  CHECK_IF_ENABLED();
  testResizeNearest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, ResizeNearest_Float_outTy) {
  CHECK_IF_ENABLED();
  testResizeNearest<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, true);
}

TEST_P(OperatorTest, ResizeNearest_Float16_outTy) {
  CHECK_IF_ENABLED();
  testResizeNearest<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                               true);
}

TEST_P(OperatorTest, ResizeNearest_BFloat16_outTy) {
  CHECK_IF_ENABLED();
  testResizeNearest<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                                true);
}

TEST_P(OperatorTest, ResizeNearest_Int8_outTy) {
  CHECK_IF_ENABLED();
  testResizeNearest<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, true);
}
TEST_P(OperatorTest, ResizeNearest_Int16_outTy) {
  CHECK_IF_ENABLED();
  testResizeNearest<int16_t>(bindings_, mod_, F_, EE_, ElemKind::Int16QTy,
                             true);
}
TEST_P(OperatorTest, ResizeNearest_Int32_outTy) {
  CHECK_IF_ENABLED();
  testResizeNearest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy,
                             true);
}

/// Helper to test ResizeNearest using \p DTy.
template <typename DataType>
static void testResizeBilinear(glow::PlaceholderBindings &bindings,
                               glow::Module &mod, glow::Function *F,
                               glow::ExecutionEngine &EE, ElemKind DTy,
                               bool v11 = false) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 2, 1},
                                                        "input", false, "NHWC");
  bindings.allocate(input)->getHandle<DataType>() = {2, 4, 8, 16};

  std::vector<float> scaleUp = {1, 2.0f, 1.5f, 1};

  ResizeBilinearNode *resizeUp = nullptr;
  ResizeBilinearNode *resizeDown = nullptr;

  if (v11) {
    dim_t newH = std::floor(2 * 2.0f);
    dim_t newW = std::floor(2 * 1.5f);
    auto outTy =
        mod.uniqueTypeWithNewShape(input->getType(), {1, newH, newW, 1});
    resizeUp = F->createResizeBilinear("resizeUp", input, outTy);
  } else {
    resizeUp = F->createResizeBilinear("resizeUp", input, scaleUp);
  }

  auto *saveUp = F->createSave("saveUp", resizeUp);
  auto *resultUp = bindings.allocate(saveUp->getPlaceholder());

  std::vector<float> scaleDown = {1, 0.9f, 0.6f, 1};

  if (v11) {
    dim_t newH = std::floor(2 * 0.9f);
    dim_t newW = std::floor(2 * 0.6f);
    auto outTy =
        mod.uniqueTypeWithNewShape(input->getType(), {1, newH, newW, 1});
    resizeDown = F->createResizeBilinear("resizeDown", input, outTy);
  } else {
    resizeDown = F->createResizeBilinear("resizeDown", input, scaleDown);
  }

  auto *saveDown = F->createSave("saveDown", resizeDown);
  auto *resultDown = bindings.allocate(saveDown->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(
      F, bindings,
      {input, saveUp->getPlaceholder(), saveDown->getPlaceholder()});

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultUpH = resultUp->getHandle<DataType>();
  std::vector<dim_t> expectedDimsUp = {1, 4, 3, 1};
  ASSERT_TRUE(resultUpH.dims().vec() == expectedDimsUp);

// use EXPECT_NEAR for float otherwise EXPECT_EQ. Optional third arg is
// allowed error for EXPECT_NEAR. If not specified uses default.
#define EXPECT_EQF(a, b, ...)                                                  \
  if ((std::is_same<DataType, float>::value) ||                                \
      (std::is_same<DataType, float16_t>::value) ||                            \
      (std::is_same<DataType, bfloat16_t>::value)) {                           \
    EXPECT_FLOAT_EQ(a, b);                                                     \
  } else {                                                                     \
    EXPECT_EQ(a, b);                                                           \
  }

  EXPECT_EQF(resultUpH.at({0, 0, 0, 0}), static_cast<DataType>(2));
  EXPECT_EQF(resultUpH.at({0, 0, 1, 0}), static_cast<DataType>(3.333333));
  EXPECT_EQF(resultUpH.at({0, 0, 2, 0}), static_cast<DataType>(4));

  EXPECT_EQF(resultUpH.at({0, 1, 0, 0}), static_cast<DataType>(5));
  EXPECT_EQF(resultUpH.at({0, 1, 1, 0}), static_cast<DataType>(8.333333));
  EXPECT_EQF(resultUpH.at({0, 1, 2, 0}), static_cast<DataType>(10));

  EXPECT_EQF(resultUpH.at({0, 2, 0, 0}), static_cast<DataType>(8));
  EXPECT_EQF(resultUpH.at({0, 2, 1, 0}), static_cast<DataType>(13.33333));
  EXPECT_EQF(resultUpH.at({0, 2, 2, 0}), static_cast<DataType>(16));

  EXPECT_EQF(resultUpH.at({0, 3, 0, 0}), static_cast<DataType>(8));
  EXPECT_EQF(resultUpH.at({0, 3, 1, 0}), static_cast<DataType>(13.33333));
  EXPECT_EQF(resultUpH.at({0, 3, 2, 0}), static_cast<DataType>(16));

  auto resultDownH = resultDown->getHandle<DataType>();
  std::vector<dim_t> expectedDimsDown = {1, 1, 1, 1};
  ASSERT_TRUE(resultDownH.dims().vec() == expectedDimsDown);
  EXPECT_EQF(resultDownH.at({0, 0, 0, 0}), static_cast<DataType>(2));
}

/// Verify that the ResizeNearest operator works correctly for Float.
TEST_P(OperatorTest, ResizeBilinear_Float) {
  CHECK_IF_ENABLED();
  testResizeBilinear<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the ResizeNearest operator works correctly for Float16.
TEST_P(OperatorTest, ResizeBilinear_Float16) {
  CHECK_IF_ENABLED();
  testResizeBilinear<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Verify that the ResizeNearest operator works correctly for BFloat16.
TEST_P(OperatorTest, ResizeBilinear_BFloat16) {
  CHECK_IF_ENABLED();
  testResizeBilinear<bfloat16_t>(bindings_, mod_, F_, EE_,
                                 ElemKind::BFloat16Ty);
}

/// Verify that the ResizeNearest operator works correctly for Int8Q.
TEST_P(OperatorTest, ResizeBilinear_Int8) {
  CHECK_IF_ENABLED();
  testResizeBilinear<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Verify that the ResizeNearest operator works correctly for Int16Q.
TEST_P(OperatorTest, ResizeBilinear_Int16) {
  CHECK_IF_ENABLED();
  testResizeBilinear<int16_t>(bindings_, mod_, F_, EE_, ElemKind::Int16QTy);
}

/// Verify that the ResizeNearest operator works correctly for Int32Q.
TEST_P(OperatorTest, ResizeBilinear_Int32) {
  CHECK_IF_ENABLED();
  testResizeBilinear<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, ResizeBilinear_Float_outTy) {
  CHECK_IF_ENABLED();
  testResizeBilinear<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, true);
}
TEST_P(OperatorTest, ResizeBilinear_Float16_outTy) {
  CHECK_IF_ENABLED();
  testResizeBilinear<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                                true);
}
TEST_P(OperatorTest, ResizeBilinear_BFloat16_outTy) {
  CHECK_IF_ENABLED();
  testResizeBilinear<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                                 true);
}
TEST_P(OperatorTest, ResizeBilinear_Int8_outTy) {
  CHECK_IF_ENABLED();
  testResizeBilinear<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy, true);
}
TEST_P(OperatorTest, ResizeBilinear_Int16_outTy) {
  CHECK_IF_ENABLED();
  testResizeBilinear<int16_t>(bindings_, mod_, F_, EE_, ElemKind::Int16QTy,
                              true);
}
TEST_P(OperatorTest, ResizeBilinear_Int32_outTy) {
  CHECK_IF_ENABLED();
  testResizeBilinear<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy,
                              true);
}

TEST_P(OperatorTest, pow) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 3}, "X", false);
  auto *Y = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "Y", false);
  auto *Exp = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "Exp", false);

  bindings_.allocate(X)->getHandle() = {5, 0.1f, -3};
  bindings_.allocate(Y)->getHandle() = {2, 100};
  bindings_.allocate(Exp)->getHandle() = {2, -1};

  auto *Pow1 = F_->createPow("Pow1", X, 2.0);
  auto *Pow2 = F_->createPow("Pow2", Y, 0.5);
  auto *Pow3 = F_->createPow("Pow3", Y, Exp);

  auto *save1 = F_->createSave("save", Pow1);
  auto *savePlaceholder1 = save1->getPlaceholder();

  auto *save2 = F_->createSave("save", Pow2);
  auto *savePlaceholder2 = save2->getPlaceholder();

  auto *save3 = F_->createSave("save", Pow3);
  auto *savePlaceholder3 = save3->getPlaceholder();

  bindings_.allocate(savePlaceholder1);
  bindings_.allocate(savePlaceholder2);
  bindings_.allocate(savePlaceholder3);

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto H_X = bindings_.get(savePlaceholder1)->getHandle();
  EXPECT_NEAR(H_X.at({0, 0, 0}), 25, 1E-5);
  EXPECT_NEAR(H_X.at({0, 0, 1}), 0.01, 1E-5);
  EXPECT_NEAR(H_X.at({0, 0, 2}), 9, 1E-5);

  auto H_Y = bindings_.get(savePlaceholder2)->getHandle();
  EXPECT_NEAR(H_Y.at({0}), sqrt(2.0), 1E-5);
  EXPECT_NEAR(H_Y.at({1}), 10, 1E-5);

  auto H_Z = bindings_.get(savePlaceholder3)->getHandle();
  EXPECT_NEAR(H_Z.at({0}), 4, 1E-5);
  EXPECT_NEAR(H_Z.at({1}), 0.01, 1E-5);
}

/// Helper to test ReplaceNaN using \p DTy.
template <typename DataType>
static void testReplaceNaN(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto value = 1.0f;
  auto *X = mod.createPlaceholder(DTy, {6}, "X", false);
  auto XH = bindings.allocate(X)->getHandle<DataType>();
  XH = {1, NAN, 2, NAN, 3, NAN};

  auto *RNN = F->createReplaceNaN("replaceNaN", X, value);

  auto *save = F->createSave("save", RNN);
  auto *saveTensor = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  auto saveH = saveTensor->getHandle<DataType>();

  for (size_t i = 0; i < 6; i++) {
    if (std::isnan((float)XH.raw(i))) {
      EXPECT_EQ(saveH.raw(i), (DataType)value);
    } else {
      EXPECT_EQ(XH.raw(i), saveH.raw(i));
    }
  }
}

/// Test that ReplaceNaN is correctly supported in FloatTy.
TEST_P(OperatorTest, replaceNaN_Float) {
  CHECK_IF_ENABLED();
  testReplaceNaN<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that ReplaceNaN is correctly supported in Float16Ty.
TEST_P(OperatorTest, replaceNaN_Float16) {
  CHECK_IF_ENABLED();
  testReplaceNaN<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test that ReplaceNaN is correctly supported in BFloat16Ty.
TEST_P(OperatorTest, replaceNaN_BFloat16) {
  CHECK_IF_ENABLED();
  testReplaceNaN<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

TEST_P(OperatorTest, log) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {210030, 600, 4, 0.7f, .005f, 0.000829f};

  auto *LN = F_->createLog("log", X);

  auto *save = F_->createSave("save", LN);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), log(XH.at({i})), 1E-5);
  }
}

/// Range of asin domain is [-1,1] and the range of output
/// is [-pi/2, pi/2]
TEST_P(OperatorTest, Asin_FloatTy) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {-0.34, 0.32, 0.0001, 1.0, -0.4, 0.78};

  auto *AS = F_->createAsin("Asin", X);

  auto *save = F_->createSave("save", AS);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), asin(XH.at({i})), 1E-5);
  }
}

/// Range of acos domain is [-1,1] and the range of output
/// is [0, pi]
TEST_P(OperatorTest, Acos_FloatTy) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {-0.34, 0.32, 0.0001, 1.0, -0.4, 0.78};

  auto *AC = F_->createAcos("Acos", X);

  auto *save = F_->createSave("save", AC);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), acos(XH.at({i})), 1E-5);
  }
}

/// Range of atan domain is [-1,1] and the range of output
/// is [-pi/2, pi/2]
TEST_P(OperatorTest, Atan_FloatTy) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {-0.34, 0.32, 0.0001, 1.0, -0.4, 0.78};

  auto *AT = F_->createAtan("Atan", X);

  auto *save = F_->createSave("save", AT);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), atan(XH.at({i})), 1E-5);
  }
}

/// Range of asin domain is [-1,1] and the range of output
/// is [-pi/2, pi/2]
TEST_P(OperatorTest, Asin_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto qParams = glow::quantization::chooseQuantizationParams({-1, 1});
  auto oParams = glow::quantization::chooseQuantizationParams({-1.57, 1.57});
  auto *data =
      mod_.uniqueType(ElemKind::Int8QTy, {6}, qParams.scale, qParams.offset);

  auto OT =
      mod_.uniqueType(ElemKind::Int8QTy, {6}, oParams.scale, oParams.offset);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {-0.34, 0.32, 0.0001, 1.0, -0.4, 0.78};
  auto *XQ = F_->createQuantize("quantizeQ", X, data);
  auto *ASQ = F_->createAsin("Asin", OT, XQ);

  auto *AS = F_->createDequantize("dequantize", ASQ, ElemKind::FloatTy);

  auto *save = F_->createSave("save", AS);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), asin(XH.at({i})), 0.25);
  }
}

/// Range of acos domain is [-1,1] and the range of output
/// is [0, pi]
TEST_P(OperatorTest, Acos_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto qParams = glow::quantization::chooseQuantizationParams({-1, 1});
  auto oParams = glow::quantization::chooseQuantizationParams({0, 3.14});
  auto *data =
      mod_.uniqueType(ElemKind::Int8QTy, {6}, qParams.scale, qParams.offset);

  auto OT =
      mod_.uniqueType(ElemKind::Int8QTy, {6}, oParams.scale, oParams.offset);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {-0.34, 0.32, 0.0001, 1.0, -0.4, 0.78};
  auto *XQ = F_->createQuantize("quantizeQ", X, data);
  auto *ACQ = F_->createAcos("Acos", OT, XQ);

  auto *AC = F_->createDequantize("dequantize", ACQ, ElemKind::FloatTy);

  auto *save = F_->createSave("save", AC);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), acos(XH.at({i})), 0.25);
  }
}

/// Range of atan domain is [-1,1] and the range of output
/// is [-pi/2, pi/2]
TEST_P(OperatorTest, Atan_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {6}, "X", false);
  auto qParams = glow::quantization::chooseQuantizationParams({-1, 1});
  auto oParams = glow::quantization::chooseQuantizationParams({-1.57, 1.57});
  auto *data =
      mod_.uniqueType(ElemKind::Int8QTy, {6}, qParams.scale, qParams.offset);

  auto OT =
      mod_.uniqueType(ElemKind::Int8QTy, {6}, oParams.scale, oParams.offset);
  auto XH = bindings_.allocate(X)->getHandle();
  XH = {-0.34, 0.32, 0.0001, 1.0, -0.4, 0.78};
  auto *XQ = F_->createQuantize("quantizeQ", X, data);
  auto *ATQ = F_->createAtan("Atan", OT, XQ);

  auto *AT = F_->createDequantize("dequantize", ATQ, ElemKind::FloatTy);

  auto *save = F_->createSave("save", AT);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto saveH = saveTensor->getHandle();

  for (dim_t i = 0; i < 6; i++) {
    EXPECT_NEAR(saveH.at({i}), atan(XH.at({i})), 0.25);
  }
}

/// Helper to test Logit using \p DTy.
template <typename DataType>
static void testLogit(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind DTy, float allowedError) {
  constexpr auto eps = 1E-6f; // the default in Caffe2
  constexpr dim_t size = 10;  // sample size for randomized tests

  auto *input = mod.createPlaceholder(DTy, {size}, "input", false);
  // generate the input data in (0.0f, 1.0f) (probabilites including degenerate
  // cases) and test that afterward the input data is clamped in
  // (eps, 1 - eps) as in Caffe2.
  bindings.allocate(input)->getHandle<DataType>().randomize(0.0f, 1.0f,
                                                            mod.getPRNG());

  auto *logitDiff = F->createLogit("logitDiff", input, eps);
  auto *saveDiff = F->createSave("saveDiff", logitDiff);
  bindings.allocate(saveDiff->getPlaceholder());

  // property: zero-sum for the log-odds for complementary events probabilities
  // i.e., logit(p) + logit(1 - p) == 0
  Node *const1 = F->createSplat("const1", input->getType(), 1.0);
  Node *complInput = F->createSub("sub", const1, input);
  Node *logitCompl = F->createLogit("logitCompl", complInput, eps);
  auto *saveCompl = F->createSave("saveCompl", logitCompl);
  bindings.allocate(saveCompl->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // results: differential test against the oracle
  auto resultDiffH =
      bindings.get(saveDiff->getPlaceholder())->getHandle<DataType>();
  auto inputH = bindings.get(input)->getHandle<DataType>();

  // results: zero-sum property
  auto resultComplH =
      bindings.get(saveCompl->getPlaceholder())->getHandle<DataType>();

  // differential test:
  // ensure we match an oracle `logit_test` (a C++ reimplementation test)
  auto clamp_test = [](float v, float lo, float hi) {
    return std::max(std::min(v, hi), lo);
  };
  auto logit_test = [clamp_test](float x, float eps = 1E-6f) {
    float p = clamp_test(x, eps, 1.0f - eps);
    return std::log(p / (1.0f - p));
  };

  // property: the logit function is the right-inverse of the logistic function
  // i.e., logistic(logit(p)) == p
  auto logistic_test = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

  for (dim_t i = 0; i != size; ++i) {
    // differential test against the oracle
    EXPECT_NEAR(resultDiffH.at({i}), logit_test(inputH.at({i})), allowedError);
    // zero-sum property
    EXPECT_NEAR(resultComplH.at({i}) + resultDiffH.at({i}), 0.0f, allowedError);
    // right-inverse property
    EXPECT_NEAR(logistic_test(resultDiffH.at({i})),
                clamp_test(inputH.at({i}), eps, 1.0f - eps), allowedError);
  }
}

/// Test the Logit operator using FloatTy.
TEST_P(OperatorTest, Logit_Float) {
  CHECK_IF_ENABLED();
  testLogit<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 1E-5);
}

/// Test the Logit operator using Float16Ty.
TEST_P(OperatorTest, Logit_Float16) {
  CHECK_IF_ENABLED();
  testLogit<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.002);
}

/// Test the Logit operator using Float16Ty.
TEST_P(OperatorTest, Logit_BFloat16) {
  CHECK_IF_ENABLED();
  testLogit<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, 0.05);
}

/// Helper to test CmpEQ using \p DTy.
template <typename DataType>
static void testCmpEQ(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind DTy) {
  auto *X = mod.createPlaceholder(DTy, {2, 7}, "X", false);
  // Values listed here in the dynamic range of both int32_t and int64_t
  bindings.allocate(X)->getHandle<DataType>() = {
      0, 1, 17, 876, 1000, 44444, 65535, 0, 1, 17, 876, 1000, 44444, 65535};
  auto *Y = mod.createPlaceholder(DTy, {2, 7}, "Y", false);
  bindings.allocate(Y)->getHandle<DataType>() = {
      1, 2, 16, 900, 1111, 44544, 65534, 0, 1, 17, 876, 1000, 44444, 65535};

  auto *cmpEQ = F->createCmpEQ("cmpEQ", X, Y);
  auto *save = F->createSave("save", cmpEQ);
  auto *saveTensor = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  auto saveH = saveTensor->getHandle<bool>();
  for (dim_t i = 0; i < 7; ++i) {
    EXPECT_FALSE(saveH.at({0, i}));
  }
  for (dim_t i = 0; i < 7; ++i) {
    EXPECT_TRUE(saveH.at({1, i}));
  }
}

/// Test the CmpEQ operator using Int64ITy.
TEST_P(OperatorTest, CmpEQ_Int64) {
  CHECK_IF_ENABLED();
  testCmpEQ<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test the CmpEQ operator using Int32ITy.
TEST_P(OperatorTest, CmpEQ_Int32) {
  CHECK_IF_ENABLED();
  testCmpEQ<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Check that the add operator works properly with FP16.
TEST_P(OperatorTest, FP16Add) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "A", false);
  bindings_.allocate(inputA)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "B", false);
  bindings_.allocate(inputB)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *Pool = F_->createAdd("pool", inputA, inputB);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleA = bindings_.get(inputA)->getHandle<float16_t>();
  auto handleB = bindings_.get(inputB)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), handleA.raw(idx) + handleB.raw(idx));
  }
}

/// Check that the add operator works properly with FP16.
TEST_P(OperatorTest, BFloat16Add) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1}, "A", false);
  bindings_.allocate(inputA)->getHandle<bfloat16_t>().randomize(-3.0, 3.0,
                                                                PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1}, "B", false);
  bindings_.allocate(inputB)->getHandle<bfloat16_t>().randomize(-3.0, 3.0,
                                                                PRNG);
  auto *Pool = F_->createAdd("pool", inputA, inputB);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<bfloat16_t>();
  auto handleA = bindings_.get(inputA)->getHandle<bfloat16_t>();
  auto handleB = bindings_.get(inputB)->getHandle<bfloat16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), handleA.raw(idx) + handleB.raw(idx));
  }
}

TEST_P(OperatorTest, matmul) {
  CHECK_IF_ENABLED();

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = saveTensor->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Test that cloneFunInsideFun works correctly with matmuls.
TEST_P(OperatorTest, matmul_ParCloneTest10) {
  CHECK_IF_ENABLED();

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  CompilationContext cctx;
  const unsigned parallelCount = 10;
  auto resultTensors = cloneFunInsideFun(std::make_pair(F_, saveTensor),
                                         &bindings_, cctx, parallelCount);

  EXPECT_EQ(resultTensors.size(), parallelCount);

  EE_.compile(cctx);
  EE_.run(bindings_);

  for (Tensor *T : resultTensors) {
    auto H = T->getHandle();
    EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
    EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
    EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
  }
}

/// Test that compareAgainstInterpreter works correctly along with quantization
/// and parallel cloning.
TEST_P(OperatorStatelessTest, matmulQuantized_InterpCompareParClone) {
  CHECK_IF_ENABLED();

  constexpr unsigned parallelCount = 10;
  compareAgainstInterpreter(
      getBackendName(),
      [](PlaceholderBindings &bindings, ExecutionEngine &EE) {
        Module &mod = EE.getModule();
        Function *F = mod.createFunction("main");
        Placeholder *lhs =
            mod.createPlaceholder(ElemKind::FloatTy, {3, 2}, "lhs", false);
        Placeholder *rhs =
            mod.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
        bindings.allocate(lhs)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
        bindings.allocate(rhs)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

        MatMulNode *R = F->createMatMul("MM", lhs, rhs);

        SaveNode *save = F->createSave("save", R);
        Tensor *saveTensor = bindings.allocate(save->getPlaceholder());
        return std::make_pair(F, saveTensor);
      },
      ElemKind::FloatTy, ElemKind::Int8QTy, 0.006, parallelCount);
}

/// Check that the matmul operator behaves correctly with FP16.
TEST_P(OperatorTest, FP16Matmul) {
  CHECK_IF_ENABLED();

  auto *lhs = mod_.createPlaceholder(ElemKind::Float16Ty, {3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::Float16Ty, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle<float16_t>() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle<float16_t>() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = saveTensor->getHandle<float16_t>();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Check that the matmul operator behaves correctly with FP16.
TEST_P(OperatorTest, BFloat16Matmul) {
  CHECK_IF_ENABLED();

  auto *lhs =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {3, 2}, "lhs", false);
  auto *rhs =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle<bfloat16_t>() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle<bfloat16_t>() = {7, 10};

  auto *R = F_->createMatMul("MM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = saveTensor->getHandle<bfloat16_t>();
  EXPECT_NEAR(H.at({0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({2, 0}), 95, 0.001);
}

/// Test that the broadcasted batch mat mul operator works as expected.
TEST_P(OperatorTest, BroadcastedBatchMatMul) {
  CHECK_IF_ENABLED();

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1,  2,  3,  4,  5,  6,
                                          -1, -2, -3, -4, -5, -6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -27, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -61, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -95, 0.001);
}

/// Test that the broadcasted batch mat mul operator works as expected when the
/// RHS does not have to be tiled.
TEST_P(OperatorTest, NonBroadcastedBatchMatMul) {
  CHECK_IF_ENABLED();
  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 2}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(rhs)->getHandle() = {7, 10};

  auto *R = F_->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
}

TEST_P(OperatorTest, ParallelBatchMatMul) {
  CHECK_IF_ENABLED();

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "lhs", false);
  auto *rhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 1}, "rhs", false);
  bindings_.allocate(lhs)->getHandle() = {1,  2,  3,  4,  5,  6,
                                          -1, -2, -3, -4, -5, -6};
  bindings_.allocate(rhs)->getHandle() = {7, 10, 12, -1};

  auto *R = F_->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0, 0}), 27, 0.001);
  EXPECT_NEAR(H.at({0, 1, 0}), 61, 0.001);
  EXPECT_NEAR(H.at({0, 2, 0}), 95, 0.001);
  EXPECT_NEAR(H.at({1, 0, 0}), -10, 0.001);
  EXPECT_NEAR(H.at({1, 1, 0}), -32, 0.001);
  EXPECT_NEAR(H.at({1, 2, 0}), -54, 0.001);
}

static FunctionTensorPair
createAndInitParallelBatchMatMulTest(glow::PlaceholderBindings &bindings,
                                     glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *lhs =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 50, 100}, "lhs", false);
  auto *rhs =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 100, 80}, "rhs", false);
  bindings.allocate(lhs)->getHandle().randomize(-0.1, 0.1, mod.getPRNG());
  bindings.allocate(rhs)->getHandle().randomize(-0.1, 0.1, mod.getPRNG());

  auto *R = F->createBatchMatMul("BMM", lhs, rhs);

  auto *save = F->createSave("save", R);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, ParallelBatchMatMul_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitParallelBatchMatMulTest, ElemKind::FloatTy,
      ElemKind::Float16Ty, 0.0005f, parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, ParallelBatchMatMul_BFloat16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitParallelBatchMatMulTest, ElemKind::FloatTy,
      ElemKind::BFloat16Ty, 0.0005f, parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, ParallelBatchMatMul_Int8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitParallelBatchMatMulTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.002f, parCloneCountOpt);
}

/// Helper to test BatchedReduceAdd using \p DTy.
template <typename DataType>
static void testBatchedReduceAdd(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *batch = mod.createPlaceholder(DTy, {2, 4}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {10, 20, 30, 40,
                                                     1,  2,  3,  4};

  auto *R = F->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {11, 22, 33, 44};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceAdd is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceAdd_Float) {
  CHECK_IF_ENABLED();

  testBatchedReduceAdd<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that BatchedReduceAdd is correctly supported in Float16Ty.
TEST_P(OperatorTest, batchedReduceAdd_Float16) {
  CHECK_IF_ENABLED();
  testBatchedReduceAdd<float16_t>(bindings_, mod_, F_, EE_,
                                  ElemKind::Float16Ty);
}

/// Test that BatchedReduceAdd is correctly supported in Float16Ty.
TEST_P(OperatorTest, batchedReduceAdd_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchedReduceAdd<bfloat16_t>(bindings_, mod_, F_, EE_,
                                   ElemKind::BFloat16Ty);
}

/// Test that BatchedReduceAdd works correctly reducing the outermost axis.
TEST_P(OperatorTest, batchedReduceAdd_outerAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {10, 20, 30, 40, 1, 2, 3, 4,
                                                   10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 4});
  expected.getHandle<float>() = {20, 40, 60, 80, 2, 4, 6, 8};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceAdd works correctly reducing an internal axis.
TEST_P(OperatorTest, batchedReduceAdd_innerAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {10, 20, 30, 40, 1, 2, 3, 4,
                                                   10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 1);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 4});
  expected.getHandle<float>() = {11, 22, 33, 44, 11, 22, 33, 44};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceAdd works correctly reducing the innermost axis.
TEST_P(OperatorTest, batchedReduceAdd_lastAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {10, 20, 30, 40, 1, 2, 3, 4,
                                                   10, 20, 30, 40, 1, 2, 3, 4};
  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 2);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle<float>() = {100, 10, 100, 10};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchReducedAdd works on a 4D input.
TEST_P(OperatorTest, batchedReduceAdd_4Dinput) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2, 4});
  expected.getHandle<float>() = {20, 40, 60, 80, 2, 4, 6, 8,
                                 20, 40, 60, 80, 2, 4, 6, 8};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchReducedAdd works on a 5D input.
TEST_P(OperatorTest, batchedReduceAdd_5Dinput) {
  CHECK_IF_ENABLED();
  auto *batch = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 2, 4},
                                       "batch", false);
  bindings_.allocate(batch)->getHandle<float>() = {
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4,
      10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceAdd("reduce.add", batch, /* axis */ 2);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2, 2, 4});
  expected.getHandle<float>() = {20, 40, 60, 80, 2,  4,  6,  8,  20, 40, 60,
                                 80, 2,  4,  6,  8,  20, 40, 60, 80, 2,  4,
                                 6,  8,  20, 40, 60, 80, 2,  4,  6,  8};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Helper to test VectorNorm using \p DTy.
template <typename DataType>
static void testVectorNorm(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind elemKind) {
  auto *input = mod.createPlaceholder(elemKind, {2, 3}, "norm", false);
  bindings.allocate(input)->getHandle<DataType>() = {1, 2, 3, -1, 1, 4};

  auto *R = F->createVectorNorm("vector.norm", input, /* axis */ 0, /* p */ 2);

  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(elemKind, {3});
  expected.getHandle<DataType>() = {1.4142, 2.2361, 5.0000};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that VectorNorm is correctly supported in FloatTy.
TEST_P(OperatorTest, VectorNorm_Float) {
  CHECK_IF_ENABLED();

  testVectorNorm<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that VectorNorm is correctly supported in Float16Ty.
TEST_P(OperatorTest, VectorNorm_Float16Ty) {
  CHECK_IF_ENABLED();

  testVectorNorm<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test that VectorNorm is correctly supported in BFloat16Ty.
TEST_P(OperatorTest, VectorNorm_BFloat16) {
  CHECK_IF_ENABLED();

  testVectorNorm<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Test that BatchedReduceAdd works correctly reducing an internal axis.
TEST_P(OperatorTest, VectorNorm_3D_innerAxis) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2}, "norm", false);
  bindings_.allocate(input)->getHandle<float>() = {0, 1, 2, 3, 4, 5, 6, 7};

  auto *R = F_->createVectorNorm("vector.norm", input, /* axis */ 1, /* p */ 2);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle<float>() = {2.0000, 3.1623, 7.2111, 8.6023};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Helper to test BatchedReduceMax using \p DTy.
template <typename DataType>
static void testBatchedReduceMax(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy) {

  auto *batch = mod.createPlaceholder(DTy, {2, 4}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {-10, 20, 30, 40,
                                                     -1,  2,  3,  4};
  auto *R = F->createBatchedReduceMax("reduce.Max", batch, /* axis */ 0);

  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {-1, 20, 30, 40};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Helper to test BatchedReduceMax using \p DTy.
template <typename DataType>
static void testBatchedReduceMaxMultiAxis(glow::PlaceholderBindings &bindings,
                                          glow::Module &mod, glow::Function *F,
                                          glow::ExecutionEngine &EE,
                                          ElemKind DTy) {
  auto *batch = mod.createPlaceholder(DTy, {2, 2, 2, 2}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {
      1, -2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto *R = F->createBatchedReduceMax("reduce.Max", batch, /* axis */ {1, 3});
  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {2, 2});
  expected.getHandle<DataType>() = {6, 8, 14, 16};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceMax is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceMax_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceMax<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that BatchedReduceMax is correctly supported in Int32Ty.
TEST_P(OperatorTest, batchedReduceMax_Int32) {
  CHECK_IF_ENABLED();
  testBatchedReduceMax<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test that BatchedReduceMax is correctly supported in Int64Ty.
TEST_P(OperatorTest, batchedReduceMax_Int64) {
  CHECK_IF_ENABLED();
  testBatchedReduceMax<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test that BatchedReduceMax is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceMaxMultiAxis_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceMaxMultiAxis<float>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy);
}

/// Test that BatchedReduceMax is correctly supported in Int32Ty.
TEST_P(OperatorTest, batchedReduceMaxMultiAxis_Int32) {
  CHECK_IF_ENABLED();
  testBatchedReduceMaxMultiAxis<int32_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int32ITy);
}

/// Test that BatchedReduceMax is correctly supported in Int64Ty.
TEST_P(OperatorTest, batchedReduceMaxMultiAxis_Int64) {
  CHECK_IF_ENABLED();
  testBatchedReduceMaxMultiAxis<int64_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int64ITy);
}

/// Helper to test BatchedReduceMin using \p DTy.
template <typename DataType>
static void testBatchedReduceMin(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy) {

  auto *batch = mod.createPlaceholder(DTy, {2, 4}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {10, 20, 30, 40,
                                                     1,  2,  3,  4};
  auto *R = F->createBatchedReduceMin("reduce.min", batch, /* axis */ 0);

  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {1, 2, 3, 4};

  EXPECT_TRUE(result->isEqual(expected));
}

/// Helper to test BatchedReduceMin using \p DTy.
template <typename DataType>
static void testBatchedReduceMinMultiAxis(glow::PlaceholderBindings &bindings,
                                          glow::Module &mod, glow::Function *F,
                                          glow::ExecutionEngine &EE,
                                          ElemKind DTy) {
  auto *batch = mod.createPlaceholder(DTy, {2, 2, 2, 2}, "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {
      1, -2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  auto *R = F->createBatchedReduceMin("reduce.min", batch, /* axis */ {1, 3});
  auto *save = F->createSave("save", R);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected(DTy, {2, 2});
  expected.getHandle<DataType>() = {-2, -4, 9, 11};
  EXPECT_TRUE(result->isEqual(expected));
}

/// Test that BatchedReduceMin is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceMin_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceMin<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that BatchedReduceMin is correctly supported in Int32Ty.
TEST_P(OperatorTest, batchedReduceMin_Int32) {
  CHECK_IF_ENABLED();
  testBatchedReduceMin<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test that BatchedReduceMin is correctly supported in Int64Ty.
TEST_P(OperatorTest, batchedReduceMin_Int64) {
  CHECK_IF_ENABLED();
  testBatchedReduceMin<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test that BatchedReduceMin is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceMinMultiAxis_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceMinMultiAxis<float>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy);
}

/// Test that BatchedReduceMin is correctly supported in Int32Ty.
TEST_P(OperatorTest, batchedReduceMinMultiAxis_Int32) {
  CHECK_IF_ENABLED();
  testBatchedReduceMinMultiAxis<int32_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int32ITy);
}

/// Test that BatchedReduceMin is correctly supported in Int64Ty.
TEST_P(OperatorTest, batchedReduceMinMultiAxis_Int64) {
  CHECK_IF_ENABLED();
  testBatchedReduceMinMultiAxis<int64_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int64ITy);
}

/// Helper to test BatchedReduceZeroDimResult using \p DTy.
template <typename DataType>
static void testBatchedReduceZeroDimResult(glow::PlaceholderBindings &bindings,
                                           glow::Module &mod, glow::Function *F,
                                           glow::ExecutionEngine &EE,
                                           ElemKind DTy) {
  auto *batch = createPlaceholderConditionallyQuantized(
      mod, DTy, {4}, "batch", /* isTrainable */ false, "N");
  bindings.allocate(batch)->getHandle<DataType>() = {2, 4, 6, 8};

  auto OT = uniqueTypeConditionallyQuantized(mod, DTy, {});
  auto *RA = F->createBatchedReduceAdd("reduce.add", OT, batch, /* axis */ 0);
  auto *RM = F->createBatchedReduceMean("reduce.mean", OT, batch, /* axis */ 0);
  auto *saveRA = F->createSave("saveRA", RA);
  auto *saveRM = F->createSave("saveRM", RM);
  auto *resultRA = bindings.allocate(saveRA->getPlaceholder());
  auto *resultRM = bindings.allocate(saveRM->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RAH = resultRA->getHandle<DataType>();
  auto RMH = resultRM->getHandle<DataType>();
  if (isQuantizedElemKind(DTy)) {
    EXPECT_EQ(RAH.at({}), static_cast<DataType>(20));
    EXPECT_EQ(RMH.at({}), static_cast<DataType>(5));
  } else {
    EXPECT_NEAR(RAH.at({}), 20, 0.001);
    EXPECT_NEAR(RMH.at({}), 5, 0.001);
  }
}

/// Test reduction down to a zero-dim tensor on FloatTy.
TEST_P(OperatorTest, batchedReduceZeroDimResult_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<float>(bindings_, mod_, F_, EE_,
                                        ElemKind::FloatTy);
}

/// Test reduction down to a zero-dim tensor on Float16Ty.
TEST_P(OperatorTest, batchedReduceZeroDimResult_Float16) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<float16_t>(bindings_, mod_, F_, EE_,
                                            ElemKind::Float16Ty);
}

/// Test reduction down to a zero-dim tensor on BFloat16Ty.
TEST_P(OperatorTest, batchedReduceZeroDimResult_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<bfloat16_t>(bindings_, mod_, F_, EE_,
                                             ElemKind::BFloat16Ty);
}

/// Test reduction down to a zero-dim tensor on Int8QTy.
TEST_P(OperatorTest, batchedReduceZeroDimResult_Int8) {
  CHECK_IF_ENABLED();
  testBatchedReduceZeroDimResult<int8_t>(bindings_, mod_, F_, EE_,
                                         ElemKind::Int8QTy);
}

/// Helper to test BatchedReduceAddWithAxis using \p DTy.
template <typename DataType>
static void testBatchedReduceAddWithAxis(glow::PlaceholderBindings &bindings,
                                         glow::Module &mod, glow::Function *F,
                                         glow::ExecutionEngine &EE,
                                         ElemKind DTy) {
  auto *batch = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2},
                                                        "batch", false);
  bindings.allocate(batch)->getHandle<DataType>() = {0, 1, 2, 3, 4,  5,
                                                     6, 7, 8, 9, 10, 11};

  auto OT1 = uniqueTypeConditionallyQuantized(mod, DTy, {2, 2});
  auto *R1 =
      F->createBatchedReduceAdd("reduce.add.axis.1", OT1, batch, /* axis */ 1);
  auto OT2 = uniqueTypeConditionallyQuantized(mod, DTy, {2, 3});
  auto *R2 =
      F->createBatchedReduceAdd("reduce.add.axis.2", OT2, batch, /* axis */ 2);
  auto *save1 = F->createSave("save1", R1);
  auto *save2 = F->createSave("save2", R2);

  auto *result1 = bindings.allocate(save1->getPlaceholder());
  auto *result2 = bindings.allocate(save2->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto expected1 = createTensorConditionallyQuantized(DTy, {2, 2});
  expected1.getHandle<DataType>() = {6, 9, 24, 27};
  EXPECT_TRUE(result1->isEqual(expected1));

  auto expected2 = createTensorConditionallyQuantized(DTy, {2, 3});
  expected2.getHandle<DataType>() = {1, 5, 9, 13, 17, 21};
  EXPECT_TRUE(result2->isEqual(expected2));
}

/// Test that batchedReduceAddWithAxis is correctly supported in FloatTy.
TEST_P(OperatorTest, batchedReduceAddWithAxis_Float) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<float>(bindings_, mod_, F_, EE_,
                                      ElemKind::FloatTy);
}

/// Test that batchedReduceAddWithAxis is correctly supported in Float16Ty.
TEST_P(OperatorTest, batchedReduceAddWithAxis_Float16) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<float16_t>(bindings_, mod_, F_, EE_,
                                          ElemKind::Float16Ty);
}

/// Test that batchedReduceAddWithAxis is correctly supported in BFloat16Ty.
TEST_P(OperatorTest, batchedReduceAddWithAxis_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<bfloat16_t>(bindings_, mod_, F_, EE_,
                                           ElemKind::BFloat16Ty);
}

/// Test that batchedReduceAddWithAxis is correctly supported in Int8QTy.
TEST_P(OperatorTest, batchedReduceAddWithAxis_Int8Q) {
  CHECK_IF_ENABLED();
  testBatchedReduceAddWithAxis<int8_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Int8QTy);
}

TEST_P(OperatorTest, batchedReduceAddQuantized) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 8; i++) {
    std::array<int32_t, 3> b{{BH.at({0, i}), BH.at({1, i}), BH.at({2, i})}};
    float s = BT->getScale() / OT->getScale();
    int32_t o = BT->getOffset();
    float result = (b[0] - o) + (b[1] - o) + (b[2] - o);
    result = s * result + OT->getOffset();

    EXPECT_NEAR(std::round(result), OH.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, batchedReduceAddQuantizedWithAxis) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {2, 3, 4}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R =
      F_->createBatchedReduceAdd("batched.reduce.add", OT, batch, /* axis */ 1);
  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 4; j++) {
      std::array<int32_t, 3> b{
          {BH.at({i, 0, j}), BH.at({i, 1, j}), BH.at({i, 2, j})}};
      float s = BT->getScale() / OT->getScale();
      int32_t o = BT->getOffset();
      float result = (b[0] - o) + (b[1] - o) + (b[2] - o);
      result = s * result + OT->getOffset();

      EXPECT_NEAR(std::round(result), OH.at({i, j}), 1.0);
    }
  }
}

TEST_P(OperatorTest, batchedReduceMean) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4}, "batch", false);
  bindings_.allocate(batch)->getHandle() = {10, 20, 30, 40, 1, 2, 3, 4};

  auto *R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0}), 5.5, 0.001);
  EXPECT_NEAR(H.at({1}), 11.0, 0.001);
  EXPECT_NEAR(H.at({2}), 16.5, 0.001);
  EXPECT_NEAR(H.at({3}), 22.0, 0.001);
}

TEST_P(OperatorTest, batchedReduceMeanWithAxis) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2}, "batch", false);
  bindings_.allocate(batch)->getHandle() = {0, 1, 2, 3, 4,  5,
                                            6, 7, 8, 9, 10, 11};

  auto *R = F_->createBatchedReduceMean("reduce.add", batch, /* axis */ 1);

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = result->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 2.0, 0.001);
  EXPECT_NEAR(H.at({0, 1}), 3.0, 0.001);
  EXPECT_NEAR(H.at({1, 0}), 8.0, 0.001);
  EXPECT_NEAR(H.at({1, 1}), 9.0, 0.001);
}

TEST_P(OperatorTest, batchedReduceMeanQuantized) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {3, 8}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {8}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 8}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 0);

  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 8; i++) {
    std::array<int32_t, 3> b{{BH.at({0, i}), BH.at({1, i}), BH.at({2, i})}};
    float s = BT->getScale() / OT->getScale();
    int32_t o = BT->getOffset();
    float result = ((b[0] - o) + (b[1] - o) + (b[2] - o)) / 3;
    result = s * result + OT->getOffset();

    EXPECT_NEAR(std::round(result), OH.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, batchedReduceMeanQuantizedWithAxis) {
  CHECK_IF_ENABLED();

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, {2, 3, 4}, 0.5, 3);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {2, 4}, 2.0, -1);

  auto *batch =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2, 3, 4}, BT->getScale(),
                             BT->getOffset(), "batch", false);

  bindings_.allocate(batch)->getHandle<int8_t>() = {
      27, -31, 16,  7,  20, 34, -2, 8,   -10, 83, 29,  -17,
      19, 13,  -11, -9, 50, 58, 0,  -20, -72, 43, -25, -1};

  auto BH = bindings_.get(batch)->getHandle<int8_t>();

  auto *R = F_->createBatchedReduceMean("batched.reduce.add", OT, batch,
                                        /* axis */ 1);
  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 4; j++) {
      std::array<int32_t, 3> b{
          {BH.at({i, 0, j}), BH.at({i, 1, j}), BH.at({i, 2, j})}};
      float s = BT->getScale() / OT->getScale();
      int32_t o = BT->getOffset();
      float result = ((b[0] - o) + (b[1] - o) + (b[2] - o)) / 3;
      result = s * result + OT->getOffset();

      EXPECT_NEAR(std::round(result), OH.at({i, j}), 1.0);
    }
  }
}

/// Verify that batchedReduceMean optimization using AvgPool works correctly.
TEST_P(OperatorTest, batchedReduceMeanUsingAvgPool) {
  CHECK_IF_ENABLED();

  std::vector<dim_t> dims = {3, 20, 4, 8};

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, dims, "batch", false, "NHWC");

  auto IH = bindings_.allocate(batch)->getHandle();
  IH.randomize(1.0, 100.0, mod_.getPRNG());

  auto *R = F_->createBatchedReduceMean("reduce.mean", batch, {2, 3});

  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);
  auto H = result->getHandle();

  std::array<std::array<float, 20>, 3> results{};
  for (dim_t i = 0; i < dims[0]; i++) {
    for (dim_t j = 0; j < dims[1]; j++) {
      for (dim_t k = 0; k < dims[2]; k++) {
        for (dim_t l = 0; l < dims[3]; l++) {
          results[i][j] += IH.at({i, j, k, l});
        }
      }
      results[i][j] /= (dims[2] * dims[3]);
      EXPECT_NEAR(H.at({i, j}), results[i][j], 0.001);
    }
  }
}

/// Verify that quantized batchedReduceMean optimization using AvgPool works
/// correctly.
TEST_P(OperatorTest, batchedReduceMeanUsingAvgPoolQuantized) {
  CHECK_IF_ENABLED();

  std::vector<dim_t> dims = {2, 3, 3, 4};

  auto BT = mod_.uniqueType(ElemKind::Int8QTy, dims, 1, 0);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {dims[0], dims[1]}, 1, 0);
  auto *batch = mod_.createPlaceholder(ElemKind::Int8QTy, dims, BT->getScale(),
                                       BT->getOffset(), "batch", false);

  auto IH = bindings_.allocate(batch)->getHandle<int8_t>();
  IH.randomize(1, 100, mod_.getPRNG());

  auto *R = F_->createBatchedReduceMean("reduce.mean", OT, batch, {2, 3});

  auto *save = F_->createSave("save", R);
  auto OH = bindings_.allocate(save->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::array<std::array<float, 3>, 2> results{};
  float s = BT->getScale() / OT->getScale();
  for (dim_t i = 0; i < dims[0]; i++) {
    for (dim_t j = 0; j < dims[1]; j++) {
      for (dim_t k = 0; k < dims[2]; k++) {
        int32_t o = BT->getOffset();
        for (dim_t l = 0; l < dims[3]; l++) {
          results[i][j] += IH.at({i, j, k, l}) - o;
        }
      }
      results[i][j] = s * results[i][j] + OT->getOffset();
      results[i][j] /= (dims[2] * dims[3]);
      EXPECT_NEAR(std::round(results[i][j]), OH.at({i, j}), 1.0);
    }
  }
}

/// Test that the BatchedAdd operator works.
TEST_P(OperatorTest, BatchedAdd) {
  CHECK_IF_ENABLED();

  auto *batch =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 3}, "batch", false);
  auto *added =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "added", false);

  bindings_.allocate(batch)->getHandle() = {9, 8, 7, 6, 5,  4,  3,  4,  5,
                                            6, 7, 8, 9, 10, 11, 12, 13, 14};
  bindings_.allocate(added)->getHandle().clear(1.0);

  auto *R = F_->createBatchedAdd("batch.add", batch, added);
  auto *save = F_->createSave("save", R);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto BH = bindings_.get(batch)->getHandle();
  auto RH = result->getHandle();
  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 3; j++) {
      for (dim_t k = 0; k < 3; k++) {
        EXPECT_NEAR(RH.at({i, j, k}), BH.at({i, j, k}) + 1.0, 0.001);
      }
    }
  }
}

/// Broadcast Tensor of shape (2,1,1) to (2,4,2) with axis 0.
TEST_P(OperatorTest, broadcastSimple) {
  CHECK_IF_ENABLED();

  const dim_t numDims_A = 3;
  const dim_t dimY_A = 2;
  const dim_t dimZ_A = 4;
  const dim_t dimW_A = 2;
  const dim_t dims_A[numDims_A] = {dimY_A, dimZ_A, dimW_A};

  const dim_t numDims_B = 3;
  const dim_t dimY_B = 2;
  const dim_t dimZ_B = 1;
  const dim_t dimW_B = 1;
  const dim_t dims_B[numDims_B] = {dimY_B, dimZ_B, dimW_B};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, dims_B, "B", false);
  auto *QB =
      mod_.createPlaceholder(ElemKind::Int8QTy, dims_B, 1.1, -2, "QB", false);
  auto H_B = bindings_.allocate(B)->getHandle();
  auto H_QB = bindings_.allocate(QB)->getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {35, -18};

  const unsigned axis = 0;

  auto *R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto *QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);

  auto *save = F_->createSave("save", R);
  auto *broadcasted = bindings_.allocate(save->getPlaceholder());

  auto *saveQ = F_->createSave("saveQ", QR);
  auto *broadcastedQ = bindings_.allocate(saveQ->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto broadcastedBHandle = broadcasted->getHandle();
  auto broadcastedQBHandle = broadcastedQ->getHandle<int8_t>();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(broadcastedBHandle.dims().size(), numDims_A);
  EXPECT_EQ(broadcastedQBHandle.dims().size(), numDims_A);
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(broadcastedBHandle.dims()[i], dims_A[i]);
    EXPECT_EQ(broadcastedQBHandle.dims()[i], dims_A[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const dim_t k_B = 0;
  const dim_t l_B = 0;
  for (dim_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B, l_B});
    const int8_t origValQ = H_QB.at({j_B, k_B, l_B});
    const dim_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (dim_t k_A = 0; k_A < dimZ_A; k_A++) {
      for (dim_t l_A = 0; l_A < dimW_A; l_A++) {
        EXPECT_EQ(broadcastedBHandle.at({j_A, k_A, l_A}), origVal);
        EXPECT_EQ(broadcastedQBHandle.at({j_A, k_A, l_A}), origValQ);
      }
    }
  }
}

/// Broadcast a Tensor of shape (2,1) to (3,2,4,2) with axis 1.
TEST_P(OperatorTest, broadcast) {
  CHECK_IF_ENABLED();

  const dim_t numDims_A = 4;
  const dim_t dimX_A = 3;
  const dim_t dimY_A = 2;
  const dim_t dimZ_A = 4;
  const dim_t dimW_A = 2;
  const dim_t dims_A[numDims_A] = {dimX_A, dimY_A, dimZ_A, dimW_A};

  const dim_t numDims_B = 2;
  const dim_t dimY_B = 2;
  const dim_t dimZ_B = 1;
  const dim_t dims_B[numDims_B] = {dimY_B, dimZ_B};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, dims_B, "B", false);
  auto *QB =
      mod_.createPlaceholder(ElemKind::Int8QTy, dims_B, 0.8, 3, "QB", false);

  auto H_B = bindings_.allocate(B)->getHandle();
  auto H_QB = bindings_.allocate(QB)->getHandle<int8_t>();
  H_B = {20, 10};
  H_QB = {-8, 41};

  const unsigned axis = 1;

  auto *R = F_->createBroadcast("broadcasted", B, dims_A, axis);
  auto *QR = F_->createBroadcast("broadcastedQ", QB, dims_A, axis);

  auto *save = F_->createSave("save", R);
  auto *broadcasted = bindings_.allocate(save->getPlaceholder());

  auto *saveQ = F_->createSave("saveQ", QR);
  auto *broadcastedQ = bindings_.allocate(saveQ->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto broadcastedBHandle = broadcasted->getHandle();
  auto broadcastedQBHandle = broadcastedQ->getHandle<int8_t>();
  // Verify broadcasted B has same shape.
  EXPECT_EQ(broadcastedBHandle.dims().size(), numDims_A);
  EXPECT_EQ(broadcastedQBHandle.dims().size(), numDims_A);
  for (size_t i = 0; i < broadcastedBHandle.dims().size(); i++) {
    EXPECT_EQ(broadcastedBHandle.dims()[i], dims_A[i]);
    EXPECT_EQ(broadcastedQBHandle.dims()[i], dims_A[i]);
  }

  // Look at the two values in X_B and verify in the three dimensions it was
  // broadcasted that the values were correctly broadcasted.
  const dim_t k_B = 0;
  for (dim_t j_B = 0; j_B < dimY_B; ++j_B) {
    const float origVal = H_B.at({j_B, k_B});
    const int8_t origValQ = H_QB.at({j_B, k_B});
    const dim_t j_A = j_B; // This dim was not broadcasted (dims were equal).
    for (dim_t i_A = 0; i_A < dimX_A; i_A++) {
      for (dim_t k_A = 0; k_A < dimZ_A; k_A++) {
        for (dim_t l_A = 0; l_A < dimW_A; l_A++) {
          EXPECT_EQ(broadcastedBHandle.at({i_A, j_A, k_A, l_A}), origVal);
          EXPECT_EQ(broadcastedQBHandle.at({i_A, j_A, k_A, l_A}), origValQ);
        }
      }
    }
  }
}

/// Perform a simple weighted sum.
TEST_P(OperatorTest, weightedSum) {
  CHECK_IF_ENABLED();

  // Create the data.
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "A", false);
  bindings_.allocate(A)->getHandle() = {1.0, 2.0, 3.0, 4.0};

  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "B", false);
  bindings_.allocate(B)->getHandle() = {5.0, 6.0, 7.0, 8.0};

  // Create the weights.
  auto *AW = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "AW", false);
  bindings_.allocate(AW)->getHandle() = {0.1f};

  auto *BW = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "BW", false);
  bindings_.allocate(BW)->getHandle() = {10.0f};

  // Create the weighted sum with the data and weights, and save it.
  auto *WS = F_->createWeightedSum("ws", {A, B}, {AW, BW});
  auto *save = F_->createSave("save", WS);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Verify the weighted sum was correctly calculated.
  auto resultH = saveTensor->getHandle();
  EXPECT_NEAR(resultH.at({0, 0}), 50.1, 1E-5);
  EXPECT_NEAR(resultH.at({0, 1}), 60.2, 1E-5);
  EXPECT_NEAR(resultH.at({1, 0}), 70.3, 1E-5);
  EXPECT_NEAR(resultH.at({1, 1}), 80.4, 1E-5);
}

/// Helper to test ReluSimple using \p DTy.
template <typename DataType>
static void testReluSimple(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *in = mod.createPlaceholder(DTy, {7}, "in", false);
  auto *relu = F->createRELU("relu", in);
  auto *save = F->createSave("relu", relu);
  auto *result = bindings.allocate(save->getPlaceholder());

  bindings.allocate(in)->getHandle<DataType>() = {0, -1, -2, -3, 4, 5, 6};

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = result->getHandle<DataType>();

  for (size_t i = 0; i < 7; i++) {
    if (i < 4) {
      EXPECT_EQ(resultH.raw(i), static_cast<DataType>(0));
    } else {
      EXPECT_EQ(resultH.raw(i), static_cast<DataType>(i));
    }
  }
}

/// Verify that the RELU operator works correctly for Float.
TEST_P(OperatorTest, ReluSimple_Float) {
  CHECK_IF_ENABLED();

  testReluSimple<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the RELU operator works correctly for Float16.
TEST_P(OperatorTest, ReluSimple_Float16) {
  CHECK_IF_ENABLED();
  testReluSimple<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Verify that the RELU operator works correctly for Float16.
TEST_P(OperatorTest, ReluSimple_BFloat16) {
  CHECK_IF_ENABLED();
  testReluSimple<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Helper to test PReluSimple using \p DTy.
template <typename DataType>
static void testPReluSimple(glow::PlaceholderBindings &bindings,
                            glow::Module &mod, glow::Function *F,
                            glow::ExecutionEngine &EE, ElemKind DTy,
                            double allowedError) {
  auto *in = mod.createPlaceholder(DTy, {7}, "in", false);
  auto *slope = mod.createPlaceholder(DTy, {7}, "slope", false);
  auto *prelu = F->createPRELU("prelu", in, slope);
  auto *save = F->createSave("prelu", prelu);
  auto *result = bindings.allocate(save->getPlaceholder());

  bindings.allocate(in)->getHandle<DataType>() = {0, -1, -2, -3, 4, 5, 6};
  bindings.allocate(slope)->getHandle<DataType>().randomize(0.1, 3.0,
                                                            mod.getPRNG());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = result->getHandle<DataType>();
  auto inH = bindings.get(in)->getHandle<DataType>();
  auto slopeH = bindings.get(slope)->getHandle<DataType>();

  for (size_t i = 0; i < 7; i++) {
    DataType expectedResult =
        slopeH.raw(i) * std::min<DataType>(0, inH.raw(i)) +
        std::max<DataType>(0, inH.raw(i));
    EXPECT_NEAR(resultH.at(i), expectedResult, allowedError);
  }
}

/// Verify that the PRELU operator works correctly for Float.
TEST_P(OperatorTest, PReluSimple_Float) {
  CHECK_IF_ENABLED();
  testPReluSimple<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 1E-32);
}

/// Verify that the PRELU operator works correctly for Float16.
TEST_P(OperatorTest, PReluSimple_Float16) {
  CHECK_IF_ENABLED();
  testPReluSimple<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                             1E-16);
}

/// Verify that the PRELU operator works correctly for BFloat16.
TEST_P(OperatorTest, PReluSimple_BFloat16) {
  CHECK_IF_ENABLED();
  testPReluSimple<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                              1E-16);
}

/// Helper to test Gelu using \p DTy.
template <typename DataType>
static void testGelu(glow::PlaceholderBindings &bindings, glow::Module &mod,
                     glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
                     double allowedError) {
  auto *in = mod.createPlaceholder(DTy, {7}, "in", false);
  auto *gelu = F->createGELU("gelu", in);
  auto *save = F->createSave("gelu", gelu);
  auto *result = bindings.allocate(save->getPlaceholder());

  bindings.allocate(in)->getHandle<DataType>().randomize(0.1, 3.0,
                                                         mod.getPRNG());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = result->getHandle<DataType>();
  auto inH = bindings.get(in)->getHandle<DataType>();
  // see https://arxiv.org/pdf/1606.08415.pdf
  float geluConst = 0.044715f;

  for (size_t i = 0; i < 7; i++) {
    float inHf = static_cast<float>(inH.raw(i));
    float expectedResult =
        0.5f * inHf *
        (1.0f + std::tanh(M_2_SQRTPI * M_SQRT1_2 *
                          (inHf + geluConst * std::pow(inHf, 3))));
    EXPECT_NEAR(resultH.at(i), expectedResult, allowedError);
  }
}

/// Verify that the GELU operator works correctly for Float.
TEST_P(OperatorTest, Gelu_Float) {
  CHECK_IF_ENABLED();
  testGelu<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 1E-6);
}

/// Verify that the GELU operator works correctly for Float16.
TEST_P(OperatorTest, Gelu_Float16) {
  CHECK_IF_ENABLED();
  testGelu<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 1.5E-2);
}

TEST_P(OperatorTest, TopK) {
  CHECK_IF_ENABLED();

  auto *inp =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 3}, "values", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {3, 1, 3}, "indices", false);

  bindings_.allocate(inp)->getHandle() = {
      28, 4, 411, 19, 42, 0.4f, 0.4f, 0.4f, -0.4f, 0.45f, 7, 5, 9, 8, 100,
  };
  bindings_.allocate(values);
  bindings_.allocate(indices);

  auto *R = F_->createTopK("TopK", inp, 3);

  F_->createSave("save.values", {R, 0}, values);
  F_->createSave("save.indices", {R, 1}, indices);

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto V = bindings_.get(values)->getHandle();
  auto I = bindings_.get(indices)->getHandle<int64_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 411);
  EXPECT_EQ(I.at({0, 0, 0}), 2);
  EXPECT_FLOAT_EQ(V.at({0, 0, 1}), 42);
  EXPECT_EQ(I.at({0, 0, 1}), 4);
  EXPECT_FLOAT_EQ(V.at({0, 0, 2}), 28);
  EXPECT_EQ(I.at({0, 0, 2}), 0);

  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 0.45);
  EXPECT_EQ(I.at({1, 0, 0}), 4);
  EXPECT_FLOAT_EQ(V.at({1, 0, 1}), 0.4);
  EXPECT_EQ(I.at({1, 0, 1}), 0);
  EXPECT_FLOAT_EQ(V.at({1, 0, 2}), 0.4);
  EXPECT_EQ(I.at({1, 0, 2}), 1);

  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 100);
  EXPECT_EQ(I.at({2, 0, 0}), 4);
  EXPECT_FLOAT_EQ(V.at({2, 0, 1}), 9);
  EXPECT_EQ(I.at({2, 0, 1}), 2);
  EXPECT_FLOAT_EQ(V.at({2, 0, 2}), 8);
  EXPECT_EQ(I.at({2, 0, 2}), 3);
}

template <typename DataType>
static void testArgMaxKeepDim(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2, 2},
                                                        "input", false, "NHWC");
  auto *argmax = mod.createPlaceholder(ElemKind::Int64ITy, {1, 3, 2, 2},
                                       "argmax", false, "NHWC");

  bindings.allocate(input)->getHandle<DataType>() = {
      11, 24, 33, 41, 15, 26, 37, 48, 12, 28, 31, 42,
      13, 24, 35, 46, 12, 28, 39, 40, 11, 22, 33, 47};
  bindings.allocate(argmax);

  auto *AM = F->createArgMax("argmax", input, 0, true);
  F->createSave("save.argmax", AM, argmax);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto I = bindings.get(argmax)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 1);
  EXPECT_EQ(I.raw(1), 0);
  EXPECT_EQ(I.raw(2), 1);
  EXPECT_EQ(I.raw(3), 1);
  EXPECT_EQ(I.raw(4), 0);
  EXPECT_EQ(I.raw(5), 1);
  EXPECT_EQ(I.raw(6), 1);
  EXPECT_EQ(I.raw(7), 0);
  EXPECT_EQ(I.raw(8), 0);
  EXPECT_EQ(I.raw(9), 0);
  EXPECT_EQ(I.raw(10), 1);
  EXPECT_EQ(I.raw(11), 1);
}

TEST_P(OperatorTest, FloatArgMaxKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxKeepDim<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedArgMaxKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxKeepDim<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

template <typename DataType>
static void testArgMaxNoKeepDim(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2, 2},
                                                        "input", false, "NHWC");
  auto *argmax =
      mod.createPlaceholder(ElemKind::Int64ITy, {2, 2, 2}, "argmax", false);

  bindings.allocate(input)->getHandle<DataType>() = {
      11, 24, 33, 41, 15, 26, 37, 48, 12, 28, 31, 42,
      13, 24, 35, 46, 12, 28, 39, 40, 11, 22, 33, 47};
  bindings.allocate(argmax);

  auto *AM = F->createArgMax("argmax", input, 1, false);
  F->createSave("save.argmax", AM, argmax);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto I = bindings.get(argmax)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 1);
  EXPECT_EQ(I.raw(1), 2);
  EXPECT_EQ(I.raw(2), 1);
  EXPECT_EQ(I.raw(3), 1);
  EXPECT_EQ(I.raw(4), 0);
  EXPECT_EQ(I.raw(5), 1);
  EXPECT_EQ(I.raw(6), 1);
  EXPECT_EQ(I.raw(7), 2);
}

TEST_P(OperatorTest, FloatArgMaxNoKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxNoKeepDim<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedArgMaxNoKeepDim) {
  CHECK_IF_ENABLED();
  testArgMaxNoKeepDim<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, FloatArgMaxNoKeepDimWithAxis1) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "input",
                                       false, "NHWC");
  auto *argmax =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 3, 4}, "argmax", false);

  bindings_.allocate(input)->getHandle<float>() = {
      -2.0031254,  1.6150867,  -0.7161922,  -0.25389647, -2.3863597,
      1.3052065,   -1.2064048, -0.12670185, 1.4289513,   0.38050872,
      -0.15112245, 1.360533,   -1.9638863,  -0.7602536,  0.68145376,
      1.1685915,   0.35476854, 1.0272173,   -1.554366,   -1.6835353,
      -1.4499142,  0.9042695,  1.0751117,   -1.0798755};

  bindings_.allocate(argmax);

  auto *AM =
      F_->createArgMax("argmax", input, /* axis */ 1, /* keepDims */ false);
  F_->createSave("save.argmax", AM, argmax);

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto I = bindings_.get(argmax)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 1);
  EXPECT_EQ(I.raw(1), 0);
  EXPECT_EQ(I.raw(2), 1);
  EXPECT_EQ(I.raw(3), 1);
  EXPECT_EQ(I.raw(4), 1);
  EXPECT_EQ(I.raw(5), 0);
  EXPECT_EQ(I.raw(6), 0);
  EXPECT_EQ(I.raw(7), 0);
  EXPECT_EQ(I.raw(8), 0);
  EXPECT_EQ(I.raw(9), 1);
  EXPECT_EQ(I.raw(10), 1);
  EXPECT_EQ(I.raw(11), 0);
}

TEST_P(OperatorTest, FloatArgMaxNoKeepDimWithAxis2) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "input",
                                       false, "NHWC");
  auto *argmax =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 2, 4}, "argmax", false);

  bindings_.allocate(input)->getHandle<float>() = {
      -0.11289205, -0.13215652, -1.184799,  0.2295995,   0.03064479,
      -0.28138036, -0.51807016, 0.89983666, -0.46122625, -0.70558083,
      0.43882176,  -0.6988644,  2.0838234,  -0.22806482, -0.6829437,
      0.70269305,  -0.8199907,  0.25597557, 0.3598691,   -0.9919779,
      2.069314,    -1.8825238,  1.2604765,  -0.78306365};

  bindings_.allocate(argmax);

  auto *AM =
      F_->createArgMax("argmax", input, /* axis */ 2, /* keepDims */ false);
  F_->createSave("save.argmax", AM, argmax);

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto I = bindings_.get(argmax)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 1);
  EXPECT_EQ(I.raw(1), 0);
  EXPECT_EQ(I.raw(2), 2);
  EXPECT_EQ(I.raw(3), 1);
  EXPECT_EQ(I.raw(4), 0);
  EXPECT_EQ(I.raw(5), 1);
  EXPECT_EQ(I.raw(6), 2);
  EXPECT_EQ(I.raw(7), 0);
}

template <typename DataType>
static void testArgMinKeepDim(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2, 2},
                                                        "input", false, "NHWC");
  auto *argmin = mod.createPlaceholder(ElemKind::Int64ITy, {1, 3, 2, 2},
                                       "argmin", false, "NHWC");

  bindings.allocate(input)->getHandle<DataType>() = {
      11, 24, 33, 41, 15, 26, 37, 48, 12, 28, 31, 42,
      13, 24, 35, 46, 12, 28, 39, 40, 11, 22, 33, 47};
  bindings.allocate(argmin);

  auto *AM = F->createArgMin("argmin", input, 0, true);
  F->createSave("save.argmin", AM, argmin);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto I = bindings.get(argmin)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 0);
  EXPECT_EQ(I.raw(1), 0);
  EXPECT_EQ(I.raw(2), 0);
  EXPECT_EQ(I.raw(3), 0);
  EXPECT_EQ(I.raw(4), 1);
  EXPECT_EQ(I.raw(5), 0);
  EXPECT_EQ(I.raw(6), 0);
  EXPECT_EQ(I.raw(7), 1);
  EXPECT_EQ(I.raw(8), 1);
  EXPECT_EQ(I.raw(9), 1);
  EXPECT_EQ(I.raw(10), 0);
  EXPECT_EQ(I.raw(11), 0);
}

TEST_P(OperatorTest, FloatArgMinKeepDim) {
  CHECK_IF_ENABLED();
  testArgMinKeepDim<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedArgMinKeepDim) {
  CHECK_IF_ENABLED();
  testArgMinKeepDim<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

template <typename DataType>
static void testArgMinNoKeepDim(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {2, 3, 2, 2},
                                                        "input", false, "NHWC");
  auto *argmin =
      mod.createPlaceholder(ElemKind::Int64ITy, {2, 2, 2}, "argmin", false);

  bindings.allocate(input)->getHandle<DataType>() = {
      11, 24, 33, 41, 15, 26, 37, 48, 12, 28, 31, 42,
      13, 24, 35, 46, 12, 28, 39, 40, 11, 22, 33, 47};
  bindings.allocate(argmin);

  auto *AM = F->createArgMin("argmin", input, 1, false);
  F->createSave("save.argmin", AM, argmin);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto I = bindings.get(argmin)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 0);
  EXPECT_EQ(I.raw(1), 0);
  EXPECT_EQ(I.raw(2), 2);
  EXPECT_EQ(I.raw(3), 0);
  EXPECT_EQ(I.raw(4), 2);
  EXPECT_EQ(I.raw(5), 2);
  EXPECT_EQ(I.raw(6), 2);
  EXPECT_EQ(I.raw(7), 1);
}

TEST_P(OperatorTest, FloatArgMinNoKeepDim) {
  CHECK_IF_ENABLED();
  testArgMinNoKeepDim<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedArgMinNoKeepDim) {
  CHECK_IF_ENABLED();
  testArgMinNoKeepDim<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, FloatArgMinNoKeepDimWithAxis1) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "input",
                                       false, "NHWC");
  auto *argmin =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 3, 4}, "argmin", false);

  bindings_.allocate(input)->getHandle<float>() = {
      -2.0031254,  1.6150867,  -0.7161922,  -0.25389647, -2.3863597,
      1.3052065,   -1.2064048, -0.12670185, 1.4289513,   0.38050872,
      -0.15112245, 1.360533,   -1.9638863,  -0.7602536,  0.68145376,
      1.1685915,   0.35476854, 1.0272173,   -1.554366,   -1.6835353,
      -1.4499142,  0.9042695,  1.0751117,   -1.0798755};

  bindings_.allocate(argmin);

  auto *AM =
      F_->createArgMin("argmin", input, /* axis */ 1, /* keepDims */ false);
  F_->createSave("save.argmin", AM, argmin);

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto I = bindings_.get(argmin)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 0);
  EXPECT_EQ(I.raw(1), 1);
  EXPECT_EQ(I.raw(2), 0);
  EXPECT_EQ(I.raw(3), 0);
  EXPECT_EQ(I.raw(4), 0);
  EXPECT_EQ(I.raw(5), 1);
  EXPECT_EQ(I.raw(6), 1);
  EXPECT_EQ(I.raw(7), 1);
  EXPECT_EQ(I.raw(8), 1);
  EXPECT_EQ(I.raw(9), 0);
  EXPECT_EQ(I.raw(10), 0);
  EXPECT_EQ(I.raw(11), 1);
}

TEST_P(OperatorTest, FloatArgMinNoKeepDimWithAxis2) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 4}, "input",
                                       false, "NHWC");
  auto *argmin =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 2, 4}, "argmin", false);

  bindings_.allocate(input)->getHandle<float>() = {
      -0.11289205, -0.13215652, -1.184799,  0.2295995,   0.03064479,
      -0.28138036, -0.51807016, 0.89983666, -0.46122625, -0.70558083,
      0.43882176,  -0.6988644,  2.0838234,  -0.22806482, -0.6829437,
      0.70269305,  -0.8199907,  0.25597557, 0.3598691,   -0.9919779,
      2.069314,    -1.8825238,  1.2604765,  -0.78306365};

  bindings_.allocate(argmin);

  auto *AM =
      F_->createArgMin("argmin", input, /* axis */ 2, /* keepDims */ false);
  F_->createSave("save.argmin", AM, argmin);

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto I = bindings_.get(argmin)->getHandle<int64_t>();
  EXPECT_EQ(I.raw(0), 2);
  EXPECT_EQ(I.raw(1), 2);
  EXPECT_EQ(I.raw(2), 0);
  EXPECT_EQ(I.raw(3), 2);
  EXPECT_EQ(I.raw(4), 1);
  EXPECT_EQ(I.raw(5), 2);
  EXPECT_EQ(I.raw(6), 0);
  EXPECT_EQ(I.raw(7), 1);
}

// Check that concatenating Nodes with multiple outputs works correctly.
TEST_P(OperatorTest, ConcatTopK) {
  CHECK_IF_ENABLED();

  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {4, 1, 2}, "indices", false);

  bindings_.allocate(inp1)->getHandle() = {1, 2, 3, 17.4f, -0.1f, -10.1f};
  bindings_.allocate(inp2)->getHandle() = {1, 2, -3, -17.4f, -0.1f, -10.1f};

  auto *R1 = F_->createTopK("TopK1", inp1, 2);
  auto *R2 = F_->createTopK("TopK2", inp2, 2);

  // Concat the values and indices separately, both on the 0th dimension,
  // matching the shapes of the values and indices variables above.
  auto *CV =
      F_->createConcat("Concat.Values", {R1->getValues(), R2->getValues()}, 0);
  auto *CI = F_->createConcat("Concat.Indices",
                              {R1->getIndices(), R2->getIndices()}, 0);

  auto *saveValues = F_->createSave("Save.Values", CV);
  auto *saveValuesTensor = bindings_.allocate(saveValues->getPlaceholder());

  auto *saveIndices = F_->createSave("Save.Indices", CI, indices);
  auto *saveIndicesTensor = bindings_.allocate(saveIndices->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto V = saveValuesTensor->getHandle();
  auto I = saveIndicesTensor->getHandle<int64_t>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 3);
  EXPECT_FLOAT_EQ(I.at({0, 0, 0}), 2);
  EXPECT_FLOAT_EQ(V.at({0, 0, 1}), 2);
  EXPECT_FLOAT_EQ(I.at({0, 0, 1}), 1);

  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 17.4f);
  EXPECT_FLOAT_EQ(I.at({1, 0, 0}), 0);
  EXPECT_FLOAT_EQ(V.at({1, 0, 1}), -0.1f);
  EXPECT_FLOAT_EQ(I.at({1, 0, 1}), 1);

  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 2);
  EXPECT_FLOAT_EQ(I.at({2, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({2, 0, 1}), 1);
  EXPECT_FLOAT_EQ(I.at({2, 0, 1}), 0);

  EXPECT_FLOAT_EQ(V.at({3, 0, 0}), -0.1f);
  EXPECT_FLOAT_EQ(I.at({3, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({3, 0, 1}), -10.1f);
  EXPECT_FLOAT_EQ(I.at({3, 0, 1}), 2);
}

// Check that matrix multiplication works well on some predefined values.
TEST_P(OperatorTest, matmul2) {
  CHECK_IF_ENABLED();

  auto *inp0 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input0", false);
  auto *inp1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input1", false);
  auto *inp2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2}, "input2", false);
  auto *rot = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "rot", false);

  float deg = 45.0 / 180.0 * 3.1415926;
  // Use the rotation matrix to manipulate some values.
  // https://en.wikipedia.org/wiki/Rotation_matrix
  bindings_.allocate(rot)->getHandle() = {
      cosf(deg),
      -sinf(deg),
      sinf(deg),
      cosf(deg),
  };

  // Some test vectors.
  bindings_.allocate(inp0)->getHandle() = {1, 4};
  bindings_.allocate(inp1)->getHandle() = {14, 2};
  bindings_.allocate(inp2)->getHandle() = {5, 2};

  auto *A0 = F_->createMatMul("m0", inp0, rot);
  auto *A1 = F_->createMatMul("m1", inp1, rot);
  auto *A2 = F_->createMatMul("m2", inp2, rot);

  auto *res0 = F_->createSave("save.values", A0);
  bindings_.allocate(res0->getPlaceholder());
  auto *res1 = F_->createSave("save.values", A1);
  bindings_.allocate(res1->getPlaceholder());
  auto *res2 = F_->createSave("save.values", A2);
  bindings_.allocate(res2->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  auto R0 = bindings_.get(res0->getPlaceholder())->getHandle();
  auto R1 = bindings_.get(res1->getPlaceholder())->getHandle();
  auto R2 = bindings_.get(res2->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(R0.at({0, 0}), 3.5355339);
  EXPECT_FLOAT_EQ(R0.at({0, 1}), 2.1213205);
  EXPECT_FLOAT_EQ(R1.at({0, 0}), 11.313709);
  EXPECT_FLOAT_EQ(R1.at({0, 1}), -8.485281);
  EXPECT_FLOAT_EQ(R2.at({0, 0}), 4.9497476);
  EXPECT_FLOAT_EQ(R2.at({0, 1}), -2.1213202);
}

template <typename HandleTy>
static void topK1Template(Module &mod_, Function *F_, ExecutionEngine &EE_,
                          PlaceholderBindings &bindings_,
                          ElemKind topKElemKind) {
  auto *inp =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 5}, "input", false);

  bindings_.allocate(inp)->getHandle() = {
      0, 18, 7, 16, 5, 14, 33, 2, 41, 0, 1, -23, 34, 4, -5,
  };

  auto *R = F_->createTopK("TopK", inp, 1, topKElemKind);

  auto *values = F_->createSave("save.values", {R, 0});
  bindings_.allocate(values->getPlaceholder());

  auto *indices = F_->createSave("save.indices", {R, 1});
  bindings_.allocate(indices->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto V = bindings_.get(values->getPlaceholder())->getHandle();
  auto I = bindings_.get(indices->getPlaceholder())->getHandle<HandleTy>();

  EXPECT_FLOAT_EQ(V.at({0, 0, 0}), 18);
  EXPECT_EQ(I.at({0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(V.at({1, 0, 0}), 41);
  EXPECT_EQ(I.at({1, 0, 0}), 3);
  EXPECT_FLOAT_EQ(V.at({2, 0, 0}), 34);
  EXPECT_EQ(I.at({2, 0, 0}), 2);
}
// Check the TopK operator for the special case of K=1.
TEST_P(OperatorTest, TopK1) {
  CHECK_IF_ENABLED();

  topK1Template<int64_t>(mod_, F_, EE_, bindings_, ElemKind::Int64ITy);
}

// Check the TopK operator for the special case of K=1.
TEST_P(OperatorTest, TopK1int32) {
  CHECK_IF_ENABLED();

  topK1Template<int32_t>(mod_, F_, EE_, bindings_, ElemKind::Int32ITy);
}

TEST_P(OperatorTest, QuantizedTopK) {
  CHECK_IF_ENABLED();

  auto *INV = mod_.createPlaceholder(ElemKind::Int8QTy, {3, 1, 5}, 1.2, 5,
                                     "input", false);
  bindings_.allocate(INV)->getHandle<int8_t>() = {
      -12, -28, -7, 8, -93, 0, 10, 3, -1, 10, -2, 3, -2, 3, 3,
  };

  auto *TK = F_->createTopK("TopK", INV, 3);

  auto *values = F_->createSave("save.values", TK->getValues());
  bindings_.allocate(values->getPlaceholder());
  auto *indices = F_->createSave("save.indices", TK->getIndices());
  bindings_.allocate(indices->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto VH = bindings_.get(values->getPlaceholder())->getHandle<int8_t>();
  auto IH = bindings_.get(indices->getPlaceholder())->getHandle<int64_t>();

  EXPECT_EQ(VH.at({0, 0, 0}), 8);
  EXPECT_EQ(IH.at({0, 0, 0}), 3);
  EXPECT_EQ(VH.at({0, 0, 1}), -7);
  EXPECT_EQ(IH.at({0, 0, 1}), 2);
  EXPECT_EQ(VH.at({0, 0, 2}), -12);
  EXPECT_EQ(IH.at({0, 0, 2}), 0);

  EXPECT_EQ(VH.at({1, 0, 0}), 10);
  EXPECT_EQ(IH.at({1, 0, 0}), 1);
  EXPECT_EQ(VH.at({1, 0, 1}), 10);
  EXPECT_EQ(IH.at({1, 0, 1}), 4);
  EXPECT_EQ(VH.at({1, 0, 2}), 3);
  EXPECT_EQ(IH.at({1, 0, 2}), 2);

  EXPECT_EQ(VH.at({2, 0, 0}), 3);
  EXPECT_EQ(IH.at({2, 0, 0}), 1);
  EXPECT_EQ(VH.at({2, 0, 1}), 3);
  EXPECT_EQ(IH.at({2, 0, 1}), 3);
  EXPECT_EQ(VH.at({2, 0, 2}), 3);
  EXPECT_EQ(IH.at({2, 0, 2}), 4);
}

/// Helper for testing Gather with different \p ITy / \p IndexType.
template <typename DataType, typename IndexType>
static void gatherFloatInputTest(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE, ElemKind DTy,
                                 ElemKind ITy) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1.0, 1.2],
            [2.3, 3.4],
            [1.0, 1.2],
            [2.3, 3.4],
        ],
        [
            [2.3, 3.4],
            [4.5, 5.7],
            [4.5, 5.7],
            [1.0, 1.2],
        ],
    ]
  */
  auto *data = mod.createPlaceholder(DTy, {3, 2}, "data", false);
  auto *indices = mod.createPlaceholder(ITy, {2, 4}, "indices", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  bindings.allocate(indices)->getHandle<IndexType>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto *R = F->createGather("gather", data, indices);

  auto *result = F->createSave("save", R);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor *resultT = bindings.get(result->getPlaceholder());
  Tensor expectedT(DTy, {2, 4, 2});
  expectedT.getHandle<DataType>() = {1.0, 1.2, 2.3, 3.4, 1.0, 1.2, 2.3, 3.4,
                                     2.3, 3.4, 4.5, 5.7, 4.5, 5.7, 1.0, 1.2};

  EXPECT_TRUE(resultT->isEqual(expectedT));
}

/// Test that Gather works with Float data and Int32 indices.
TEST_P(OperatorTest, GatherDataFloatIdxInt32) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float, int32_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test that Gather works with Float data and Int64 indices.
TEST_P(OperatorTest, GatherDataFloatIdxInt64) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float, int64_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::FloatTy, ElemKind::Int64ITy);
}
#endif

/// Test that Gather works with Float16 data and Int32 indices.
TEST_P(OperatorTest, GatherDataFloat16IdxInt32) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int32ITy);
}

/// Test that Gather works with BFloat16 data and Int32 indices.
TEST_P(OperatorTest, GatherDataBFloat16IdxInt32) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<bfloat16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, ElemKind::Int32ITy);
}

/// Test that Gather works with Float16 data and Int64 indices.
TEST_P(OperatorTest, GatherDataFloat16IdxInt64) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<float16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int64ITy);
}

/// Test that Gather works with BFloat16 data and Int64 indices.
TEST_P(OperatorTest, GatherDataBFloat16IdxInt64) {
  CHECK_IF_ENABLED();
  gatherFloatInputTest<bfloat16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, ElemKind::Int64ITy);
}

/// Helper for testing Gather with different \p ITy / \p IndexType.
template <typename IndexType>
static void gatherInt8InputTest(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind ITy) {
  /*
    DATA  = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ],
        [
            [3, 4],
            [5, 6],
            [5, 6],
            [1, 2],
        ],
    ]
  */
  auto *data =
      mod.createPlaceholder(ElemKind::Int8QTy, {3, 2}, 1.0, 0, "data", false);
  auto *indices = mod.createPlaceholder(ITy, {2, 4}, "indices", false);

  bindings.allocate(data)->getHandle<int8_t>() = {
      1, 2, 3, 4, 5, 6,
  };
  bindings.allocate(indices)->getHandle<IndexType>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto *R = F->createGather("gather", data, indices);

  auto *result = F->createSave("save", R);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor *resultT = bindings.get(result->getPlaceholder());
  Tensor expectedT(ElemKind::Int8QTy, {2, 4, 2}, 1.0, 0);
  expectedT.getHandle<int8_t>() = {1, 2, 3, 4, 1, 2, 3, 4,
                                   3, 4, 5, 6, 5, 6, 1, 2};

  EXPECT_TRUE(resultT->isEqual(expectedT));
}

/// Test that Gather works with Int8 data and Int32 indices.
TEST_P(OperatorTest, GatherDataInt8IdxInt32) {
  CHECK_IF_ENABLED();
  gatherInt8InputTest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test that Gather works with Int8 data and Int64 indices.
TEST_P(OperatorTest, GatherDataInt8IdxInt64) {
  CHECK_IF_ENABLED();
  gatherInt8InputTest<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}
#endif

/// Helper for testing GatherRanges with different \p ITy / \p IndexType.
template <typename DataType, typename IndexType>
void gatherRangesTest(glow::PlaceholderBindings &bindings_, glow::Module &mod_,
                      glow::Function *F_, glow::ExecutionEngine &EE_,
                      ElemKind DTy, ElemKind ITy) {
  /*
    DATA  = [1, 2, 3, 4, 5, 6]
    RANGES = [
      [
        [0, 1],
        [2, 2],
      ],
      [
        [4, 1],
        [5, 1],
      ]
    ]
    OUTPUT = [1, 3, 4, 5, 6]
    LENGTHS = [3, 2]
  */
  auto *data = createPlaceholderConditionallyQuantized(mod_, DTy, {6}, "data",
                                                       false, "N");
  auto *ranges = mod_.createPlaceholder(ITy, {2, 2, 2}, "ranges", false);

  bindings_.allocate(data)->getHandle<DataType>() = {1, 2, 3, 4, 5, 6};
  bindings_.allocate(ranges)->getHandle<IndexType>() = {0, 1, 2, 2, 4, 1, 5, 1};

  auto *R =
      F_->createGatherRanges("gatherranges", data, ranges, /*maxOutputSize=*/5);

  auto *output = F_->createSave("output", R->getOutput());
  auto *lengths = F_->createSave("lengths", R->getLengths());

  Tensor *outputT = bindings_.allocate(output->getPlaceholder());
  Tensor *lengthsT = bindings_.allocate(lengths->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto expectedOutputT = createTensorConditionallyQuantized(DTy, {5});
  expectedOutputT.getHandle<DataType>() = {1, 3, 4, 5, 6};
  EXPECT_TRUE(outputT->isEqual(expectedOutputT));

  Tensor expectedLengthsT(ITy, {2});
  expectedLengthsT.getHandle<IndexType>() = {3, 2};
  EXPECT_TRUE(lengthsT->isEqual(expectedLengthsT));
}

/// Test GatherRanges with Int64 data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataInt64IdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int64_t, int32_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int64ITy, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Int64 data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataInt64IdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int64_t, int64_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int64ITy, ElemKind::Int64ITy);
}
#endif

/// Test GatherRanges with Float data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataFloatIdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float, int32_t>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                                   ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Float data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataFloatIdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float, int64_t>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                                   ElemKind::Int64ITy);
}
#endif

/// Test GatherRanges with Float16 data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataFloat16IdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float16_t, int32_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Float16Ty, ElemKind::Int32ITy);
}

/// Test GatherRanges with BFloat16 data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataBFloat16IdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<bfloat16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Float16 data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataFloat16IdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<float16_t, int64_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Float16Ty, ElemKind::Int64ITy);
}

/// Test GatherRanges with BFloat16 data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataBFloat16IdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<bfloat16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, ElemKind::Int64ITy);
}
#endif

/// Test GatherRanges with Int8Q data and Int32 indices.
TEST_P(OperatorTest, GatherRangesDataInt8QIdxInt32) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int8_t, int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                                    ElemKind::Int32ITy);
}

#if DIM_T_BITWIDTH >= 64
/// Test GatherRanges with Int8Q data and Int64 indices.
TEST_P(OperatorTest, GatherRangesDataInt8QIdxInt64) {
  CHECK_IF_ENABLED();
  gatherRangesTest<int8_t, int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                                    ElemKind::Int64ITy);
}
#endif

/// Check if the code generation of transposes
/// is correct for tensors with 2 dimensions.
/// Note: This assumes that Tensor::transpose is correct.
TEST_P(OperatorTest, Transpose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle().randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::FloatTy, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check that transpose is supported for FP16.
TEST_P(OperatorTest, FP16Transpose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle<float16_t>().randomize(-3.0, 3.0,
                                                          mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::Float16Ty, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check that transpose is supported for BFloat16.
TEST_P(OperatorTest, BFloat16Transpose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BFloat16Ty, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle<bfloat16_t>().randomize(-3.0, 3.0,
                                                           mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::BFloat16Ty, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Check that transpose is supported for BoolTy.
TEST_P(OperatorTest, BoolTranspose2Dims) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BoolTy, {20, 13}, "A", false);
  bindings_.allocate(A)->getHandle<bool>().randomize(0, 1, mod_.getPRNG());

  auto *tr = F_->createTranspose("tr", A, {1, 0});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor dest(ElemKind::BoolTy, {13, 20});
  bindings_.get(A)->transpose(&dest, {1, 0});
  EXPECT_TRUE(bindings_.get(result->getPlaceholder())->isEqual(dest));
}

/// Helper to check if the code generation of transposes
/// is correct for tensors with 3 dimensions using \p DTy.
/// Note: This assumes that Tensor::transpose is correct.
template <typename DataType>
static void testTranspose3Dims(glow::PlaceholderBindings &bindings,
                               glow::Module &mod, glow::Function *F,
                               glow::ExecutionEngine &EE, ElemKind DTy) {
  constexpr dim_t dims[] = {20, 13, 7};
  auto *A = createPlaceholderConditionallyQuantized(mod, DTy, dims, "A", false);
  bindings.allocate(A)->getHandle<DataType>().randomize(-3.0, 3.0,
                                                        mod.getPRNG());

  int nbOfShuffle = 0;
  SaveNode *savedTransposes[6];
  unsigned_t shuffles[6][3];

  for (unsigned_t i = 0; i < 3; ++i) {
    for (unsigned_t j = 0; j < 3; ++j) {
      if (j == i) {
        continue;
      }
      for (unsigned_t k = 0; k < 3; ++k) {
        if (k == j || k == i) {
          continue;
        }
        shuffles[nbOfShuffle][0] = i;
        shuffles[nbOfShuffle][1] = j;
        shuffles[nbOfShuffle][2] = k;
        auto *tr = F->createTranspose("tr", A, shuffles[nbOfShuffle]);
        savedTransposes[nbOfShuffle] = F->createSave("saveTranspose", tr);
        bindings.allocate(savedTransposes[nbOfShuffle]->getPlaceholder());
        ++nbOfShuffle;
      }
    }
  }

  // We should have exactly 6 possible permutations for 3 dimensions.
  EXPECT_EQ(6, nbOfShuffle);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  for (int i = 0; i < 6; ++i) {
    auto dest = createTensorConditionallyQuantized(
        DTy,
        {dims[shuffles[i][0]], dims[shuffles[i][1]], dims[shuffles[i][2]]});
    bindings.get(A)->transpose(&dest, shuffles[i]);
    EXPECT_TRUE(
        bindings.get(savedTransposes[i]->getPlaceholder())->isEqual(dest));
  }
}

/// Test Transpose3Dims with Float.
TEST_P(OperatorTest, Transpose3Dims_Float) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test Transpose3Dims with Float16.
TEST_P(OperatorTest, Transpose3Dims_Float16) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test Transpose3Dims with BFloat16.
TEST_P(OperatorTest, Transpose3Dims_BFloat16) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<bfloat16_t>(bindings_, mod_, F_, EE_,
                                 ElemKind::BFloat16Ty);
}

/// Test Transpose3Dims with Int8.
TEST_P(OperatorTest, Transpose3Dims_Int8) {
  CHECK_IF_ENABLED();
  testTranspose3Dims<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test that Transpose optimization into Reshape yields expected results.
TEST_P(OperatorTest, TransposeIntoReshapeOptim) {
  CHECK_IF_ENABLED();
  auto *batch = mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 2, 4}, "batch",
                                       false, "NHWC");
  auto IH = bindings_.allocate(batch)->getHandle();
  for (size_t i = 0; i < 24; i++) {
    IH.raw(i) = i + 1;
  }

  Node *T = F_->createTranspose("transpose", batch, {1, 2, 0, 3}, "HWNC");
  Node *R = F_->createBatchedReduceMean("reduce.mean", T, {2, 3});
  SaveNode *O = F_->createSave("ret", R);
  bindings_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(O->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {3, 2};
  EXPECT_TRUE(result.dims().vec() == expectedDims);

  std::vector<float> expectedValues = {2.5f, 6.5f, 10.5f, 14.5f, 18.5f, 22.5f};
  for (size_t i = 0; i < 3 * 2; i++) {
    EXPECT_EQ(result.raw(i), expectedValues[i]);
  }
}

/// Helper to check the code generation for flip nodes.
template <typename elemType>
static void testFlip(glow::PlaceholderBindings &bindings, glow::Module &mod,
                     glow::Function *F, glow::ExecutionEngine &EE,
                     std::vector<elemType> inputData,
                     std::vector<elemType> expectedData,
                     llvm::ArrayRef<dim_t> dims, dim_t axis,
                     ElemKind elemKind = ElemKind::FloatTy) {

  // Create network.
  auto *input =
      createPlaceholderConditionallyQuantized(mod, elemKind, dims, "input",
                                              /* isTrainable */ false);
  auto *flip = F->createFlip("flip", input, axis);
  Placeholder *output = F->createSave("save", flip)->getPlaceholder();

  // Allocate input/output and initialize input.
  auto inputH = bindings.allocate(input)->getHandle<elemType>();
  auto outputH = bindings.allocate(output)->getHandle<elemType>();
  inputH = inputData;

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare output with reference.
  EXPECT_EQ(outputH.size(), expectedData.size());
  for (size_t i = 0; i < expectedData.size(); i++) {
    EXPECT_EQ(outputH.raw(i), expectedData[i]);
  }
}

/// Test Flip 1D with Int8.
TEST_P(OperatorTest, Flip1D_Int8) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<int8_t>(bindings_, mod_, F_, EE_, {1, 2, 3, 4}, {4, 3, 2, 1}, {4}, 0,
                   ElemKind::Int8QTy);
}

/// Test Flip 1D with Int32.
TEST_P(OperatorTest, Flip1D_Int32) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<int32_t>(bindings_, mod_, F_, EE_, {1, 2, 3, 4}, {4, 3, 2, 1}, {4},
                    0, ElemKind::Int32QTy);
}

/// Test Flip 1D with Int64.
TEST_P(OperatorTest, Flip1D_Int64) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<int64_t>(bindings_, mod_, F_, EE_, {1, 2, 3, 4}, {4, 3, 2, 1}, {4},
                    0, ElemKind::Int64ITy);
}

#define FLIP_3D_INPUT                                                          \
  { 1, 2, 3, 4, 5, 6, 7, 8 }
#define FLIP_3D_AXIS0                                                          \
  { 5, 6, 7, 8, 1, 2, 3, 4 }
#define FLIP_3D_AXIS1                                                          \
  { 3, 4, 1, 2, 7, 8, 5, 6 }
#define FLIP_3D_AXIS2                                                          \
  { 2, 1, 4, 3, 6, 5, 8, 7 }

#define FLIP_4D_INPUT                                                          \
  { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
#define FLIP_4D_AXIS0                                                          \
  { 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8 }
#define FLIP_4D_AXIS1                                                          \
  { 5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12 }
#define FLIP_4D_AXIS2                                                          \
  { 3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14 }
#define FLIP_4D_AXIS3                                                          \
  { 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15 }

#define FLIP_5D_INPUT                                                          \
  {                                                                            \
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32                             \
  }
#define FLIP_5D_AXIS0                                                          \
  {                                                                            \
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 1, 2, 3,   \
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16                           \
  }
#define FLIP_5D_AXIS1                                                          \
  {                                                                            \
    9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, \
        30, 31, 32, 17, 18, 19, 20, 21, 22, 23, 24                             \
  }
#define FLIP_5D_AXIS2                                                          \
  {                                                                            \
    5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 21, 22, 23, 24, 17, \
        18, 19, 20, 29, 30, 31, 32, 25, 26, 27, 28                             \
  }
#define FLIP_5D_AXIS3                                                          \
  {                                                                            \
    3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14, 19, 20, 17, 18, 23, \
        24, 21, 22, 27, 28, 25, 26, 31, 32, 29, 30                             \
  }
#define FLIP_5D_AXIS4                                                          \
  {                                                                            \
    2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, \
        21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31                             \
  }

#define FLIP_6D_INPUT                                                          \
  {                                                                            \
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, \
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,    \
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,    \
        56, 57, 58, 59, 60, 61, 62, 63, 64                                     \
  }
#define FLIP_6D_AXIS0                                                          \
  {                                                                            \
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,    \
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 1, 2, 3, 4, 5, \
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,    \
        24, 25, 26, 27, 28, 29, 30, 31, 32                                     \
  }
#define FLIP_6D_AXIS1                                                          \
  {                                                                            \
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 1, 2, 3,   \
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 49, 50, 51, 52, 53, 54,  \
        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 33, 34, 35, 36, 37, 38, 39,    \
        40, 41, 42, 43, 44, 45, 46, 47, 48                                     \
  }
#define FLIP_6D_AXIS2                                                          \
  {                                                                            \
    9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, \
        30, 31, 32, 17, 18, 19, 20, 21, 22, 23, 24, 41, 42, 43, 44, 45, 46,    \
        47, 48, 33, 34, 35, 36, 37, 38, 39, 40, 57, 58, 59, 60, 61, 62, 63,    \
        64, 49, 50, 51, 52, 53, 54, 55, 56                                     \
  }
#define FLIP_6D_AXIS3                                                          \
  {                                                                            \
    5, 6, 7, 8, 1, 2, 3, 4, 13, 14, 15, 16, 9, 10, 11, 12, 21, 22, 23, 24, 17, \
        18, 19, 20, 29, 30, 31, 32, 25, 26, 27, 28, 37, 38, 39, 40, 33, 34,    \
        35, 36, 45, 46, 47, 48, 41, 42, 43, 44, 53, 54, 55, 56, 49, 50, 51,    \
        52, 61, 62, 63, 64, 57, 58, 59, 60                                     \
  }
#define FLIP_6D_AXIS4                                                          \
  {                                                                            \
    3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14, 19, 20, 17, 18, 23, \
        24, 21, 22, 27, 28, 25, 26, 31, 32, 29, 30, 35, 36, 33, 34, 39, 40,    \
        37, 38, 43, 44, 41, 42, 47, 48, 45, 46, 51, 52, 49, 50, 55, 56, 53,    \
        54, 59, 60, 57, 58, 63, 64, 61, 62                                     \
  }
#define FLIP_6D_AXIS5                                                          \
  {                                                                            \
    2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, \
        21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31, 34, 33, 36, 35, 38, 37,    \
        40, 39, 42, 41, 44, 43, 46, 45, 48, 47, 50, 49, 52, 51, 54, 53, 56,    \
        55, 58, 57, 60, 59, 62, 61, 64, 63                                     \
  }

/// Test Flip 1D with Float.
TEST_P(OperatorTest, Flip1D_Axis0_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, {1, 2}, {2, 1}, {2}, 0);
}

/// Test Flip 2D with Float.
TEST_P(OperatorTest, Flip2D_Axis0_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, {1, 2, 3, 4}, {3, 4, 1, 2}, {2, 2},
                  0);
}
TEST_P(OperatorTest, Flip2D_Axis1_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, {1, 2, 3, 4}, {2, 1, 4, 3}, {2, 2},
                  1);
}

/// Test Flip 3D with Float.
TEST_P(OperatorTest, Flip3D_Axis0_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_3D_INPUT, FLIP_3D_AXIS0,
                  {2, 2, 2}, 0);
}
TEST_P(OperatorTest, Flip3D_Axis1_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_3D_INPUT, FLIP_3D_AXIS1,
                  {2, 2, 2}, 1);
}
TEST_P(OperatorTest, Flip3D_Axis2_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_3D_INPUT, FLIP_3D_AXIS2,
                  {2, 2, 2}, 2);
}

/// Test Flip 4D with Float.
TEST_P(OperatorTest, Flip4D_Axis0_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_4D_INPUT, FLIP_4D_AXIS0,
                  {2, 2, 2, 2}, 0);
}
TEST_P(OperatorTest, Flip4D_Axis1_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_4D_INPUT, FLIP_4D_AXIS1,
                  {2, 2, 2, 2}, 1);
}
TEST_P(OperatorTest, Flip4D_Axis2_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_4D_INPUT, FLIP_4D_AXIS2,
                  {2, 2, 2, 2}, 2);
}
TEST_P(OperatorTest, Flip4D_Axis3_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_4D_INPUT, FLIP_4D_AXIS3,
                  {2, 2, 2, 2}, 3);
}

/// Test Flip 5D with Float.
TEST_P(OperatorTest, Flip5D_Axis0_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_5D_INPUT, FLIP_5D_AXIS0,
                  {2, 2, 2, 2, 2}, 0);
}
TEST_P(OperatorTest, Flip5D_Axis1_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_5D_INPUT, FLIP_5D_AXIS1,
                  {2, 2, 2, 2, 2}, 1);
}
TEST_P(OperatorTest, Flip5D_Axis2_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_5D_INPUT, FLIP_5D_AXIS2,
                  {2, 2, 2, 2, 2}, 2);
}
TEST_P(OperatorTest, Flip5D_Axis3_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_5D_INPUT, FLIP_5D_AXIS3,
                  {2, 2, 2, 2, 2}, 3);
}
TEST_P(OperatorTest, Flip5D_Axis4_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_5D_INPUT, FLIP_5D_AXIS4,
                  {2, 2, 2, 2, 2}, 4);
}

/// Test Flip 6D with Float.
TEST_P(OperatorTest, Flip6D_Axis0_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_6D_INPUT, FLIP_6D_AXIS0,
                  {2, 2, 2, 2, 2, 2}, 0);
}
TEST_P(OperatorTest, Flip6D_Axis1_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_6D_INPUT, FLIP_6D_AXIS1,
                  {2, 2, 2, 2, 2, 2}, 1);
}
TEST_P(OperatorTest, Flip6D_Axis2_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_6D_INPUT, FLIP_6D_AXIS2,
                  {2, 2, 2, 2, 2, 2}, 2);
}
TEST_P(OperatorTest, Flip6D_Axis3_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_6D_INPUT, FLIP_6D_AXIS3,
                  {2, 2, 2, 2, 2, 2}, 3);
}
TEST_P(OperatorTest, Flip6D_Axis4_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_6D_INPUT, FLIP_6D_AXIS4,
                  {2, 2, 2, 2, 2, 2}, 4);
}
TEST_P(OperatorTest, Flip6D_Axis5_Float) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  testFlip<float>(bindings_, mod_, F_, EE_, FLIP_6D_INPUT, FLIP_6D_AXIS5,
                  {2, 2, 2, 2, 2, 2}, 5);
}

#undef FLIP_3D_INPUT
#undef FLIP_3D_AXIS0
#undef FLIP_3D_AXIS1
#undef FLIP_3D_AXIS2
#undef FLIP_4D_INPUT
#undef FLIP_4D_AXIS0
#undef FLIP_4D_AXIS1
#undef FLIP_4D_AXIS2
#undef FLIP_4D_AXIS3
#undef FLIP_5D_INPUT
#undef FLIP_5D_AXIS0
#undef FLIP_5D_AXIS1
#undef FLIP_5D_AXIS2
#undef FLIP_5D_AXIS3
#undef FLIP_5D_AXIS4
#undef FLIP_6D_INPUT
#undef FLIP_6D_AXIS0
#undef FLIP_6D_AXIS1
#undef FLIP_6D_AXIS2
#undef FLIP_6D_AXIS3
#undef FLIP_6D_AXIS4
#undef FLIP_6D_AXIS5

/// Check that gather on Int64ITy/size_t works.
TEST_P(OperatorTest, GatherSizeT) {
  CHECK_IF_ENABLED();

  /*
    DATA  = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]
    INDICES = [
        [0, 1, 0, 1],
        [1, 2, 2, 0],
    ]
    OUTPUT = [
        [
            [1, 2],
            [3, 4],
            [1, 2],
            [3, 4],
        ],
        [
            [3, 4],
            [5, 6],
            [5, 6],
            [1, 2],
        ],
    ]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 4}, "indices", false);

  bindings_.allocate(data)->getHandle<int64_t>() = {
      1, 2, 3, 4, 5, 6,
  };
  bindings_.allocate(indices)->getHandle<int64_t>() = {
      0, 1, 0, 1, 1, 2, 2, 0,
  };

  auto *R = F_->createGather("gather", data, indices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  EXPECT_EQ(H.at({0, 0, 0}), 1);
  EXPECT_EQ(H.at({0, 0, 1}), 2);
  EXPECT_EQ(H.at({0, 1, 0}), 3);
  EXPECT_EQ(H.at({0, 1, 1}), 4);
  EXPECT_EQ(H.at({0, 2, 0}), 1);
  EXPECT_EQ(H.at({0, 2, 1}), 2);
  EXPECT_EQ(H.at({0, 3, 0}), 3);
  EXPECT_EQ(H.at({0, 3, 1}), 4);

  EXPECT_EQ(H.at({1, 0, 0}), 3);
  EXPECT_EQ(H.at({1, 0, 1}), 4);
  EXPECT_EQ(H.at({1, 1, 0}), 5);
  EXPECT_EQ(H.at({1, 1, 1}), 6);
  EXPECT_EQ(H.at({1, 2, 0}), 5);
  EXPECT_EQ(H.at({1, 2, 1}), 6);
  EXPECT_EQ(H.at({1, 3, 0}), 1);
  EXPECT_EQ(H.at({1, 3, 1}), 2);
}

TEST_P(OperatorTest, BatchedGather) {
  CHECK_IF_ENABLED();

  /*
   DATA  = [
    [1.0, 1.2, 2.4, 4.5],
    [2.3, 3.4, 3.6, 2.3],
    [4.5, 5.7, 1.2, 4.5],
   ]

   INDICES = [0, 2],

   OUTPUT = [
    [1.0, 2.4],
    [2.3, 3.6],
    [4.5, 1.2],
   ]
   */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 4}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2}, "indices", false);

  bindings_.allocate(data)->getHandle() = {
      1.0f, 1.2f, 2.4f, 4.5f, 2.3f, 3.4f, 3.6f, 2.3f, 4.5f, 5.7f, 1.2f, 4.5f,
  };
  bindings_.allocate(indices)->getHandle<int64_t>() = {
      0,
      2,
  };

  // Create a batched gather (a single batch dimension).
  auto *R = F_->createGather("gather", data, indices, 1);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.4);
  EXPECT_FLOAT_EQ(H.at({1, 0}), 2.3);
  EXPECT_FLOAT_EQ(H.at({1, 1}), 3.6);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 4.5);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 1.2);
}

TEST_P(OperatorTest, ScatterData) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 1}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  bindings_.allocate(data)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  bindings_.allocate(indices)->getHandle<int64_t>() = {1, 3};
  bindings_.allocate(slices)->getHandle() = {-3, -4, -7, -8};

  auto *R = F_->createScatterData("scatterdata", data, indices, slices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();

  EXPECT_FLOAT_EQ(H.at({0, 0}), 1.0);
  EXPECT_FLOAT_EQ(H.at({0, 1}), 2.0);
  EXPECT_FLOAT_EQ(H.at({1, 0}), -3.0);
  EXPECT_FLOAT_EQ(H.at({1, 1}), -4.0);
  EXPECT_FLOAT_EQ(H.at({2, 0}), 5.0);
  EXPECT_FLOAT_EQ(H.at({2, 1}), 6.0);
  EXPECT_FLOAT_EQ(H.at({3, 0}), -7.0);
  EXPECT_FLOAT_EQ(H.at({3, 1}), -8.0);
  EXPECT_FLOAT_EQ(H.at({4, 0}), 9.0);
  EXPECT_FLOAT_EQ(H.at({4, 1}), 10.0);
}

TEST_P(OperatorTest, ScatterDataQuantized) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 1}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  bindings_.allocate(data)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  bindings_.allocate(indices)->getHandle<int64_t>() = {1, 3};
  bindings_.allocate(slices)->getHandle() = {-3, -4, -7, -8};

  auto qParams = glow::quantization::chooseQuantizationParams({-11, 11});
  auto dataTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 2}, qParams.scale, qParams.offset);
  auto slicesTy =
      mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, qParams.scale, qParams.offset);

  auto *dataQ = F_->createQuantize("quantizeQ", data, dataTy);
  auto *slicesQ = F_->createQuantize("quantizeS", slices, slicesTy);
  auto *SA = F_->createScatterData("scatterdata", dataQ, indices, slicesQ);
  auto *DQ = F_->createDequantize("dequantize", SA, ElemKind::FloatTy);

  auto *result = F_->createSave("save", DQ);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();

  EXPECT_NEAR(H.at({0, 0}), 1.0, 0.05);
  EXPECT_NEAR(H.at({0, 1}), 2.0, 0.05);
  EXPECT_NEAR(H.at({1, 0}), -3.0, 0.05);
  EXPECT_NEAR(H.at({1, 1}), -4.0, 0.05);
  EXPECT_NEAR(H.at({2, 0}), 5.0, 0.05);
  EXPECT_NEAR(H.at({2, 1}), 6.0, 0.05);
  EXPECT_NEAR(H.at({3, 0}), -7.0, 0.05);
  EXPECT_NEAR(H.at({3, 1}), -8.0, 0.05);
  EXPECT_NEAR(H.at({4, 0}), 9.0, 0.05);
  EXPECT_NEAR(H.at({4, 1}), 10.0, 0.05);
}

TEST_P(OperatorTest, ScatterDataNDimensionalSimple) {
  CHECK_IF_ENABLED();

  // Data = {{1,2},{3,4},{5,6}}
  // Slices = {-3,-4}
  // Indices = {{1,0},{1,1}}
  // Result = {{1,2},{-3,-4},{5,6}}
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init(6);
  std::iota(init.begin(), init.end(), 1);
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<int64_t>() = {1, 0, 1, 1};
  bindings_.allocate(slices)->getHandle() = {-3., -4.};
  auto *R = F_->createScatterData("scatterdata", data, indices, slices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {3, 2};
  std::vector<float> expectedValues = {1., 2., -3., -4., 5., 6.};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

TEST_P(OperatorTest, ScatterDataNDimensional) {
  CHECK_IF_ENABLED();

  // In tensor 2x4x4x3, make two updates with 2-dimensional slices by
  // 2-dimensional indices:
  // 1. By index [0, 3], set [[-1.,  -2.,  -3.]
  //                          [-4.,  -5.,  -6.]
  //                          [-7.,  -8.,  -9.]
  //                          [-10., -11., -12.]];
  //
  // 2. By index [1, 1], set [[-13., -14., -15.]
  //                          [-16., -17., -18.]
  //                          [-19., -20., -21.]
  //                          [-22., -23., -24.]];
  //
  auto *data =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4, 4, 3}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4, 3}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init(2 * 4 * 4 * 3);
  std::iota(init.begin(), init.end(), 0);
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<int64_t>() = {0, 3, 1, 1};
  std::vector<float> initUpdates;
  for (int32_t i = -1; i > -25; i--) {
    initUpdates.push_back(static_cast<float>(i));
  }
  bindings_.allocate(slices)->getHandle() = initUpdates;

  auto *R = F_->createScatterData("scatterdata", data, indices, slices);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {2, 4, 4, 3};
  std::vector<float> expectedValues = {
      0.0f,   1.0f,   2.0f,   3.0f,   4.0f,   5.0f,
      6.0f,   7.0f,   8.0f,   9.0f,   10.0f,  11.0f,

      12.0f,  13.0f,  14.0f,  15.0f,  16.0f,  17.0f,
      18.0f,  19.0f,  20.0f,  21.0f,  22.0f,  23.0f,

      24.0f,  25.0f,  26.0f,  27.0f,  28.0f,  29.0f,
      30.0f,  31.0f,  32.0f,  33.0f,  34.0f,  35.0f,

      -1.0f,  -2.0f,  -3.0f,  -4.0f,  -5.0f,  -6.0f,
      -7.0f,  -8.0f,  -9.0f,  -10.0f, -11.0f, -12.0f,

      48.0f,  49.0f,  50.0f,  51.0f,  52.0f,  53.0f,
      54.0f,  55.0f,  56.0f,  57.0f,  58.0f,  59.0f,

      -13.0f, -14.0f, -15.0f, -16.0f, -17.0f, -18.0f,
      -19.0f, -20.0f, -21.0f, -22.0f, -23.0f, -24.0f,

      72.0f,  73.0f,  74.0f,  75.0f,  76.0f,  77.0f,
      78.0f,  79.0f,  80.0f,  81.0f,  82.0f,  83.0f,

      84.0f,  85.0f,  86.0f,  87.0f,  88.0f,  89.0f,
      90.0f,  91.0f,  92.0f,  93.0f,  94.0f,  95.0f};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

TEST_P(OperatorTest, ScatterAddQuantized) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {5, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 1}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "slices", false);

  bindings_.allocate(data)->getHandle() = {1, 2, -3, -8, 5, 6, 7, 8, 9, 10};
  bindings_.allocate(indices)->getHandle<int64_t>() = {1, 3};
  bindings_.allocate(slices)->getHandle() = {3, -8, -7, 8};

  auto qParams = glow::quantization::chooseQuantizationParams({-11, 11});
  auto dataTy =
      mod_.uniqueType(ElemKind::Int8QTy, {5, 2}, qParams.scale, qParams.offset);
  auto slicesTy =
      mod_.uniqueType(ElemKind::Int8QTy, {2, 2}, qParams.scale, qParams.offset);

  auto *dataQ = F_->createQuantize("quantizeQ", data, dataTy);
  auto *slicesQ = F_->createQuantize("quantizeS", slices, slicesTy);
  auto *SA = F_->createScatterData("scatteradd", dataQ, indices, slicesQ,
                                   /*Cumulative*/ true);
  auto *DQ = F_->createDequantize("dequantize", SA, ElemKind::FloatTy);

  auto *result = F_->createSave("save", DQ);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle();

  EXPECT_NEAR(H.at({0, 0}), 1.0, 0.05);
  EXPECT_NEAR(H.at({0, 1}), 2.0, 0.05);
  EXPECT_NEAR(H.at({1, 0}), 0.0, 0.05);
  EXPECT_NEAR(H.at({1, 1}), -11.0, 0.05);
  EXPECT_NEAR(H.at({2, 0}), 5.0, 0.05);
  EXPECT_NEAR(H.at({2, 1}), 6.0, 0.05);
  EXPECT_NEAR(H.at({3, 0}), 0.0, 0.05);
  EXPECT_NEAR(H.at({3, 1}), 11.0, 0.05);
  EXPECT_NEAR(H.at({4, 0}), 9.0, 0.05);
  EXPECT_NEAR(H.at({4, 1}), 10.0, 0.05);
}

TEST_P(OperatorTest, ScatterAddNDimensionalSimple) {
  CHECK_IF_ENABLED();
  // Test that scatter addition works.
  // Data = {{1,2},{3,4},{5,6}}
  // Slices = {-3,-4}
  // Indices = {{1,0},{1,1}}
  // Result = {{1,2},{0,0},{5,6}}
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {2, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init;
  for (int32_t i = 1; i < 7; i++) {
    init.push_back(static_cast<float>(i));
  }
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<int64_t>() = {1, 0, 1, 1};
  bindings_.allocate(slices)->getHandle() = {-3., -4.};
  auto *R = F_->createScatterData("scatteradd", data, indices, slices,
                                  /*Cumulative*/ true);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {3, 2};
  std::vector<float> expectedValues = {1., 2., 0., 0., 5., 6.};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

TEST_P(OperatorTest, ScatterAddNDimensionalDuplicatingIndices) {
  CHECK_IF_ENABLED();
  // Test that scatter addition with duplicating indices works.
  // Data = {{1,2},{3,4},{5,6}}
  // Slices = {-3,-4,-3,-4}
  // Indices = {{1,0},{1,1}{1,0},{1,1}}
  // Result = {{1,2},{-3,-4},{5,6}}
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {4, 2}, "indices", false);
  auto *slices =
      mod_.createPlaceholder(ElemKind::FloatTy, {4}, "slices", false);

  // Fill tensor with consecutive data.
  std::vector<float> init;
  for (int32_t i = 1; i < 7; i++) {
    init.push_back(static_cast<float>(i));
  }
  bindings_.allocate(data)->getHandle() = init;
  bindings_.allocate(indices)->getHandle<int64_t>() = {1, 0, 1, 1, 1, 0, 1, 1};
  bindings_.allocate(slices)->getHandle() = {-3., -4., -3., -4.};
  auto *R = F_->createScatterData("scatteradd", data, indices, slices,
                                  /*Cumulative*/ true);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<dim_t> expectedDims = {3, 2};
  std::vector<float> expectedValues = {1., 2., -3., -4., 5., 6.};
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_TRUE(H.dims().vec() == expectedDims);
  for (dim_t i = 0; i < expectedValues.size(); i++) {
    EXPECT_EQ(expectedValues[i], H.raw(i));
  }
}

#define COMPARE_ARITH_FUN(_OP_NAME_)                                           \
  static FunctionTensorPair createAndInitBasic##_OP_NAME_##Test(               \
      glow::PlaceholderBindings &bindings, glow::ExecutionEngine &EE) {        \
    auto &mod = EE.getModule();                                                \
    Function *F = mod.createFunction("main");                                  \
                                                                               \
    auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);    \
    auto *B = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "B", false);    \
    bindings.allocate(A)->getHandle() = {1.0f, -1.2f, 0.5f, -1.3f};            \
    bindings.allocate(B)->getHandle() = {1.8f, -0.2f, -2.4f, 2.7f};            \
                                                                               \
    auto *add = F->create##_OP_NAME_("arith", A, B);                           \
    auto *result = F->createSave("save", add);                                 \
    auto *resultTensor = bindings.allocate(result->getPlaceholder());          \
                                                                               \
    return std::make_pair(F, resultTensor);                                    \
  }
COMPARE_ARITH_FUN(Add)
COMPARE_ARITH_FUN(Sub)
COMPARE_ARITH_FUN(Mul)
COMPARE_ARITH_FUN(Div)
COMPARE_ARITH_FUN(FloorDiv)
COMPARE_ARITH_FUN(Max)
COMPARE_ARITH_FUN(Min)
#undef COMPARE_ARITH_FUN

#define COMPARE_ARITH_FLOAT_VS_INT8(_OP_NAME_)                                 \
  TEST_P(OperatorStatelessTest, Basic##_OP_NAME_##NetFloatVsInt8) {            \
    CHECK_IF_ENABLED();                                                        \
    compareAgainstInterpreter(                                                 \
        getBackendName(), createAndInitBasic##_OP_NAME_##Test,                 \
        ElemKind::FloatTy, ElemKind::Int8QTy, 0.035f, parCloneCountOpt);       \
  }
COMPARE_ARITH_FLOAT_VS_INT8(Add)
COMPARE_ARITH_FLOAT_VS_INT8(Sub)
COMPARE_ARITH_FLOAT_VS_INT8(Mul)
COMPARE_ARITH_FLOAT_VS_INT8(Div)
COMPARE_ARITH_FLOAT_VS_INT8(Max)
COMPARE_ARITH_FLOAT_VS_INT8(Min)
#undef COMPARE_ARITH_FLOAT_VS_INT8

#define COMPARE_ARITH_FLOAT_VS_FLOAT16(_OP_NAME_)                              \
  TEST_P(OperatorStatelessTest, Basic##_OP_NAME_##NetFloatVsFloat16) {         \
    CHECK_IF_ENABLED();                                                        \
    compareAgainstInterpreter(                                                 \
        getBackendName(), createAndInitBasic##_OP_NAME_##Test,                 \
        ElemKind::FloatTy, ElemKind::Float16Ty, 0.01f, parCloneCountOpt);      \
  }

#define COMPARE_ARITH_FLOAT_VS_BFLOAT16(_OP_NAME_)                             \
  TEST_P(OperatorStatelessTest, Basic##_OP_NAME_##NetFloatVsBFloat16) {        \
    CHECK_IF_ENABLED();                                                        \
    compareAgainstInterpreter(                                                 \
        getBackendName(), createAndInitBasic##_OP_NAME_##Test,                 \
        ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.01f, parCloneCountOpt);     \
  }
COMPARE_ARITH_FLOAT_VS_FLOAT16(Add)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Sub)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Mul)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Div)
COMPARE_ARITH_FLOAT_VS_FLOAT16(FloorDiv)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Max)
COMPARE_ARITH_FLOAT_VS_FLOAT16(Min)

COMPARE_ARITH_FLOAT_VS_BFLOAT16(Add)
COMPARE_ARITH_FLOAT_VS_BFLOAT16(Sub)
COMPARE_ARITH_FLOAT_VS_BFLOAT16(Mul)
COMPARE_ARITH_FLOAT_VS_BFLOAT16(Div)
COMPARE_ARITH_FLOAT_VS_BFLOAT16(FloorDiv)
COMPARE_ARITH_FLOAT_VS_BFLOAT16(Max)
COMPARE_ARITH_FLOAT_VS_BFLOAT16(Min)
#undef COMPARE_ARITH_FLOAT_VS_FLOAT16
#undef COMPARE_ARITH_FLOAT_VS_BFLOAT16

#define ARITH_FUN_IMPL(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)         \
  template <typename DataType>                                                 \
  static void testArithmetic##_OP_NAME_##Impl(                                 \
      glow::PlaceholderBindings &bindings, glow::Module &mod,                  \
      glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy) {            \
    std::vector<DataType> data1 = {3, 17, 7, 23};                              \
    std::vector<DataType> data2 = {13, 5, 19, 11};                             \
    auto *A = mod.createPlaceholder(DTy, {1, 4}, "A", false);                  \
    auto *B = mod.createPlaceholder(DTy, {1, 4}, "B", false);                  \
    bindings.allocate(A)->getHandle<DataType>() = data1;                       \
    bindings.allocate(B)->getHandle<DataType>() = data2;                       \
                                                                               \
    auto *add = F->create##_OP_NAME_("arith", A, B);                           \
    auto *result = F->createSave("save", add);                                 \
    auto *resultTensor = bindings.allocate(result->getPlaceholder());          \
                                                                               \
    EE.compile(CompilationMode::Infer);                                        \
    EE.run(bindings);                                                          \
    std::vector<DataType> reference;                                           \
    assert(data1.size() == data2.size() && "Size mismatch!");                  \
    for (size_t i = 0; i < data1.size(); i++) {                                \
      reference.push_back(                                                     \
          _REFERENCE_FUNCTION_<DataType> _PARENTHESES_(data1[i], data2[i]));   \
    }                                                                          \
    auto RH = resultTensor->getHandle<DataType>();                             \
    EXPECT_EQ(reference.size(), RH.size());                                    \
    for (size_t i = 0; i < reference.size(); i++) {                            \
      EXPECT_EQ(reference[i], RH.raw(i));                                      \
    }                                                                          \
  }

#define ARITH_FUNC_TEST_TYPED(_OP_NAME_, _DATA_TYPE_, _ELEM_KIND_)             \
  TEST_P(OperatorTest, Arith##_OP_NAME_##_##_DATA_TYPE_) {                     \
    CHECK_IF_ENABLED();                                                        \
    testArithmetic##_OP_NAME_##Impl<_DATA_TYPE_>(bindings_, mod_, F_, EE_,     \
                                                 _ELEM_KIND_);                 \
  }

#define ARITH_FUNC_TEST(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)        \
  ARITH_FUN_IMPL(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)               \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, int32_t, ElemKind::Int32ITy)                \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, int64_t, ElemKind::Int64ITy)                \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, float, ElemKind::FloatTy)                   \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, float16_t, ElemKind::Float16Ty)             \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, bfloat16_t, ElemKind::BFloat16Ty)

template <typename DataType>
static DataType floorDivide(DataType a, DataType b) {
  return std::floor(static_cast<float>(a / b));
}

ARITH_FUNC_TEST(Add, std::plus, ())
ARITH_FUNC_TEST(Sub, std::minus, ())
ARITH_FUNC_TEST(Mul, std::multiplies, ())
ARITH_FUNC_TEST(Max, std::max, )
ARITH_FUNC_TEST(Min, std::min, )

#define ARITH_FUNC_TEST_FLOAT(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)  \
  ARITH_FUN_IMPL(_OP_NAME_, _REFERENCE_FUNCTION_, _PARENTHESES_)               \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, float, ElemKind::FloatTy)                   \
  ARITH_FUNC_TEST_TYPED(_OP_NAME_, float16_t, ElemKind::Float16Ty)

ARITH_FUNC_TEST_FLOAT(FloorDiv, floorDivide, )

#undef ARITH_FUN_IMPL
#undef ARITH_FUNC_TEST_TYPED
#undef ARITH_FUNC_TEST
#undef ARITH_FUNC_TEST_FLOAT

TEST_P(OperatorTest, IntMatMul) {
  CHECK_IF_ENABLED();

  // The scaling factor 1.4x was carefully selected to make sure we don't
  // overflow or underflow the calculation.
  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.60, 4);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, -2);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.075, 2);

  auto *lhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "lhs", false);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", false);

  bindings_.allocate(lhs)->getHandle() = {
      1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, 9.0,
  };

  bindings_.allocate(rhs)->getHandle() = {
      0.1f, -0.2f, 0.3f, 9.0f, -8.0f, 7.0f, 6.0f, 5.0f, 9.0f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createMatMul("matmul.q", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq, ElemKind::FloatTy);

  auto *result = F_->createSave("save", rq);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  /*
   Test the following matrix multiplication:
   A = [[1.0, 2.0, 3.0], [4.0, 5.0, -5.0], [-4.0, -3.0, 9.0]]
   B = [[0.1, -0.2, 0.3], [9.0, -8.0, 7.0], [6.0, 5.0, 9.0]]
   A x B = [36.1,  -1.2,  41.3], [15.4, -65.8, -8.8], [26.6, 69.8,  58.8]]
   */

  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_NEAR(H.at({0, 0}), 36.1, 1.0);
  EXPECT_NEAR(H.at({0, 1}), -1.2, 1.0);
  EXPECT_NEAR(H.at({0, 2}), 41.3, 1.0);
  EXPECT_NEAR(H.at({1, 0}), 15.4, 1.0);
  EXPECT_NEAR(H.at({1, 1}), -65.8, 1.0);
  EXPECT_NEAR(H.at({1, 2}), -8.8, 1.0);
  EXPECT_NEAR(H.at({2, 0}), 26.6, 1.0);
  EXPECT_NEAR(H.at({2, 1}), 69.8, 1.0);
  EXPECT_NEAR(H.at({2, 2}), 58.8, 1.0);
}

TEST_P(OperatorTest, IntBatchedArith) {
  CHECK_IF_ENABLED();

  TypeRef resTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.10, 1.0);
  TypeRef lhsTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3}, 0.11, 4.0);
  TypeRef rhsTy = mod_.uniqueType(ElemKind::Int8QTy, {3, 3}, 0.14, -2.0);

  auto *lhs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3}, "lhs", false);
  bindings_.allocate(lhs);
  auto *rhs = mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "rhs", false);
  bindings_.allocate(rhs);

  bindings_.get(lhs)->getHandle() = {
      8.7f, 6.5f, 4.3f, 2.1f, 1.0f, -5.1f, -4.0f, -12.0f, 0.2f,
  };

  bindings_.get(rhs)->getHandle() = {
      -9.1f, -0.4f, 1.3f, 2.2f, -8.1f, 7.6f, -6.4f, 10.0f, 9.1f,
  };

  auto *lhsq = F_->createQuantize("lhs.q", lhs, lhsTy);
  auto *rhsq = F_->createQuantize("rhs.q", rhs, rhsTy);

  auto *matmulq = F_->createBatchedAdd("add", resTy, lhsq, rhsq);

  auto *rq = F_->createDequantize("dequant", matmulq, ElemKind::FloatTy);

  auto *result = F_->createSave("save", rq);
  bindings_.allocate(result->getPlaceholder());
  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  // A = [8.7, 6.5, 4.3, 2.1, 1.0, -5.1, -4.0, -12.0, 0.2]
  // B = [-9.1, -0.4, 1.3, 2.2, -8.1, 7.6, -6.4, 10.0, 9.1]
  // A + B = [-0.4, 6.1, 5.6, 4.3, -7.1, 2.5, -10.4, -2. , 9.3]
  auto H = bindings_.get(result->getPlaceholder())->getHandle();
  constexpr float allowedError = 0.105;
  EXPECT_NEAR(H.at({0, 0, 0}), -0.4, allowedError);
  EXPECT_NEAR(H.at({0, 0, 1}), 6.1, allowedError);
  EXPECT_NEAR(H.at({0, 0, 2}), 5.6, allowedError);
  EXPECT_NEAR(H.at({0, 1, 0}), 4.3, allowedError);
  EXPECT_NEAR(H.at({0, 1, 1}), -7.1, allowedError);
  EXPECT_NEAR(H.at({0, 1, 2}), 2.5, allowedError);
  EXPECT_NEAR(H.at({0, 2, 0}), -10.4, allowedError);
  EXPECT_NEAR(H.at({0, 2, 1}), -2, allowedError);
  EXPECT_NEAR(H.at({0, 2, 2}), 9.3, allowedError);
}

TEST_P(OperatorTest, convTest) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  FH = {0, 0, 0, 1, 1, 1, 0, 0, 0};

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());

  Tensor expected(outTy);
  expected.getHandle() = {2, 3, 2, 2, 3, 2, 2, 3, 2};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, convTest_Float16) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle<float16_t>();
  IH = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  auto filter = mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1},
                                       "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle<float16_t>();
  FH = {0.25, 0.5, 0.25, 1, 1, 1, 0.25, 0.5, 0.25};

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::Float16Ty, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();

  Tensor expected(outTy);
  auto expectedH = expected.getHandle<float16_t>();
  expectedH = {3.375, 5.102, 3.676, 5.051, 7.5, 5.449, 4.574, 6.898, 4.875};

  for (dim_t x = 0; x < 3; x++) {
    for (dim_t y = 0; y < 3; y++) {
      EXPECT_NEAR(result.at({0, x, y, 0}), expectedH.at({0, x, y, 0}), 0.001);
    }
  }
}

TEST_P(OperatorTest, convTest_BFloat16) {
  CHECK_IF_ENABLED();
  auto *input = mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle<bfloat16_t>();
  IH = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};

  auto filter = mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1},
                                       "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle<bfloat16_t>();
  FH = {0.25, 0.5, 0.25, 1, 1, 1, 0.25, 0.5, 0.25};

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::BFloat16Ty, {1, 3, 3, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 1, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<bfloat16_t>();

  Tensor expected(outTy);
  auto expectedH = expected.getHandle<bfloat16_t>();
  expectedH = {3.375, 5.102, 3.676, 5.051, 7.5, 5.449, 4.574, 6.898, 4.875};

  for (dim_t x = 0; x < 3; x++) {
    for (dim_t y = 0; y < 3; y++) {
      EXPECT_NEAR(result.at({0, x, y, 0}), expectedH.at({0, x, y, 0}), 0.05);
    }
  }
}

template <size_t convDepth>
static FunctionTensorPair
createAndInitConvDepthTest(glow::PlaceholderBindings &bindings,
                           glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *conv = F->createConv(bindings, "conv", input, convDepth, 5, 1, 0, 1);
  auto *bias = llvm::cast<Placeholder>(conv->getBias().getNode());

  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.get(bias)->getHandle().randomize(-2.0, 2.0, mod.getPRNG());

  auto *res = F->createSave("save", conv);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, Int8ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.045f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Int16ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::Int16QTy, 0.03f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Int8ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.03f,
                            parCloneCountOpt);
}
TEST_P(OperatorStatelessTest, Int16ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::Int16QTy, 0.03f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, FP16ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.015f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, BFloat16ConvolutionDepth10) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<10>,
                            ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.015f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, FP16ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.015f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, BFloat16ConvolutionDepth8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitConvDepthTest<8>,
                            ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.015f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int8_BiasInt8) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.03f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int8QTy);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int8_BiasInt32) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.03f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int16_BiasInt16) {
  ENABLED_BACKENDS("Interpreter");
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0003f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int16QTy);
}

TEST_P(OperatorStatelessTest, ConvolutionDepth10_Int16_BiasInt32) {
  ENABLED_BACKENDS("Interpreter");
  compareAgainstInterpreter(
      getBackendName(), createAndInitConvDepthTest<10>, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0003f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

static FunctionTensorPair
createAndInitBasicConcatTest(glow::PlaceholderBindings &bindings,
                             glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {3, 3}, "A", false);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {2, 3}, "B", false);
  bindings.allocate(A)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(B)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());

  auto *C = F->createConcat("concat", {A, B}, 0);
  auto *res = F->createSave("save", C);
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {A, B, res->getPlaceholder()});

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, IntConcat) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicConcatTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.05f,
                            parCloneCountOpt);
}

TEST_P(OperatorTest, FCWithFlatten) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 1, 3}, "input", false);
  Constant *weights = mod_.createConstant(ElemKind::FloatTy, {3, 4}, "weights");
  Constant *bias = mod_.createConstant(ElemKind::FloatTy, {4}, "bias");

  bindings_.allocate(input)->getHandle() = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  weights->getPayloadMutable().getHandle() = {1.0f, 4.0f, 7.0f, 10.0f, //
                                              2.0f, 5.0f, 8.0f, 11.0f, //
                                              3.0f, 6.0f, 9.0f, 12.0f};
  bias->getPayloadMutable().getHandle() = {0.1f, 0.2f, 0.3f, 0.4f};

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *S = F_->createSave("save", FC);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDimensions = {2, 4};
  std::vector<float> expectedValues = {14.1f, 32.2f, 50.3f,  68.4f,
                                       32.1f, 77.2f, 122.3f, 167.4f};
  EXPECT_TRUE(result.dims().vec() == expectedDimensions);
  for (size_t i = 0; i < 2 * 4; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

TEST_P(OperatorTest, TestFP32Accumulator) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3}, "input", false);
  Constant *weights =
      mod_.createConstant(ElemKind::Float16Ty, {3, 2}, "weights");
  Constant *bias = mod_.createConstant(ElemKind::Float16Ty, {2}, "bias");

  /* 9.7e-4 is smaller than what the mantissa can represent
    when the initial value is 1, but 2 * 9.7e-4 is exactly
    the smallest number that can be represented after 1
    In Fp16 accumulation, we will be losing the update leading to 1,
    in fp32, we get a value slightly larger than 1.
  */
  bindings_.allocate(input)->getHandle<float16_t>() = {1.0f, 9.7e-4, 9.7e-4f};
  weights->getPayloadMutable().getHandle<float16_t>() = {1.0f, 1.0f, 0.5f,
                                                         1.0f, 0.5f, 1.0f};
  bias->getPayloadMutable().getHandle<float16_t>() = {0.0f, 0.0f};

  auto *FC = F_->createFullyConnected("fc", input, weights, bias);
  auto *S = F_->createSave("save", FC);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  std::vector<dim_t> expectedDimensions = {1, 2};

  EXPECT_TRUE(result.dims().vec() == expectedDimensions);
  float finalResult = result.raw(0);
  if (finalResult == 1.0) {
    llvm::outs() << "fp16 accumulator\n";
  } else if (fabs(finalResult - 1.00098) < 1e-3) {
    llvm::outs() << "fp32 accumulator\n";
  } else {
    // Unhandled case
    FAIL() << "unknown " << finalResult;
  }
  llvm::outs().flush();
}

static FunctionTensorPair
createAndInitBasicFCTest(glow::PlaceholderBindings &bindings,
                         glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *fc = F->createFullyConnected(bindings, "FC", input, 30);

  auto *weights = llvm::cast<Placeholder>(fc->getWeights());
  auto *bias = llvm::cast<Placeholder>(fc->getBias());

  bindings.allocate(input)->getHandle().randomize(-0.5, 0.5, mod.getPRNG());
  bindings.get(bias)->getHandle().randomize(0, 0.00001, mod.getPRNG());
  bindings.get(weights)->getHandle().randomize(-0.7, 0.7, mod.getPRNG());

  auto *res = F->createSave("save", fc);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, IntFC) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicFCTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.05f,
                            parCloneCountOpt);
}

/// Test FC with Float16.
TEST_P(OperatorStatelessTest, FC_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicFCTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.02f,
                            parCloneCountOpt);
}

/// Test FC with BFloat16.
TEST_P(OperatorStatelessTest, FC_BFloat16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicFCTest,
                            ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.02f,
                            parCloneCountOpt);
}

/// Test Int8 FullyConnected with Int8 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int8_BiasInt8) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.05f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int8QTy);
}

/// Test Int8 FullyConnected with Int32 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int8_BiasInt32) {
  ENABLED_BACKENDS("Interpreter", "CPU", "NNPI");
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.05f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

/// Test Int16 FullyConnected with Int16 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int16_BiasInt16) {
  ENABLED_BACKENDS("Interpreter");
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0005f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int16QTy);
}

/// Test Int16 FullyConnected with Int32 bias.
TEST_P(OperatorStatelessTest, FullyConnected_Int16_BiasInt32) {
  ENABLED_BACKENDS("Interpreter");
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicFCTest, ElemKind::FloatTy,
      ElemKind::Int16QTy, 0.0005f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, EntropyLossTest) {
  CHECK_IF_ENABLED();

  auto *P = mod_.createPlaceholder(ElemKind::FloatTy, {2, 3}, "P", false);
  auto *Y = mod_.createPlaceholder(ElemKind::Int64ITy, {2}, "Y", false);

  bindings_.allocate(P)->getHandle() = {0.2f, 0.5f, 0.3f, 0.4f, 0.3f, 0.3f};
  bindings_.allocate(Y)->getHandle<int64_t>() = {1, 2};
  auto *ceLoss = F_->createCrossEntropyLoss("CELoss", P, Y);
  auto *L = F_->createSave("save", ceLoss);
  bindings_.allocate(L->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto R = bindings_.get(L->getPlaceholder())->getHandle();
  EXPECT_NEAR(R.at({0}), -log(0.5) - log(0.3), 0.1);
}

/// Check that the max operator works properly with FP16.
TEST_P(OperatorTest, FP16Max) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "A", false);
  bindings_.allocate(inputA)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "B", false);
  bindings_.allocate(inputB)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *Max = F_->createMax("max", inputA, inputB);
  auto *S = F_->createSave("save", Max);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleA = bindings_.get(inputA)->getHandle<float16_t>();
  auto handleB = bindings_.get(inputB)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), std::max(handleA.raw(idx), handleB.raw(idx)));
  }
}

/// Check that the max operator works properly with FP16.
TEST_P(OperatorTest, BFloat16Max) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *inputA =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1}, "A", false);
  bindings_.allocate(inputA)->getHandle<bfloat16_t>().randomize(-3.0, 3.0,
                                                                PRNG);
  auto *inputB =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1}, "B", false);
  bindings_.allocate(inputB)->getHandle<bfloat16_t>().randomize(-3.0, 3.0,
                                                                PRNG);
  auto *Max = F_->createMax("max", inputA, inputB);
  auto *S = F_->createSave("save", Max);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<bfloat16_t>();
  auto handleA = bindings_.get(inputA)->getHandle<bfloat16_t>();
  auto handleB = bindings_.get(inputB)->getHandle<bfloat16_t>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx), std::max(handleA.raw(idx), handleB.raw(idx)));
  }
}

/// Helper to test Broadcast Max/Min using \p DTy and \p NTy
template <typename DataType, typename NodeType>
static void testBroadcastMaxMin(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind DTy) {

  auto *inputA = mod.createPlaceholder(DTy, {1, 3, 3, 1}, "A", false);
  bindings.allocate(inputA)->getHandle<DataType>().randomize(-3.0, 3.0,
                                                             mod.getPRNG());
  auto *inputB = mod.createPlaceholder(DTy, {1, 3, 3, 1}, "B", false);
  bindings.allocate(inputB)->getHandle<DataType>().randomize(-3.0, 3.0,
                                                             mod.getPRNG());

  Node *maxorMinOp = F->createNodeWithBroadcast<NodeType>(
      "maxormin", -1 /*axis */, inputA, inputB);

  auto *S = F->createSave("save", maxorMinOp);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  ASSERT_TRUE(F->verify(&EE.getBackend()))
      << "Function must pass verification.";

  auto result = bindings.get(S->getPlaceholder())->getHandle<DataType>();
  auto handleA = bindings.get(inputA)->getHandle<DataType>();
  auto handleB = bindings.get(inputB)->getHandle<DataType>();
  ASSERT_EQ(result.size(), handleA.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    if (std::is_same<NodeType, MaxNode>::value) {
      EXPECT_EQ(result.raw(idx), std::max(handleA.raw(idx), handleB.raw(idx)));
    } else {
      EXPECT_EQ(result.raw(idx), std::min(handleA.raw(idx), handleB.raw(idx)));
    }
  }
}

TEST_P(OperatorTest, BroadCastMax) {
  CHECK_IF_ENABLED();
  testBroadcastMaxMin<int64_t, MaxNode>(bindings_, mod_, F_, EE_,
                                        ElemKind::Int64ITy);
}

TEST_P(OperatorTest, BroadCastMin) {
  CHECK_IF_ENABLED();
  testBroadcastMaxMin<int64_t, MinNode>(bindings_, mod_, F_, EE_,
                                        ElemKind::Int64ITy);
}

TEST_P(OperatorTest, RescaleNode) {
  CHECK_IF_ENABLED();

  // Check the outputs of the RescaleQuantized operation.
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {4, 10}, 0.4, -3,
                                       "input", false);
  bindings_.allocate(input)->init(Tensor::InitKind::Broadcast, 40,
                                  mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.7, 5);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.3, -4);
  auto resTy = mod_.uniqueType(ElemKind::Int8QTy, {4, 10}, 0.4, -4);

  // Test a sequence of rescale operations that the optimizer may try to
  // optimize at some point.
  auto *X = F_->createRescaleQuantized("R1", input, T1);
  auto *Y = F_->createRescaleQuantized("R2", X, T2);
  auto *Z = F_->createRescaleQuantized("R3", Y, resTy);

  auto *output = F_->createSave("save", Z);
  bindings_.allocate(output->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto RI = bindings_.get(input)->getHandle<int8_t>();
  auto RO = bindings_.get(output->getPlaceholder())->getHandle<int8_t>();

  EXPECT_EQ(RI.raw(0), 40);
  EXPECT_NEAR(RO.raw(0), 40, 1);
}

TEST_P(OperatorTest, QuantizedArithmeticRescaled) {
  CHECK_IF_ENABLED();

  const dim_t len = 100;

  // In this test we check the correctness of the quantized Max, Min, Add,
  // Sub, Mul, and Div nodes as well as how they interact with the rescaling
  // node.
  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "A", false);
  auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "B", false);
  auto *C = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "C", false);

  auto AH = bindings_.allocate(A)->getHandle();
  auto BH = bindings_.allocate(B)->getHandle();
  auto CH = bindings_.allocate(C)->getHandle();

  AH.randomize(-10, 10, mod_.getPRNG());
  BH.randomize(-10, 10, mod_.getPRNG());
  // Below, randomize between 1 and 10 to avoid division by 0 later.
  CH.randomize(1, 10, mod_.getPRNG());

  auto TA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.2, 0);
  auto TB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 0);
  auto TC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, 0);

  auto TI1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TI2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 0);
  auto TI3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TI4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TI5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);
  auto TI6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.7, 0);

  auto TO1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 0);
  auto TO3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);
  auto TO4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 0);
  auto TO5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto TO6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, 0);

  // Quantize input vars and apply max/min/add/sub/mul/div quantized.
  auto *QA = F_->createQuantize("QA", A, TA);
  auto *QB = F_->createQuantize("QB", B, TB);
  auto *QC = F_->createQuantize("QC", C, TC);

  Node *max = F_->createMax("max", TI1, QA, QB);
  Node *min = F_->createMin("min", TI2, QA, QB);
  Node *add = F_->createAdd("add", TI3, QA, QB);
  Node *sub = F_->createSub("sub", TI4, QA, QB);
  Node *mul = F_->createMul("mul", TI5, QA, QB);
  Node *div = F_->createDiv("div", TI6, QB, QC);

  // Rescale quantized results.
  max = F_->createRescaleQuantized("rescaleMax", max, TO1);
  min = F_->createRescaleQuantized("rescaleMin", min, TO2);
  add = F_->createRescaleQuantized("rescaleAdd", add, TO3);
  sub = F_->createRescaleQuantized("rescaleSub", sub, TO4);
  mul = F_->createRescaleQuantized("rescaleMul", mul, TO5);
  div = F_->createRescaleQuantized("rescaleDiv", div, TO6);

  // Dequantize results back to floating-point.
  max = F_->createDequantize("maxDQ", max, ElemKind::FloatTy);
  min = F_->createDequantize("minDQ", min, ElemKind::FloatTy);
  add = F_->createDequantize("addDQ", add, ElemKind::FloatTy);
  sub = F_->createDequantize("subDQ", sub, ElemKind::FloatTy);
  mul = F_->createDequantize("mulDQ", mul, ElemKind::FloatTy);
  div = F_->createDequantize("divDQ", div, ElemKind::FloatTy);

  // Save results of the operations.
  auto *O1 = F_->createSave("saveMax", max);
  auto *O2 = F_->createSave("saveMin", min);
  auto *O3 = F_->createSave("saveAdd", add);
  auto *O4 = F_->createSave("saveSub", sub);
  auto *O5 = F_->createSave("saveMul", mul);
  auto *O6 = F_->createSave("saveDiv", div);

  bindings_.allocate(O1->getPlaceholder());
  bindings_.allocate(O2->getPlaceholder());
  bindings_.allocate(O3->getPlaceholder());
  bindings_.allocate(O4->getPlaceholder());
  bindings_.allocate(O5->getPlaceholder());
  bindings_.allocate(O6->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < len; i++) {
    auto max = std::max(AH.at({i}), BH.at({i}));
    auto min = std::min(AH.at({i}), BH.at({i}));
    auto add = AH.at({i}) + BH.at({i});
    auto sub = AH.at({i}) - BH.at({i});
    auto mul = AH.at({i}) * BH.at({i});
    auto div = BH.at({i}) / CH.at({i});

    // We generate numbers up to 110, so a difference of 2 (~2%) is
    // reasonable.
    EXPECT_NEAR(max, bindings_.get(O1->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(min, bindings_.get(O2->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(add, bindings_.get(O3->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(sub, bindings_.get(O4->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(mul, bindings_.get(O5->getPlaceholder())->getHandle().at({i}),
                2.0);
    EXPECT_NEAR(div, bindings_.get(O6->getPlaceholder())->getHandle().at({i}),
                2.0);
  }
}

static FunctionTensorPair
createAndInitTransposeNet(glow::PlaceholderBindings &bindings,
                          glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {2, 3}, "A", false);
  bindings.allocate(A)->getHandle() = {1, 1.2f, 0.5f, 1.3f, 2.7f, 3.1f};
  auto *tr = F->createTranspose("Tr", A, {1, 0});
  auto *result = F->createSave("Ret", tr);
  auto *resultTensor = bindings.allocate(result->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

TEST_P(OperatorStatelessTest, QuantizedTranspose) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitTransposeNet,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.0045f,
                            parCloneCountOpt);
}

TEST_P(OperatorTest, QuantizedArithmeticUnrescaled) {
  CHECK_IF_ENABLED();

  const dim_t len = 1000;

  // In this test we check the correctness of the quantized Max, Min, Add,
  // Sub, Mul, and Div operations.
  auto TQA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, -1);
  auto TQB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  // For TQC, set offset to -11 to avoid division by 0 later.
  auto TQC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, -11);
  auto TO1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.4, 3);
  auto TO2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 2);
  auto TO3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.7, 5);
  auto TO4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -7);
  auto TO5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, 3);
  auto TO6 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, -2);

  auto *QA = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQA->getScale(),
                                    TQA->getOffset(), "QA", false);
  auto *QB = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQB->getScale(),
                                    TQB->getOffset(), "QB", false);
  auto *QC = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQC->getScale(),
                                    TQC->getOffset(), "QC", false);

  bindings_.allocate(QA)->getHandle<int8_t>().randomize(-10, 10,
                                                        mod_.getPRNG());
  bindings_.allocate(QB)->getHandle<int8_t>().randomize(-10, 10,
                                                        mod_.getPRNG());
  bindings_.allocate(QC)->getHandle<int8_t>().randomize(-10, 10,
                                                        mod_.getPRNG());

  // Apply max/min/add/sub/mul/div quantized.
  Node *max = F_->createMax("max", TO1, QA, QB);
  Node *min = F_->createMin("min", TO2, QA, QB);
  Node *add = F_->createAdd("add", TO3, QA, QB);
  Node *sub = F_->createSub("sub", TO4, QA, QB);
  Node *mul = F_->createMul("mul", TO5, QA, QB);
  Node *div = F_->createDiv("div", TO6, QB, QC);

  // Save results of the operations.
  auto *O1 = F_->createSave("saveMax", max);
  auto *O2 = F_->createSave("saveMin", min);
  auto *O3 = F_->createSave("saveAdd", add);
  auto *O4 = F_->createSave("saveSub", sub);
  auto *O5 = F_->createSave("saveMul", mul);
  auto *O6 = F_->createSave("saveDiv", div);

  bindings_.allocate(O1->getPlaceholder());
  bindings_.allocate(O2->getPlaceholder());
  bindings_.allocate(O3->getPlaceholder());
  bindings_.allocate(O4->getPlaceholder());
  bindings_.allocate(O5->getPlaceholder());
  bindings_.allocate(O6->getPlaceholder());

  auto QAH = bindings_.get(QA)->getHandle<int8_t>();
  auto QBH = bindings_.get(QB)->getHandle<int8_t>();
  auto QCH = bindings_.get(QC)->getHandle<int8_t>();
  auto O1H = bindings_.get(O1->getPlaceholder())->getHandle<int8_t>();
  auto O2H = bindings_.get(O2->getPlaceholder())->getHandle<int8_t>();
  auto O3H = bindings_.get(O3->getPlaceholder())->getHandle<int8_t>();
  auto O4H = bindings_.get(O4->getPlaceholder())->getHandle<int8_t>();
  auto O5H = bindings_.get(O5->getPlaceholder())->getHandle<int8_t>();
  auto O6H = bindings_.get(O6->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < len; i++) {
    float a = TQA->getScale() * (QAH.at({i}) - TQA->getOffset());
    float b = TQB->getScale() * (QBH.at({i}) - TQB->getOffset());
    float c = TQC->getScale() * (QCH.at({i}) - TQC->getOffset());
    float max = std::max(a, b) / TO1->getScale() + TO1->getOffset();
    float min = std::min(a, b) / TO2->getScale() + TO2->getOffset();
    float add = (a + b) / TO3->getScale() + TO3->getOffset();
    float sub = (a - b) / TO4->getScale() + TO4->getOffset();
    float mul = (a * b) / TO5->getScale() + TO5->getOffset();
    float div = (b / c) / TO6->getScale() + TO6->getOffset();

    EXPECT_NEAR(std::round(max), O1H.at({i}), 1.0);
    EXPECT_NEAR(std::round(min), O2H.at({i}), 1.0);
    EXPECT_NEAR(std::round(add), O3H.at({i}), 1.0);
    EXPECT_NEAR(std::round(sub), O4H.at({i}), 1.0);
    EXPECT_NEAR(std::round(mul), O5H.at({i}), 1.0);
    EXPECT_NEAR(std::round(div), O6H.at({i}), 1.0);
  }
}

TEST_P(OperatorTest, QuantizedCmpLTEAndSelect) {
  CHECK_IF_ENABLED();

  // In this test we check the correctness of the quantized
  // less-than-or-equal-to comparison operator.
  const dim_t len = 1000;
  auto TQA = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.1, -3);
  auto TQB = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 5);
  auto TQC = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.8, 3);
  auto TQD = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.2, -4);
  auto OT = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.5, -2);

  auto *QA = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQA->getScale(),
                                    TQA->getOffset(), "QA", false);
  auto *QB = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQB->getScale(),
                                    TQB->getOffset(), "QB", false);
  auto *QC = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQC->getScale(),
                                    TQC->getOffset(), "QC", false);
  auto *QD = mod_.createPlaceholder(ElemKind::Int8QTy, {len}, TQD->getScale(),
                                    TQD->getOffset(), "QD", false);

  auto QAH = bindings_.allocate(QA)->getHandle<int8_t>();
  auto QBH = bindings_.allocate(QB)->getHandle<int8_t>();
  auto QCH = bindings_.allocate(QC)->getHandle<int8_t>();
  auto QDH = bindings_.allocate(QD)->getHandle<int8_t>();

  QAH.randomize(-128, 127, mod_.getPRNG());
  QBH.randomize(-128, 127, mod_.getPRNG());
  QCH.randomize(-128, 127, mod_.getPRNG());
  QDH.randomize(-128, 127, mod_.getPRNG());

  // Apply comparison and selection quantized.
  Node *cmpLTE = F_->createCmpLTE("cmpLTE", QA, QB);
  Node *select = F_->createSelect("select", OT, cmpLTE, QC, QD);

  // Save result of the operation.
  auto *out = F_->createSave("save", select);
  auto OH = bindings_.allocate(out->getPlaceholder())->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  int count_strict = 0;
  int count = 0;
  for (dim_t i = 0; i < len; i++) {
    float a = TQA->getScale() * (QAH.at({i}) - TQA->getOffset());
    float b = TQB->getScale() * (QBH.at({i}) - TQB->getOffset());
    float c = TQC->getScale() * (QCH.at({i}) - TQC->getOffset());
    float d = TQD->getScale() * (QDH.at({i}) - TQD->getOffset());
    float tmp = (a <= b) ? c : d;
    int32_t q = std::round(tmp / 1.5 - 2);
    int8_t select = quantization::clip<int32_t, int8_t>(q);

    if (OH.at({i}) != select) {
      count_strict++;
      if (std::abs(OH.at({i}) - select) > 1) {
        count++;
      }
    }
  }
  // Require that the number of off-by-1 errors be at most 0.6%.
  EXPECT_LE(count_strict, 6);
  EXPECT_LE(count, 4);
}

TEST_P(OperatorTest, TestQuantizedRescaleSequence) {
  CHECK_IF_ENABLED();

  const dim_t len = 100;

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {len}, "A", false);

  auto AH = bindings_.allocate(A)->getHandle();

  // Notice that the range below is the an approximation of the scale factors
  // in T3 and T4. If we increase the size of the range we may start losing
  // some values.
  AH.randomize(-12, 12, mod_.getPRNG());

  auto T1 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 1.0, 0);
  auto T2 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.9, 2);
  auto T3 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, -3);
  auto T4 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.1, 7);
  auto T5 = mod_.uniqueType(ElemKind::Int8QTy, {len}, 0.3, -3);

  Node *R = F_->createQuantize("R", A, T1);
  // Check that a sequence of type conversions does not change the result.
  R = F_->createRescaleQuantized("R", R, T1);
  R = F_->createRescaleQuantized("R", R, T2);
  R = F_->createRescaleQuantized("R", R, T3);
  // Check that adding the quantized zero does not change the result.
  auto *G = F_->createSplat("splatZero", T3, 0.0);
  R = F_->createAdd("addZero", G, R);
  R = F_->createRescaleQuantized("R", R, T4);
  R = F_->createRescaleQuantized("R", R, T5);
  R = F_->createRescaleQuantized("R", R, T1);
  auto *DQ = F_->createDequantize("DQ", R, ElemKind::FloatTy);

  // Test a sequence of rescale operations t
  auto *result = F_->createSave("save", DQ);
  auto OH = bindings_.allocate(result->getPlaceholder())->getHandle();
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < len; i++) {
    EXPECT_NEAR(AH.at({i}), OH.at({i}), 1.0);
  }
}

/// Helper to test concatVectors using \p DTy.
template <typename DataType>
static void testConcatVectors(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("concatVectors");

  auto *V1 =
      createPlaceholderConditionallyQuantized(mod, DTy, {10}, "V1", false);
  auto *V2 =
      createPlaceholderConditionallyQuantized(mod, DTy, {20}, "V2", false);
  auto *V3 =
      createPlaceholderConditionallyQuantized(mod, DTy, {30}, "V3", false);

  bindings.allocate(V1);
  bindings.allocate(V2);
  bindings.allocate(V3);

  Node *L = F->createConcat("concat", {V1, V2, V3}, 0);
  auto *result = F->createSave("ret", L);
  bindings.allocate(result->getPlaceholder());

  auto I1 = createTensorConditionallyQuantized(DTy, {10});
  auto I2 = createTensorConditionallyQuantized(DTy, {20});
  auto I3 = createTensorConditionallyQuantized(DTy, {30});

  for (dim_t i = 0; i < 10; i++) {
    I1.getHandle<DataType>().at({i}) = i;

    I2.getHandle<DataType>().at({i}) = i + 10;
    I2.getHandle<DataType>().at({i + 10}) = i + 20;
    I3.getHandle<DataType>().at({i}) = i + 30;
    I3.getHandle<DataType>().at({i + 10}) = i + 40;
    I3.getHandle<DataType>().at({i + 20}) = i + 50;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output vector.
  updateInputPlaceholders(bindings, {V1, V2, V3}, {&I1, &I2, &I3});
  EE.run(bindings);

  auto RNWH = bindings.get(result->getPlaceholder())->getHandle<DataType>();
  (void)RNWH;

  for (dim_t i = 0; i < 60; i++) {
    EXPECT_NEAR(RNWH.at({i}), static_cast<DataType>(i), 0.001);
  }
}

/// Test concatenating vectors that are Int64ITy.
TEST_P(OperatorTest, concatVectors_Int64) {
  CHECK_IF_ENABLED();
  testConcatVectors<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test concatenating vectors that are Int32ITy.
TEST_P(OperatorTest, concatVectors_Int32) {
  CHECK_IF_ENABLED();
  testConcatVectors<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test concatenating vectors that are Int8Qty.
TEST_P(OperatorTest, concatVectors_Int8) {
  CHECK_IF_ENABLED();
  testConcatVectors<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test concatenating vectors that are BoolTy.
TEST_P(OperatorTest, concatVectors_Bool) {
  CHECK_IF_ENABLED();
  testConcatVectors<bool>(bindings_, mod_, F_, EE_, ElemKind::BoolTy);
}

/// Test concatenating vectors that are FloatTy.
TEST_P(OperatorTest, concatVectors_Float) {
  CHECK_IF_ENABLED();
  testConcatVectors<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test concatenating vectors that are Float16Ty.
TEST_P(OperatorTest, concatVectors_Float16) {
  CHECK_IF_ENABLED();
  testConcatVectors<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test concatenating vectors that are Float16Ty.
TEST_P(OperatorTest, concatVectors_BFloat16) {
  CHECK_IF_ENABLED();
  testConcatVectors<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Helper to test ConcatVectorsRepeated using \p DTy.
template <typename DataType>
static void testConcatVectorsRepeated(glow::PlaceholderBindings &bindings,
                                      glow::Module &mod, glow::Function *F,
                                      glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("concatVectors");

  auto *V1 =
      createPlaceholderConditionallyQuantized(mod, DTy, {10}, "V1", false);
  auto *V2 =
      createPlaceholderConditionallyQuantized(mod, DTy, {20}, "V2", false);
  bindings.allocate(V1);
  bindings.allocate(V2);

  // Alternate adding sequences of V1 and V2, so that the IRGen'd
  // InsertTensors have different counts.
  Node *L = F->createConcat("concat", {V2, V1, V1, V1, V2, V2, V1, V1, V2}, 0);
  auto *result = F->createSave("ret", L);
  bindings.allocate(result->getPlaceholder());

  auto I1 = createTensorConditionallyQuantized(DTy, {10});
  auto I2 = createTensorConditionallyQuantized(DTy, {20});
  auto I1H = I1.getHandle<DataType>();
  auto I2H = I2.getHandle<DataType>();
  for (dim_t i = 0; i < 10; i++) {
    I1H.at({i}) = 1;

    I2H.at({i}) = 2;
    I2H.at({i + 10}) = 2;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output vector.
  updateInputPlaceholders(bindings, {V1, V2}, {&I1, &I2});
  EE.run(bindings);

  auto outH = bindings.get(result->getPlaceholder())->getHandle<DataType>();

  // Simply verify here that the values are in their correct places, based on
  // the number of times/order V1 and V2 are concatenated and their sizes.
  for (dim_t i = 0; i < 130; i++) {
    if ((i < 20) || (i >= 50 && i < 90) || (i >= 110)) {
      EXPECT_EQ(outH.at({i}), static_cast<DataType>(2));
    } else {
      EXPECT_EQ(outH.at({i}), static_cast<DataType>(1));
    }
  }
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Int64ITy data.
TEST_P(OperatorTest, concatVectorsRepeated_Int64) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<int64_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int64ITy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Int32ITy data.
TEST_P(OperatorTest, concatVectorsRepeated_Int32) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<int32_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::Int32ITy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Int8QTy data.
TEST_P(OperatorTest, concatVectorsRepeated_Int8) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<int8_t>(bindings_, mod_, F_, EE_,
                                    ElemKind::Int8QTy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing BoolTy data.
TEST_P(OperatorTest, concatVectorsRepeated_Bool) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<bool>(bindings_, mod_, F_, EE_, ElemKind::BoolTy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing FloatTy data.
TEST_P(OperatorTest, concatVectorsRepeated_Float) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing Float16Ty data.
TEST_P(OperatorTest, concatVectorsRepeated_Float16) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<float16_t>(bindings_, mod_, F_, EE_,
                                       ElemKind::Float16Ty);
}

/// Check that concatenating two tensors repeatedly is correct. This is
/// intended to verify that IRGen to InsertTensor instructions with axis/count
/// works correctly. Testing BFloat16Ty data.
TEST_P(OperatorTest, concatVectorsRepeated_BFloat16) {
  CHECK_IF_ENABLED();
  testConcatVectorsRepeated<bfloat16_t>(bindings_, mod_, F_, EE_,
                                        ElemKind::BFloat16Ty);
}

/// Helper to test SliceVectors using \p DTy.
template <typename DataType>
static void testSliceVectors(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("sliceVectors");

  auto *V =
      createPlaceholderConditionallyQuantized(mod, DTy, {3, 30}, "V", false);
  bindings.allocate(V);

  Node *S1 = F->createSlice("slice1", V, {0, 10}, {3, 13});
  Node *S2 = F->createSlice("slice2", V, {1, 0}, {2, 30});
  Node *S3 = F->createSlice("slice3", V, {2, 10}, {3, 12});

  auto *result1 = F->createSave("ret1", S1);
  auto *result2 = F->createSave("ret2", S2);
  auto *result3 = F->createSave("ret3", S3);

  bindings.allocate(result1->getPlaceholder());
  bindings.allocate(result2->getPlaceholder());
  bindings.allocate(result3->getPlaceholder());

  auto I = createTensorConditionallyQuantized(DTy, {3, 30});
  auto IH = I.getHandle<DataType>();
  for (dim_t j = 0; j < 30; j++) {
    IH.at({0, j}) = j;
    IH.at({1, j}) = j + 30;
    IH.at({2, j}) = j + 60;
  }

  EE.compile(CompilationMode::Infer);

  // Testing the output slices.
  updateInputPlaceholders(bindings, {V}, {&I});
  EE.run(bindings);

  auto RNWH1 = bindings.get(result1->getPlaceholder())->getHandle<DataType>();
  auto RNWH2 = bindings.get(result2->getPlaceholder())->getHandle<DataType>();
  auto RNWH3 = bindings.get(result3->getPlaceholder())->getHandle<DataType>();

  EXPECT_EQ(3, RNWH1.dims()[0]);
  EXPECT_EQ(3, RNWH1.dims()[1]);
  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 10; j < 13; j++) {
      EXPECT_NEAR(RNWH1.at({i, j - 10}), j + i * 30, 0.001);
    }
  }
  EXPECT_EQ(1, RNWH2.dims()[0]);
  EXPECT_EQ(30, RNWH2.dims()[1]);
  for (dim_t j = 0; j < 30; j++) {
    EXPECT_NEAR(RNWH2.at({0, j}), j + 30, 0.001);
  }
  EXPECT_EQ(1, RNWH3.dims()[0]);
  EXPECT_EQ(2, RNWH3.dims()[1]);
  for (dim_t j = 10; j < 12; j++) {
    EXPECT_NEAR(RNWH3.at({0, j - 10}), j + 60, 0.001);
  }
}

/// Test slicing with Int64ITy.
TEST_P(OperatorTest, sliceVectors_Int64) {
  CHECK_IF_ENABLED();
  testSliceVectors<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test slicing with FloatTy.
TEST_P(OperatorTest, sliceVectors_Float) {
  CHECK_IF_ENABLED();
  testSliceVectors<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test slicing with Float16Ty.
TEST_P(OperatorTest, sliceVectors_Float16) {
  CHECK_IF_ENABLED();
  testSliceVectors<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test slicing with BFloat16Ty.
TEST_P(OperatorTest, sliceVectors_BFloat16) {
  CHECK_IF_ENABLED();
  testSliceVectors<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Test slicing with Int8QTy.
TEST_P(OperatorTest, sliceVectors_Int8) {
  CHECK_IF_ENABLED();
  testSliceVectors<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test slicing with Int32QTy.
TEST_P(OperatorTest, sliceVectors_Int32Q) {
  CHECK_IF_ENABLED();
  testSliceVectors<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

/// Test slicing with Int32ITy.
TEST_P(OperatorTest, sliceVectors_Int32I) {
  CHECK_IF_ENABLED();
  testSliceVectors<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Helper to test SliceConcatVectors using \p DTy.
template <typename DataType>
static void testSliceConcatVectors(glow::PlaceholderBindings &bindings,
                                   glow::Module &mod, glow::Function *F,
                                   glow::ExecutionEngine &EE, ElemKind DTy) {
  F->setName("sliceConcatVectors");

  auto *V =
      createPlaceholderConditionallyQuantized(mod, DTy, {5, 4}, "V", false);
  bindings.allocate(V);

  auto I = createTensorConditionallyQuantized(DTy, {5, 4});
  auto IH = I.getHandle<DataType>();
  for (dim_t i = 0; i < 5; i++) {
    for (dim_t j = 0; j < 4; j++) {
      IH.at({i, j}) = i * 10 + j;
    }
  }

  Node *S0 = F->createSlice("slice0", V, {1, 0}, {5, 4});
  Node *S1 = F->createSlice("slice1", S0, {0, 0}, {2, 4});
  Node *S2 = F->createSlice("slice2", S0, {2, 0}, {4, 4});
  Node *S3 = F->createSlice("slice3", S0, {0, 0}, {2, 2});
  Node *S4 = F->createSlice("slice4", S0, {2, 2}, {4, 4});
  Node *S5 = F->createSlice("slice5", V, {0, 0}, {1, 4});

  Node *C0 = F->createConcat("concat0", {S5, S1}, 0);
  Node *C1 = F->createConcat("concat1", {S3, S4}, 1);
  Node *C2 = F->createConcat("concat2", {S2, C1, C0}, 0);

  auto *result = F->createSave("ret", C2);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings, {V}, {&I});
  EE.run(bindings);

  const DataType expected[7][4] = {
      {30, 31, 32, 33}, {40, 41, 42, 43}, {10, 11, 32, 33}, {20, 21, 42, 43},
      {0, 1, 2, 3},     {10, 11, 12, 13}, {20, 21, 22, 23}};

  auto resultH = bindings.get(result->getPlaceholder())->getHandle<DataType>();
  EXPECT_EQ(7, resultH.dims()[0]);
  EXPECT_EQ(4, resultH.dims()[1]);
  for (dim_t i = 0; i < 7; i++) {
    for (dim_t j = 0; j < 4; j++) {
      EXPECT_EQ(resultH.at({i, j}), expected[i][j]);
    }
  }
}

/// Test a combination of slicing and concating, in Int64ITy.
TEST_P(OperatorTest, sliceConcatVectors_Int64) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test a combination of slicing and concating, in Int8QTy.
TEST_P(OperatorTest, sliceConcatVectors_Int8) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test a combination of slicing and concating, in FloatTy.
TEST_P(OperatorTest, sliceConcatVectors_Float) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test a combination of slicing and concating, in Float16Ty.
TEST_P(OperatorTest, sliceConcatVectors_Float16) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<float16_t>(bindings_, mod_, F_, EE_,
                                    ElemKind::Float16Ty);
}

/// Test a combination of slicing and concating, in BFloat16Ty.
TEST_P(OperatorTest, sliceConcatVectors_BFloat16) {
  CHECK_IF_ENABLED();
  testSliceConcatVectors<bfloat16_t>(bindings_, mod_, F_, EE_,
                                     ElemKind::BFloat16Ty);
}

TEST_P(OperatorTest, Tile) {
  CHECK_IF_ENABLED();

  F_->setName("concatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::FloatTy, {4, 5}, "V", false);
  bindings_.allocate(V);

  Node *T0 = F_->createTile("tile0", V, /* tiles */ 3, /* axis */ 0);
  auto *result0 = F_->createSave("res0", T0);
  bindings_.allocate(result0->getPlaceholder());

  Node *T1 = F_->createTile("tile1", V, /* tiles */ 3, /* axis */ 1);
  auto *result1 = F_->createSave("res1", T1);
  bindings_.allocate(result1->getPlaceholder());

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings_, {V}, {&VT});
  EE_.run(bindings_);

  // Testing the output vector with axis 0.
  auto res0 = bindings_.get(result0->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < res0.dims()[0]; i++) {
    for (dim_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_EQ(res0.at({i, j}), (i % 4) * 5 + j);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = bindings_.get(result1->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < res1.dims()[0]; i++) {
    for (dim_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_EQ(res1.at({i, j}), i * 5 + (j % 5));
    }
  }
}

TEST_P(OperatorTest, QuantizedTile) {
  CHECK_IF_ENABLED();

  F_->setName("concatVectors");

  auto *V = mod_.createPlaceholder(ElemKind::FloatTy, {4, 5}, "V", false);
  bindings_.allocate(V);

  auto quantizationParams =
      glow::quantization::chooseQuantizationParams({0, 20});
  auto quantizeTy =
      mod_.uniqueType(ElemKind::Int8QTy, {4, 5}, quantizationParams.scale,
                      quantizationParams.offset);
  auto *Q = F_->createQuantize("quantize", V, quantizeTy);

  Node *T0 = F_->createTile("tile0", Q, /* tiles */ 3, /* axis */ 0);
  auto *DQ0 = F_->createDequantize("dequantize0", T0, ElemKind::FloatTy);
  auto *result0 = F_->createSave("res0", DQ0);
  bindings_.allocate(result0->getPlaceholder());

  Node *T1 = F_->createTile("tile1", Q, /* tiles */ 3, /* axis */ 1);
  auto *DQ1 = F_->createDequantize("dequantize1", T1, ElemKind::FloatTy);
  auto *result1 = F_->createSave("res1", DQ1);
  bindings_.allocate(result1->getPlaceholder());

  Tensor VT(ElemKind::FloatTy, {4, 5});

  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 5; j++) {
      VT.getHandle<float>().at({i, j}) = i * 5 + j;
    }
  }

  EE_.compile(CompilationMode::Infer);

  updateInputPlaceholders(bindings_, {V}, {&VT});
  EE_.run(bindings_);

  // Testing the output vector with axis 0.
  auto res0 = bindings_.get(result0->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < res0.dims()[0]; i++) {
    for (dim_t j = 0; j < res0.dims()[1]; j++) {
      EXPECT_NEAR(res0.at({i, j}), (i % 4) * 5 + j, 0.05);
    }
  }

  // Testing the output vector with axis 1.
  auto res1 = bindings_.get(result1->getPlaceholder())->getHandle<float>();
  (void)res1;
  for (dim_t i = 0; i < res1.dims()[0]; i++) {
    for (dim_t j = 0; j < res1.dims()[1]; j++) {
      EXPECT_NEAR(res1.at({i, j}), i * 5 + (j % 5), 0.05);
    }
  }
}

TEST_P(OperatorTest, Clip) {
  CHECK_IF_ENABLED();

  auto *X = mod_.createPlaceholder(ElemKind::FloatTy, {5, 5}, "X", false);
  auto xHandle = bindings_.allocate(X)->getHandle();
  xHandle = {45.0, 16.0, 59.0, 99.0, 48.0, 12.0, 44.0, 46.0, 82.0,
             28.0, 1.0,  91.0, 18.0, 9.0,  71.0, 24.0, 37.0, 61.0,
             12.0, 81.0, 36.0, 38.0, 30.0, 84.0, 40.0};

  float min = 20.0;
  float max = 60.0;
  auto *node = F_->createClip("clip", X, min, max);
  auto *save = F_->createSave("save", node);
  auto *saveTensor = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = saveTensor->getHandle();
  std::vector<dim_t> expectedDims = {5, 5};
  std::vector<float> expectedValues = {45.0, 20.0, 59.0, 60.0, 48.0, 20.0, 44.0,
                                       46.0, 60.0, 28.0, 20.0, 60.0, 20.0, 20.0,
                                       60.0, 24.0, 37.0, 60.0, 20.0, 60.0, 36.0,
                                       38.0, 30.0, 60.0, 40.0};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  for (size_t i = 0; i < 5 * 5; i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expectedValues[i]);
  }
}

TEST_P(OperatorTest, LeakyRelu_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {3}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-2, 0.0, 2};
  auto *node = F_->createLeakyRELU("leaky_relu", inp, /* alpha */ 0.5);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_FLOAT_EQ(outH.raw(0), -1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 0.0);
  EXPECT_FLOAT_EQ(outH.raw(2), 2.0);
}

TEST_P(OperatorTest, LeakyRelu_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {5}, 0.5, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-4, -2, 0, 2, 4};
  auto *node = F_->createLeakyRELU("leaky_relu", inp, /* alpha */ 0.5);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 5);
  EXPECT_EQ(outH.raw(0), -2);
  EXPECT_EQ(outH.raw(1), -1);
  EXPECT_EQ(outH.raw(2), 0);
  EXPECT_EQ(outH.raw(3), 2);
  EXPECT_EQ(outH.raw(4), 4);
}

TEST_P(OperatorTest, Not) {
  CHECK_IF_ENABLED();
  auto *input = mod_.createPlaceholder(ElemKind::BoolTy, {2}, "inp", false);
  bindings_.allocate(input)->getHandle<bool>() = {false, true};
  auto *node = F_->createNot("not", input);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 2);
  EXPECT_EQ(outH.raw(0), true);
  EXPECT_EQ(outH.raw(1), false);
}

TEST_P(OperatorTest, And) {
  CHECK_IF_ENABLED();
  auto *LHS = mod_.createPlaceholder(ElemKind::BoolTy, {4}, "LHS", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::BoolTy, {4}, "RHS", false);
  bindings_.allocate(LHS)->getHandle<bool>() = {false, true, false, true};
  bindings_.allocate(RHS)->getHandle<bool>() = {false, false, true, true};
  auto *node = F_->createAnd("and", LHS, RHS);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), false);
  EXPECT_EQ(outH.raw(1), false);
  EXPECT_EQ(outH.raw(2), false);
  EXPECT_EQ(outH.raw(3), true);
}

TEST_P(OperatorTest, Or) {
  CHECK_IF_ENABLED();
  auto *LHS = mod_.createPlaceholder(ElemKind::BoolTy, {4}, "LHS", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::BoolTy, {4}, "RHS", false);
  bindings_.allocate(LHS)->getHandle<bool>() = {false, true, false, true};
  bindings_.allocate(RHS)->getHandle<bool>() = {false, false, true, true};
  auto *node = F_->createOr("or", LHS, RHS);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), false);
  EXPECT_EQ(outH.raw(1), true);
  EXPECT_EQ(outH.raw(2), true);
  EXPECT_EQ(outH.raw(3), true);
}

TEST_P(OperatorTest, Xor) {
  CHECK_IF_ENABLED();
  auto *LHS = mod_.createPlaceholder(ElemKind::BoolTy, {4}, "LHS", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::BoolTy, {4}, "RHS", false);
  bindings_.allocate(LHS)->getHandle<bool>() = {false, true, false, true};
  bindings_.allocate(RHS)->getHandle<bool>() = {false, false, true, true};
  auto *node = F_->createXor("xor", LHS, RHS);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), false);
  EXPECT_EQ(outH.raw(1), true);
  EXPECT_EQ(outH.raw(2), true);
  EXPECT_EQ(outH.raw(3), false);
}

TEST_P(OperatorTest, Abs_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-1.0, 1.0};
  auto *node = F_->createAbs("abs", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 2);
  EXPECT_FLOAT_EQ(outH.raw(0), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 1.0);
}

TEST_P(OperatorTest, Abs_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-1, 1};
  auto *node = F_->createAbs("abs", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 2);
  EXPECT_EQ(outH.raw(0), 1);
  EXPECT_EQ(outH.raw(1), 1);
}

TEST_P(OperatorTest, Neg_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {1.0, -1.0};
  auto *node = F_->createNeg("neg", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 2);
  EXPECT_FLOAT_EQ(outH.raw(0), -1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 1.0);
}

TEST_P(OperatorTest, Neg_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {2}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-1, 1};
  auto *node = F_->createNeg("neg", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 2);
  EXPECT_EQ(outH.raw(0), 1);
  EXPECT_EQ(outH.raw(1), -1);
}

TEST_P(OperatorTest, Floor_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {3}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-0.2, 1.0, 1.99};
  auto *node = F_->createFloor("floor", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_FLOAT_EQ(outH.raw(0), -1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(2), 1.0);
}

TEST_P(OperatorTest, Floor_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {5}, 0.5, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-2, -1, 0, 1, 2};
  auto *node = F_->createFloor("floor", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 5);
  EXPECT_EQ(outH.raw(0), -2);
  EXPECT_EQ(outH.raw(1), -2);
  EXPECT_EQ(outH.raw(2), 0);
  EXPECT_EQ(outH.raw(3), 0);
  EXPECT_EQ(outH.raw(4), 2);
}

TEST_P(OperatorTest, Sign_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {3}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-1.0, 0.0, 1.0};
  auto *node = F_->createSign("Sign", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_FLOAT_EQ(outH.raw(0), -1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 0.0);
  EXPECT_FLOAT_EQ(outH.raw(2), 1.0);
}

TEST_P(OperatorTest, Sign_Int8QTy) {
  CHECK_IF_ENABLED();

  auto qParams = glow::quantization::chooseQuantizationParams({-100, 100});
  auto *inp = mod_.createPlaceholder(ElemKind::Int8QTy, {3}, qParams.scale,
                                     qParams.offset, "input", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-100, 0, 100};

  auto *node = F_->createSign("Sign", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_EQ(outH.raw(0), -1);
  EXPECT_EQ(outH.raw(1), 0);
  EXPECT_EQ(outH.raw(2), 1);
}

TEST_P(OperatorTest, Ceil_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {3}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-0.2, 1.0, 1.99};
  auto *node = F_->createCeil("ceil", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_FLOAT_EQ(outH.raw(0), 0.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(2), 2.0);
}

TEST_P(OperatorTest, Ceil_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {5}, 0.5, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-2, -1, 0, 1, 2};
  auto *node = F_->createCeil("ceil", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 5);
  EXPECT_EQ(outH.raw(0), -2);
  EXPECT_EQ(outH.raw(1), 0);
  EXPECT_EQ(outH.raw(2), 0);
  EXPECT_EQ(outH.raw(3), 2);
  EXPECT_EQ(outH.raw(4), 2);
}

TEST_P(OperatorTest, Round_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {0.9, 2.5, 2.3, 1.5, -4.5};
  auto *node = F_->createRound("round", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 5);
  // Rounding mode required by ONNX, Numpy, TensorFlow is round to even which
  // rounds to nearest even integer those values with fractional part 0.5.
  EXPECT_FLOAT_EQ(outH.raw(0), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 2.0);
  EXPECT_FLOAT_EQ(outH.raw(2), 2.0);
  EXPECT_FLOAT_EQ(outH.raw(3), 2.0);
  EXPECT_FLOAT_EQ(outH.raw(4), -4.0);
}

TEST_P(OperatorTest, Round_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {5}, 0.1, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-8, -2, 0, 2, 8};
  auto *node = F_->createRound("round", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 5);
  EXPECT_EQ(outH.raw(0), -10);
  EXPECT_EQ(outH.raw(1), 0);
  EXPECT_EQ(outH.raw(2), 0);
  EXPECT_EQ(outH.raw(3), 0);
  EXPECT_EQ(outH.raw(4), 10);
}

TEST_P(OperatorTest, Sqrt_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {0.0, 1.0, 4.0, 9.0};
  auto *node = F_->createSqrt("sqrt", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_FLOAT_EQ(outH.raw(0), 0.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(2), 2.0);
  EXPECT_FLOAT_EQ(outH.raw(3), 3.0);
}

TEST_P(OperatorTest, Sqrt_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {4}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {0, 1, 4, 9};
  auto *node = F_->createSqrt("sqrt", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), 0);
  EXPECT_EQ(outH.raw(1), 1);
  EXPECT_EQ(outH.raw(2), 2);
  EXPECT_EQ(outH.raw(3), 3);
}

TEST_P(OperatorTest, Rsqrt_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {1.0, 4.0, 16.0, 64.0};
  auto *node = F_->createRsqrt("rsqrt", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_FLOAT_EQ(outH.raw(0), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 0.5);
  EXPECT_FLOAT_EQ(outH.raw(2), 0.25);
  EXPECT_FLOAT_EQ(outH.raw(3), 0.125);
}

TEST_P(OperatorTest, Rsqrt_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {4}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {1, 4, 16, 64};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {4}, 1.0 / 8.0, 0);
  auto *node = F_->createRsqrt("rsqrt", outTy, inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), 8);
  EXPECT_EQ(outH.raw(1), 4);
  EXPECT_EQ(outH.raw(2), 2);
  EXPECT_EQ(outH.raw(3), 1);
}

TEST_P(OperatorTest, Reciprocal_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {1.0, 2.0, 4.0, 8.0};
  auto *node = F_->createReciprocal("reciprocal", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_FLOAT_EQ(outH.raw(0), 1.0);
  EXPECT_FLOAT_EQ(outH.raw(1), 0.5);
  EXPECT_FLOAT_EQ(outH.raw(2), 0.25);
  EXPECT_FLOAT_EQ(outH.raw(3), 0.125);
}

TEST_P(OperatorTest, Reciprocal_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {4}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {1, 2, 4, 8};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {4}, 1.0 / 8.0, 0);
  auto *node = F_->createReciprocal("reciprocal", outTy, inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), 8);
  EXPECT_EQ(outH.raw(1), 4);
  EXPECT_EQ(outH.raw(2), 2);
  EXPECT_EQ(outH.raw(3), 1);
}

TEST_P(OperatorTest, Sin_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-1.0, 0.0, 1.0, 2.0};
  auto *node = F_->createSin("sin", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_FLOAT_EQ(outH.raw(0), std::sin(-1.0));
  EXPECT_FLOAT_EQ(outH.raw(1), std::sin(0.0));
  EXPECT_FLOAT_EQ(outH.raw(2), std::sin(1.0));
  EXPECT_FLOAT_EQ(outH.raw(3), std::sin(2.0));
}

TEST_P(OperatorTest, Sin_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {4}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-1, 0, 1, 2};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {4}, 1.0 / 127.0, 0);
  auto *node = F_->createSin("sin", outTy, inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), static_cast<int8_t>(std::round(std::sin(-1) * 127)));
  EXPECT_EQ(outH.raw(1), static_cast<int8_t>(std::round(std::sin(0) * 127)));
  EXPECT_EQ(outH.raw(2), static_cast<int8_t>(std::round(std::sin(1) * 127)));
  EXPECT_EQ(outH.raw(3), static_cast<int8_t>(std::round(std::sin(2) * 127)));
}

TEST_P(OperatorTest, Cos_FloatTy) {
  CHECK_IF_ENABLED();
  auto *inp = mod_.createPlaceholder(ElemKind::FloatTy, {4}, "inp", false);
  bindings_.allocate(inp)->getHandle<float>() = {-1.0, 0.0, 1.0, 2.0};
  auto *node = F_->createCos("cos", inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<float>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_FLOAT_EQ(outH.raw(0), std::cos(-1.0));
  EXPECT_FLOAT_EQ(outH.raw(1), std::cos(0.0));
  EXPECT_FLOAT_EQ(outH.raw(2), std::cos(1.0));
  EXPECT_FLOAT_EQ(outH.raw(3), std::cos(2.0));
}

TEST_P(OperatorTest, Cos_Int8QTy) {
  CHECK_IF_ENABLED();
  auto *inp =
      mod_.createPlaceholder(ElemKind::Int8QTy, {4}, 1.0, 0, "inp", false);
  bindings_.allocate(inp)->getHandle<int8_t>() = {-1, 0, 1, 2};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {4}, 1.0 / 127.0, 0);
  auto *node = F_->createCos("cos", outTy, inp);
  auto *save = F_->createSave("save", node);
  auto *outT = bindings_.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto outH = outT->getHandle<int8_t>();
  EXPECT_EQ(outH.size(), 4);
  EXPECT_EQ(outH.raw(0), static_cast<int8_t>(std::round(std::cos(-1) * 127)));
  EXPECT_EQ(outH.raw(1), static_cast<int8_t>(std::round(std::cos(0) * 127)));
  EXPECT_EQ(outH.raw(2), static_cast<int8_t>(std::round(std::cos(1) * 127)));
  EXPECT_EQ(outH.raw(3), static_cast<int8_t>(std::round(std::cos(2) * 127)));
}

/// Helper to test CmpNEQ using \p elemKind.
template <typename ElemType>
static void testCmpNEQ(glow::PlaceholderBindings &bindings, glow::Module &mod,
                       glow::Function *F, glow::ExecutionEngine &EE,
                       ElemKind elemKind) {
  auto *LHS =
      createPlaceholderConditionallyQuantized(mod, elemKind, {2}, "LHS", false);
  auto *RHS =
      createPlaceholderConditionallyQuantized(mod, elemKind, {2}, "RHS", false);
  bindings.allocate(LHS)->getHandle<ElemType>() = {1, 1};
  bindings.allocate(RHS)->getHandle<ElemType>() = {1, 2};
  auto *node = F->createCmpNEQ("cmpNEQ", LHS, RHS);
  auto *save = F->createSave("save", node);
  auto *outT = bindings.allocate(save->getPlaceholder());
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 2);
  EXPECT_EQ(outH.raw(0), false);
  EXPECT_EQ(outH.raw(1), true);
}

TEST_P(OperatorTest, CmpNEQ_FloatTy) {
  CHECK_IF_ENABLED();
  testCmpNEQ<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, CmpNEQ_Int8QTy) {
  CHECK_IF_ENABLED();
  testCmpNEQ<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, CmpNEQ_Int32ITy) {
  CHECK_IF_ENABLED();
  testCmpNEQ<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

TEST_P(OperatorTest, CmpNEQ_Int64ITy) {
  CHECK_IF_ENABLED();
  testCmpNEQ<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Helper to test CmpGT using \p elemKind.
template <typename ElemType>
static void testCmpGT(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind elemKind) {
  auto *LHS =
      createPlaceholderConditionallyQuantized(mod, elemKind, {3}, "LHS", false);
  auto *RHS =
      createPlaceholderConditionallyQuantized(mod, elemKind, {3}, "RHS", false);
  bindings.allocate(LHS)->getHandle<ElemType>() = {1, 1, 2};
  bindings.allocate(RHS)->getHandle<ElemType>() = {1, 2, 1};
  auto *node = F->createCmpGT("cmpGT", LHS, RHS);
  auto *save = F->createSave("save", node);
  auto *outT = bindings.allocate(save->getPlaceholder());
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_EQ(outH.raw(0), false);
  EXPECT_EQ(outH.raw(1), false);
  EXPECT_EQ(outH.raw(2), true);
}

TEST_P(OperatorTest, CmpGT_FloatTy) {
  CHECK_IF_ENABLED();
  testCmpGT<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, CmpGT_Int8QTy) {
  CHECK_IF_ENABLED();
  testCmpGT<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, CmpGT_Int32ITy) {
  CHECK_IF_ENABLED();
  testCmpGT<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

TEST_P(OperatorTest, CmpGT_Int64ITy) {
  CHECK_IF_ENABLED();
  testCmpGT<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Helper to test CmpGTE using \p elemKind.
template <typename ElemType>
static void testCmpGTE(glow::PlaceholderBindings &bindings, glow::Module &mod,
                       glow::Function *F, glow::ExecutionEngine &EE,
                       ElemKind elemKind) {
  auto *LHS =
      createPlaceholderConditionallyQuantized(mod, elemKind, {3}, "LHS", false);
  auto *RHS =
      createPlaceholderConditionallyQuantized(mod, elemKind, {3}, "RHS", false);
  bindings.allocate(LHS)->getHandle<ElemType>() = {1, 1, 2};
  bindings.allocate(RHS)->getHandle<ElemType>() = {1, 2, 1};
  auto *node = F->createCmpGTE("cmpGTE", LHS, RHS);
  auto *save = F->createSave("save", node);
  auto *outT = bindings.allocate(save->getPlaceholder());
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  auto outH = outT->getHandle<bool>();
  EXPECT_EQ(outH.size(), 3);
  EXPECT_EQ(outH.raw(0), true);
  EXPECT_EQ(outH.raw(1), false);
  EXPECT_EQ(outH.raw(2), true);
}

TEST_P(OperatorTest, CmpGTE_FloatTy) {
  CHECK_IF_ENABLED();
  testCmpGTE<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, CmpGTE_Int8QTy) {
  CHECK_IF_ENABLED();
  testCmpGTE<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, CmpGTE_Int32ITy) {
  CHECK_IF_ENABLED();
  testCmpGTE<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

TEST_P(OperatorTest, CmpGTE_Int64ITy) {
  CHECK_IF_ENABLED();
  testCmpGTE<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

TEST_P(OperatorTest, simpleCmpSelectPredication) {
  CHECK_IF_ENABLED();

  // A simple test that checks predication of some values using the
  // compare-select pair of instructions. Keep doubling some values
  // until some condition is met.

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "inputs", false);
  auto *counters =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);

  bindings_.allocate(counters)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  bindings_.allocate(inputs)->getHandle().clear(1);

  Node *cnt = counters;
  NodeValue data = inputs;
  Node *const1 = F_->createSplat("const1", counters->getType(), 1.0);
  Node *const0 = F_->createSplat("const0", counters->getType(), 0.0);

  for (int i = 0; i < 10; i++) {
    cnt = F_->createSub("sub1", cnt, const1);
    Node *pred = F_->createCmpLTE("cmp", const0, cnt);

    Node *const2 = F_->createSplat("const2", data.getType(), 2.0);
    Node *newData = F_->createMul("mul2x", data, const2);

    data = F_->createSelect("select", pred, newData, data);
  }

  auto *SN = F_->createSave("ret", data);
  bindings_.allocate(SN->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto H = bindings_.get(SN->getPlaceholder())->getHandle();
  ASSERT_NEAR(H.at(0), 1, 0.001);
  ASSERT_NEAR(H.at(1), 2, 0.001);
  ASSERT_NEAR(H.at(2), 4, 0.001);
  ASSERT_NEAR(H.at(3), 8, 0.001);
  ASSERT_NEAR(H.at(4), 16, 0.001);
  ASSERT_NEAR(H.at(5), 32, 0.001);
  ASSERT_NEAR(H.at(6), 64, 0.001);
  ASSERT_NEAR(H.at(7), 128, 0.001);
  ASSERT_NEAR(H.at(8), 256, 0.001);
  ASSERT_NEAR(H.at(9), 512, 0.001);
}

TEST_P(OperatorTest, simplePredication) {
  CHECK_IF_ENABLED();

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 10, 10}, "inputs", false);
  auto *counters =
      mod_.createPlaceholder(ElemKind::FloatTy, {10}, "counters", false);

  bindings_.allocate(counters)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  bindings_.allocate(inputs)->getHandle().randomize(-10, 10, mod_.getPRNG());

  Node *C5 = F_->createSplat("C5", counters->getType(), 5.0);
  Node *pred = F_->createCmpLTE("cmp", C5, counters);

  auto *FC0 = F_->createFullyConnected(bindings_, "FC0", inputs, 128);
  auto *RL0 = F_->createRELU("RL0", FC0);
  auto *FC1 = F_->createFullyConnected(bindings_, "FC1", RL0, 64);
  auto *RL1 = F_->createRELU("RL1", FC1);
  auto *FC2 = F_->createFullyConnected(bindings_, "FC2", RL1, 32);
  auto *RL2 = F_->createRELU("RL2", FC2);

  auto *save = F_->createSave("ret", RL2);
  bindings_.allocate(save->getPlaceholder());

  FC0->setPredicate(pred);
  FC1->setPredicate(pred);
  FC2->setPredicate(pred);

  ::glow::convertPlaceholdersToConstants(
      F_, bindings_, {inputs, counters, save->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
}

TEST_P(OperatorTest, ChannelShuffle) {
  CHECK_IF_ENABLED();

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 12, 1, 1}, "inputs", false);
  bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4,  5,  6,
                                             7, 8, 9, 10, 11, 12};

  Node *CS = F_->createChannelShuffle("CS", inputs, 3, 1);
  SaveNode *S = F_->createSave("save", CS);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto results = bindings_.get(S->getPlaceholder())->getHandle();

  EXPECT_EQ(results.size(), 12);
  std::vector<float> expected = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  for (dim_t i = 0; i < expected.size(); i++)
    EXPECT_FLOAT_EQ(results.at({0, i, 0, 0}), expected[i]);
}

TEST_P(OperatorTest, SqueezeOneAxis) {
  CHECK_IF_ENABLED();

  auto *inputs =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs", false);
  bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> expectedValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<dim_t> axes = {0};
  Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
  SaveNode *S = F_->createSave("save", SQZ);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto results = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {2, 1, 5};
  EXPECT_TRUE(results.dims().vec() == expectedDims);
  for (size_t i = 0; i < 10; i++)
    EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
}

TEST_P(OperatorTest, SqueezeTwoAxes) {
  CHECK_IF_ENABLED();

  auto mod = &EE_.getModule();
  auto *inputs =
      mod->createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs", false);
  bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<float> expectedValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<dim_t> axes = {0, 2, 2};
  Node *SQZ = F_->createSqueeze("SQZ", inputs, axes);
  SaveNode *S = F_->createSave("save", SQZ);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto results = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {2, 5};
  EXPECT_TRUE(results.dims().vec() == expectedDims);
  for (size_t i = 0; i < 10; i++)
    EXPECT_FLOAT_EQ(results.raw(i), expectedValues[i]);
}

TEST_P(OperatorTest, SqueezeExpand) {
  CHECK_IF_ENABLED();

  auto mod = &EE_.getModule();
  auto *inputs =
      mod->createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 5}, "inputs", false);
  bindings_.allocate(inputs)->getHandle() = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto *emptyInput =
      mod->createPlaceholder(ElemKind::FloatTy, {1}, "emptyInput", false);
  bindings_.allocate(emptyInput)->getHandle() = {42.0};

  std::vector<float> expectedValues = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<dim_t> axes = {0};
  Node *SQZ = F_->createSqueeze("SQZ", emptyInput, axes);
  SaveNode *S1 = F_->createSave("save", SQZ);
  Node *UnSQZ = F_->createExpandDims("UnSQZ", SQZ, axes);
  SaveNode *S2 = F_->createSave("save", UnSQZ);

  bindings_.allocate(S1->getPlaceholder());
  bindings_.allocate(S2->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto res1 = bindings_.get(S1->getPlaceholder())->getHandle();
  EXPECT_TRUE(res1.dims().vec() == std::vector<dim_t>());
  EXPECT_FLOAT_EQ(res1.raw(0), 42.0);
  auto res2 = bindings_.get(S2->getPlaceholder())->getHandle();
  EXPECT_TRUE(res2.dims().vec() == std::vector<dim_t>(1, 1));
  EXPECT_FLOAT_EQ(res2.raw(0), 42.0);
}

/// Helper to test ExpandDims using \p DTy.
template <typename DataType>
static void testExpandDims(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *inputs = createPlaceholderConditionallyQuantized(mod, DTy, {2, 2},
                                                         "inputs", false);
  auto IH = bindings.allocate(inputs)->getHandle<DataType>();
  IH = {1, 2, 3, 4};

  // This should be uniqued and sorted, so should become {0, 1, 3, 5}.
  std::vector<dim_t> axes = {3, 0, 5, 1, 3};
  Node *EDN = F->createExpandDims("expand", inputs, axes);
  SaveNode *S = F->createSave("save", EDN);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Expected dims based on the axes above; inserted new dimensions of 1 in
  // every unique axes location, based on the output tensor shape.
  std::vector<dim_t> expectedDims = {1, 1, 2, 1, 2, 1};
  auto results = bindings.get(S->getPlaceholder())->getHandle<DataType>();
  EXPECT_TRUE(results.dims().vec() == expectedDims);

  // The data should be the same, as this was just a reshape.
  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(results.raw(i), IH.raw(i));
  }
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in FloatTy.
TEST_P(OperatorTest, ExpandDims_Float) {
  CHECK_IF_ENABLED();
  testExpandDims<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in Float16Ty.
TEST_P(OperatorTest, ExpandDims_Float16) {
  CHECK_IF_ENABLED();
  testExpandDims<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in BFloat16Ty.
TEST_P(OperatorTest, ExpandDims_BFloat16) {
  CHECK_IF_ENABLED();
  testExpandDims<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Check that the expand dims operator works, which is implemented with a
/// reshape, in Int8QTy.
TEST_P(OperatorTest, ExpandDims_Int8) {
  CHECK_IF_ENABLED();
  testExpandDims<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Helper to test Split using \p DTy.
template <typename DataType>
static void testSplit(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind DTy) {
  auto *inputs = createPlaceholderConditionallyQuantized(mod, DTy, {1, 2, 6},
                                                         "inputs", false);
  bindings.allocate(inputs)->getHandle<DataType>() = {1, 2, 3, 4,  5,  6,
                                                      7, 8, 9, 10, 11, 12};

  std::vector<SliceNode *> outputs1;
  F->createSplit("Split1", inputs, /*outputNum = */ 2, /*axis = */ 2,
                 /*split = */ {}, outputs1);
  std::vector<SliceNode *> outputs2;
  F->createSplit("Split2", inputs, /*outputNum = */ 2, /*axis = */ 2,
                 /*split = */ {2, 4}, outputs2);
  auto *S1 = F->createSave("save1", outputs1[0]);
  auto *S2 = F->createSave("save2", outputs1[1]);
  auto *S3 = F->createSave("save3", outputs2[0]);
  auto *S4 = F->createSave("save4", outputs2[1]);

  auto *result1 = bindings.allocate(S1->getPlaceholder());
  auto *result2 = bindings.allocate(S2->getPlaceholder());
  auto *result3 = bindings.allocate(S3->getPlaceholder());
  auto *result4 = bindings.allocate(S4->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expected1 = createTensorConditionallyQuantized(DTy, {1, 2, 3});
  expected1.getHandle<DataType>() = {1, 2, 3, 7, 8, 9};
  EXPECT_TRUE(result1->isEqual(expected1));

  Tensor expected2 = createTensorConditionallyQuantized(DTy, {1, 2, 3});
  expected2.getHandle<DataType>() = {4, 5, 6, 10, 11, 12};
  EXPECT_TRUE(result2->isEqual(expected2));

  Tensor expected3 = createTensorConditionallyQuantized(DTy, {1, 2, 2});
  expected3.getHandle<DataType>() = {1, 2, 7, 8};
  EXPECT_TRUE(result3->isEqual(expected3));

  Tensor expected4 = createTensorConditionallyQuantized(DTy, {1, 2, 4});
  expected4.getHandle<DataType>() = {3, 4, 5, 6, 9, 10, 11, 12};
  EXPECT_TRUE(result4->isEqual(expected4));
}

/// Test that Split is correctly supported in FloatTy.
TEST_P(OperatorTest, Split_Float) {
  CHECK_IF_ENABLED();
  testSplit<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test that Split is correctly supported in Float16Ty.
TEST_P(OperatorTest, Split_Float16) {
  CHECK_IF_ENABLED();
  testSplit<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test that Split is correctly supported in BFloat16Ty.
TEST_P(OperatorTest, Split_BFloat16) {
  CHECK_IF_ENABLED();
  testSplit<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Test that Split is correctly supported in Int8QTy.
TEST_P(OperatorTest, Split_Int8) {
  CHECK_IF_ENABLED();
  testSplit<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Test Relu with Int8QTy.
TEST_P(OperatorTest, Relu_Int8) {
  CHECK_IF_ENABLED();

  std::vector<float> inputVals = {-2.0, -1.0, 0.0, 1.0, 2.0};
  dim_t size = inputVals.size();
  const float inputScale = 1.0;
  const int32_t inputOffset = 5;
  const float outputScale = 0.5;
  const int32_t outputOffset = -128;

  auto *inputTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, inputScale, inputOffset);
  auto *outputTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, outputScale, outputOffset);
  auto *input = mod_.createPlaceholder(inputTy, "input", false);
  auto *relu = F_->createRELU("relu", input, outputTy);
  auto *dequantize =
      F_->createDequantize("dequantize", relu, ElemKind::FloatTy);
  auto *save = F_->createSave("save", dequantize);
  bindings_.allocate(mod_.getPlaceholders());

  auto inputH = bindings_.get(input)->getHandle<int8_t>();
  for (dim_t idx = 0; idx < size; idx++) {
    inputH.raw(idx) =
        quantization::quantize(inputVals[idx], {inputScale, inputOffset});
  }

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputH = bindings_.get(save->getPlaceholder())->getHandle();
  for (dim_t idx = 0; idx < size; idx++) {
    float expectedValue = std::max(0.0f, inputVals[idx]);
    EXPECT_EQ(expectedValue, outputH.raw(idx));
  }
}

// Test for elementwise add with quantization and broadcast support
TEST_P(OperatorTest, IntElementWiseBroadcast) {
  CHECK_IF_ENABLED();

  const float in1Scale = 0.9;
  const float in2Scale = 1.2;
  const float outScale = 1;
  const int32_t in1Offset = 2;
  const int32_t in2Offset = -11;
  const int32_t outOffset = -2;
  const dim_t N = 2;
  const dim_t C = 3;
  const dim_t H = 4;
  const dim_t W = 5;

  auto in1Ty =
      mod_.uniqueType(ElemKind::Int8QTy, {N, C, H, W}, in1Scale, in1Offset);
  auto in2Ty = mod_.uniqueType(ElemKind::Int8QTy, {W}, in2Scale, in2Offset);
  auto outTy =
      mod_.uniqueType(ElemKind::Int8QTy, {N, C, H, W}, outScale, outOffset);

  auto *in1 = mod_.createPlaceholder(in1Ty, "in1", false);
  auto *in2 = mod_.createPlaceholder(in2Ty, "in2", false);

  bindings_.allocate(in1)->getHandle<int8_t>().randomize(-10, 10,
                                                         mod_.getPRNG());
  bindings_.allocate(in2)->getHandle<int8_t>().randomize(-10, 10,
                                                         mod_.getPRNG());
  constexpr int axis = -1;
  auto *addBroadcast = F_->createNodeWithBroadcastOutTy<AddNode>(
      "addBroadcast", axis, outTy, in1, in2);

  auto *subBroadcast = F_->createNodeWithBroadcastOutTy<SubNode>(
      "subBroadcast", axis, outTy, in1, in2);

  auto *mulBroadcast = F_->createNodeWithBroadcastOutTy<MulNode>(
      "mulBroadcast", axis, outTy, in1, in2);

  auto *divBroadcast = F_->createNodeWithBroadcastOutTy<DivNode>(
      "divBroadcast", axis, outTy, in1, in2);

  auto *saveAdd = F_->createSave("saveAdd", addBroadcast);
  auto *saveSub = F_->createSave("saveSub", subBroadcast);
  auto *saveMul = F_->createSave("saveMul", mulBroadcast);
  auto *saveDiv = F_->createSave("saveDiv", divBroadcast);

  bindings_.allocate(saveAdd->getPlaceholder());
  bindings_.allocate(saveSub->getPlaceholder());
  bindings_.allocate(saveMul->getPlaceholder());
  bindings_.allocate(saveDiv->getPlaceholder());

  auto Qin1H = bindings_.get(in1)->getHandle<int8_t>();
  auto Qin2H = bindings_.get(in2)->getHandle<int8_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultAdd =
      bindings_.get(saveAdd->getPlaceholder())->getHandle<int8_t>();
  auto resultSub =
      bindings_.get(saveSub->getPlaceholder())->getHandle<int8_t>();
  auto resultMul =
      bindings_.get(saveMul->getPlaceholder())->getHandle<int8_t>();
  auto resultDiv =
      bindings_.get(saveDiv->getPlaceholder())->getHandle<int8_t>();

  for (dim_t w = 0; w < W; w++) {
    float b = quantization::dequantize(Qin2H.at({w}), {in2Scale, in2Offset});
    for (dim_t n = 0; n < N; n++) {
      for (dim_t c = 0; c < C; c++) {
        for (dim_t h = 0; h < H; h++) {
          float a = quantization::dequantize(Qin1H.at({n, c, h, w}),
                                             {in1Scale, in1Offset});
          int8_t add = quantization::quantize((a + b), {outScale, outOffset});
          int8_t sub = quantization::quantize((a - b), {outScale, outOffset});
          int8_t mul = quantization::quantize((a * b), {outScale, outOffset});
          int8_t div = quantization::quantize((a / b), {outScale, outOffset});
          EXPECT_NEAR(add, resultAdd.at({n, c, h, w}), 1);
          EXPECT_NEAR(sub, resultSub.at({n, c, h, w}), 1);
          EXPECT_NEAR(mul, resultMul.at({n, c, h, w}), 1);
          EXPECT_NEAR(div, resultDiv.at({n, c, h, w}), 1);
        }
      }
    }
  }
}

/// Test Clip with Int8QTy.
TEST_P(OperatorTest, Clip_Int8) {
  CHECK_IF_ENABLED();

  std::vector<float> inputVals = {-3, -2, -1, 0, 1, 2, 3, 4};
  float clipMin = -2.0;
  float clipMax = 3.0;
  dim_t size = inputVals.size();
  const float inputScale = 1.0;
  const int32_t inputOffset = 5;
  const float outputScale = 0.5;
  const int32_t outputOffset = -3;

  auto *inputTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, inputScale, inputOffset);
  auto *outputTy =
      mod_.uniqueType(ElemKind::Int8QTy, {size}, outputScale, outputOffset);
  auto *input = mod_.createPlaceholder(inputTy, "input", false);
  auto *relu = F_->createClip("clip", input, outputTy, clipMin, clipMax);
  auto *dequantize =
      F_->createDequantize("dequantize", relu, ElemKind::FloatTy);
  auto *save = F_->createSave("save", dequantize);
  bindings_.allocate(mod_.getPlaceholders());

  auto inputH = bindings_.get(input)->getHandle<int8_t>();
  for (dim_t idx = 0; idx < size; idx++) {
    inputH.raw(idx) =
        quantization::quantize(inputVals[idx], {inputScale, inputOffset});
  }

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputH = bindings_.get(save->getPlaceholder())->getHandle();
  for (dim_t idx = 0; idx < size; idx++) {
    float expectedValue = std::min(clipMax, std::max(clipMin, inputVals[idx]));
    EXPECT_EQ(expectedValue, outputH.raw(idx));
  }
}

/// Verify quantized splats work correctly (add 0 to it to ensure constant
/// folding doesn't make this test meaningless).
TEST_P(OperatorTest, IntSplat) {
  CHECK_IF_ENABLED();

  const float splatValue = 10;
  const float scale = 1.0;
  const int32_t offset = 5;
  const dim_t size = 3;

  auto *in = mod_.createPlaceholder(ElemKind::Int8QTy, {size}, scale, offset,
                                    "in", false);
  auto splatTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, scale, offset);
  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *add = F_->createAdd("add", in, splat);
  auto *dequantize = F_->createDequantize("dequantize", add, ElemKind::FloatTy);
  auto *save = F_->createSave("save", dequantize);

  bindings_.allocate(in)->zero();
  auto resultH = bindings_.allocate(save->getPlaceholder())->getHandle();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < resultH.size(); i++) {
    EXPECT_EQ(splatValue, resultH.raw(i));
  }
}

/// Verify fp16 splats work correctly (add 0 to it to ensure constant
/// folding doesn't make this test meaningless).
TEST_P(OperatorTest, Fp16Splat) {
  CHECK_IF_ENABLED();

  const float splatValue = 10;
  const dim_t size = 3;

  auto *in = mod_.createPlaceholder(ElemKind::Float16Ty, {size}, "in", false);
  auto splatTy = mod_.uniqueType(ElemKind::Float16Ty, {size});
  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *add = F_->createAdd("add", in, splat);
  auto *save = F_->createSave("save", add);

  bindings_.allocate(in)->zero();
  auto resultH =
      bindings_.allocate(save->getPlaceholder())->getHandle<float16_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < resultH.size(); i++) {
    EXPECT_EQ(float16_t(splatValue), resultH.raw(i));
  }
}

/// Verify bfloat16 splats work correctly (add 0 to it to ensure constant
/// folding doesn't make this test meaningless).
TEST_P(OperatorTest, BFloat16Splat) {
  CHECK_IF_ENABLED();

  const float splatValue = 10;
  const dim_t size = 3;

  auto *in = mod_.createPlaceholder(ElemKind::BFloat16Ty, {size}, "in", false);
  auto splatTy = mod_.uniqueType(ElemKind::BFloat16Ty, {size});
  auto *splat = F_->createSplat("splat", splatTy, splatValue);
  auto *add = F_->createAdd("add", in, splat);
  auto *save = F_->createSave("save", add);

  bindings_.allocate(in)->zero();
  auto resultH =
      bindings_.allocate(save->getPlaceholder())->getHandle<bfloat16_t>();

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  for (dim_t i = 0; i < resultH.size(); i++) {
    EXPECT_EQ(bfloat16_t(splatValue), resultH.raw(i));
  }
}

// simple convTranspose. symmetric, no pads, strides or channels.
TEST_P(OperatorTest, sanityConvTranspose) {
  CHECK_IF_ENABLED();

  float biasVal[2] = {1.1, 2.2};
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {2., 3., 4., 5.};

  auto *filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 3, 1}, "filter", false);
  bindings_.allocate(filter)->getHandle() = {2., 3., 4.,  5., 6.,  7.,
                                             8., 9., 10., 3., 4.,  5.,
                                             6., 7., 8.,  9., 10., 11.};

  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(bias)->getHandle() = biasVal;

  std::pair<dim_t, dim_t> outWH =
      calculateConvTransposeOutputDims(2, 2, {3, 3}, {1, 1}, {0, 0, 0, 0});
  auto outTy =
      mod_.uniqueType(ElemKind::FloatTy, {1, outWH.first, outWH.second, 2});

  ConvTransposeNode *CN =
      F_->createConvTranspose("ConvTranspose", input, filter, bias, outTy,
                              {3, 3}, {1, 1}, {0, 0, 0, 0}, 1);

  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {1, 4, 4, 2};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 4 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 12 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 17 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3, 0}), 12 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 18 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1, 0}), 49 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2, 0}), 63 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3, 0}), 41 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 0}), 36 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 1, 0}), 91 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 2, 0}), 105 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 3, 0}), 65 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 0}), 32 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 1, 0}), 76 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 2, 0}), 85 + biasVal[0]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 3, 0}), 50 + biasVal[0]);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 6 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 17 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 22 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3, 1}), 15 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1}), 24 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1, 1}), 63 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2, 1}), 77 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3, 1}), 49 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 1}), 42 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 1, 1}), 105 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 2, 1}), 119 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 2, 3, 1}), 73 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 1}), 36 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 1, 1}), 85 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 2, 1}), 94 + biasVal[1]);
  EXPECT_FLOAT_EQ(result.at({0, 3, 3, 1}), 55 + biasVal[1]);
}

/// ConvTranspose with multi-channel input/output and asymmetric kernel,
/// strides, pads.
TEST_P(OperatorTest, ConvTransposedAsymmetric) {

  CHECK_IF_ENABLED();

  float biasVal[2] = {1.1, 2.2};
  auto bias = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(bias)->getHandle() = biasVal;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 3}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (dim_t i = 0; i < IH.size(); i++) {
    IH.raw(i) = i;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 3, 2, 3}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < FH.size(); i++) {
    FH.raw(i) = i * 2;
  }

  std::pair<dim_t, dim_t> outWH =
      calculateConvTransposeOutputDims(4, 4, {3, 2}, {1, 2}, {0, 3, 1, 2});
  auto outTy =
      mod_.uniqueType(ElemKind::FloatTy, {1, outWH.first, outWH.second, 2});

  ConvTransposeNode *CN =
      F_->createConvTranspose("ConvTranspose", input, filter, bias, outTy,
                              {3, 2}, {1, 2}, {0, 3, 1, 2}, 1);

  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto result = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {1, 5, 3, 2};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  // values from onnxruntime w/o bias, thus bias is added during compare.
  std::vector<float> expected = {
      100,  532,   46,   802,   172,  928,   632,  2792,  416,  3224,
      884,  3692,  2028, 7212,  1542, 7698,  2568, 8724,  4188, 13260,
      3054, 13098, 4728, 14772, 5096, 12440, 4232, 12224, 5564, 13556};
  for (dim_t i = 0; i < result.size(); i++) {
    float exp = expected[i] + biasVal[i % 2];
    EXPECT_FLOAT_EQ(result.raw(i), exp);
  }
}

/// ConvTranspose test with Group>1
TEST_P(OperatorTest, ConvTransposedGroup) {

  CHECK_IF_ENABLED();

  float biasVal[2] = {0, 0};
  auto bias = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(bias)->getHandle() = biasVal;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 2}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 9.,  1., 10., 2., 11.,
                                            3., 12., 4., 13., 5., 14.,
                                            6., 15., 7., 16., 8., 17.};

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 2}, "filter", false);
  bindings_.allocate(filter)->getHandle() = {
      0., 8., 2., 10., 4., 12., 6., 14,
  };

  std::pair<dim_t, dim_t> outWH =
      calculateConvTransposeOutputDims(3, 3, {2, 2}, {2, 2}, {0, 0, 0, 0});
  auto outTy =
      mod_.uniqueType(ElemKind::FloatTy, {1, outWH.first, outWH.second, 2});

  ConvTransposeNode *CN =
      F_->createConvTranspose("ConvTranspose", input, filter, bias, outTy,
                              {2, 2}, {2, 2}, {0, 0, 0, 0}, 2, 1);

  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  auto result = bindings_.get(S->getPlaceholder())->getHandle();
  std::vector<dim_t> expectedDims = {1, 6, 6, 2};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  std::vector<float> expected = {
      0,   72,  0,   90,  0,   80,  2,   100, 0,   88,  4,   110, 0,   108, 0,
      126, 4,   120, 6,   140, 8,   132, 12,  154, 0,   96,  6,   120, 0,   104,
      8,   130, 0,   112, 10,  140, 12,  144, 18,  168, 16,  156, 24,  182, 20,
      168, 30,  196, 0,   120, 12,  150, 0,   128, 14,  160, 0,   136, 16,  170,
      24,  180, 36,  210, 28,  192, 42,  224, 32,  204, 48,  238};

  for (dim_t i = 0; i < result.size(); i++) {
    EXPECT_FLOAT_EQ(result.raw(i), expected[i]);
  }
}

/// Compare ConvTranspose with equivalent Convolution, no strides.
/// TODO - need version with Strides (dilate input).
template <unsigned_t kernel, unsigned_t stride, unsigned_t pad, unsigned_t idim>
static void convTransposeConvCompare(glow::PlaceholderBindings &bindings,
                                     glow::Module &mod, glow::Function *F,
                                     glow::ExecutionEngine &EE) {
  unsigned_t Cpad = kernel - pad - 1;
  llvm::SmallVector<unsigned_t, 4> pads = {pad, pad, pad, pad};
  llvm::SmallVector<unsigned_t, 4> Cpads = {Cpad, Cpad, Cpad, Cpad};
  llvm::SmallVector<unsigned_t, 2> kernels = {kernel, kernel};
  llvm::SmallVector<unsigned_t, 2> strides = {stride, stride};

  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {1, idim, idim, 1},
                                      "input", false);
  bindings.allocate(input)->getHandle().randomize(-10.0, 10.0, mod.getPRNG());

  auto *filterConv = mod.createPlaceholder(
      ElemKind::FloatTy, {1, kernel, kernel, 1}, "filterC", false);
  bindings.allocate(filterConv)
      ->getHandle()
      .randomize(-10.0, 10.0, mod.getPRNG());
  auto FCH = bindings.get(filterConv)->getHandle();

  auto *filterConvTr = mod.createPlaceholder(
      ElemKind::FloatTy, {1, kernel, kernel, 1}, "filterD", false);
  auto FDH = bindings.allocate(filterConvTr)->getHandle();
  for (dim_t i = 0; i < kernel * kernel; i++) {
    FDH.raw(i) = FCH.raw(kernel * kernel - i - 1);
  }

  auto *bias = mod.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings.allocate(bias)->zero();

  std::pair<dim_t, dim_t> outHW =
      calculateConvTransposeOutputDims(idim, idim, kernels, strides, pads);

  auto outTy =
      mod.uniqueType(ElemKind::FloatTy, {1, outHW.first, outHW.second, 1});

  ConvolutionNode *CN = F->createConv("conv", input, filterConv, bias, outTy,
                                      kernels, strides, Cpads, 1, 1);
  ConvTransposeNode *DN =
      F->createConvTranspose("ConvTranspose", input, filterConvTr, bias, outTy,
                             kernels, strides, pads, 1, 1);

  SaveNode *SC = F->createSave("saveC", CN);
  bindings.allocate(SC->getPlaceholder());

  SaveNode *SD = F->createSave("saveD", DN);
  bindings.allocate(SD->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(
      F, bindings, {input, SC->getPlaceholder(), SD->getPlaceholder()});
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  outHW = calculateConvPoolOutputDims(idim, idim, kernels, strides, Cpads, 1);

  auto resultConv = bindings.get(SC->getPlaceholder())->getHandle();
  auto resultConvTranspose = bindings.get(SD->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, outHW.first, outHW.second, 1};
  ASSERT_TRUE(resultConv.dims().vec() == expectedDims);
  ASSERT_TRUE(resultConvTranspose.dims().vec() == expectedDims);

  for (dim_t i = 0; i < outHW.first; i++) {
    for (dim_t j = 0; j < outHW.second; j++) {
      EXPECT_FLOAT_EQ(static_cast<float>(resultConv.at({0, i, j, 0})),
                      static_cast<float>(resultConvTranspose.at({0, i, j, 0})));
    }
  }
}

TEST_P(OperatorTest, ConvTransposeonvolutionCompareSimpleK8S1P0I3) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  convTransposeConvCompare<8, 1, 0, 3>(bindings_, mod_, F_, EE_);
}

TEST_P(OperatorTest, ConvTransposeConvolutionCompareSimpleK6S1P1I4) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  convTransposeConvCompare<6, 1, 1, 4>(bindings_, mod_, F_, EE_);
}

TEST_P(OperatorTest, ConvTransposeConvolutionCompareSimpleK5S1P2I3) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  convTransposeConvCompare<5, 1, 2, 3>(bindings_, mod_, F_, EE_);
}

TEST_P(OperatorTest, GroupConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 8}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (dim_t i = 0; i < 2 * 8; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {6, 1, 1, 4}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 6; i++)
    for (dim_t j = 0; j < 4; j++) {
      FH.at({i, 0, 0, j}) = pow(10.0, i);
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 1, 6});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 2, 1, 6};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 1 + 2 + 3 + 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), (1 + 2 + 3 + 4) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), (1 + 2 + 3 + 4) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), (5 + 6 + 7 + 8) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 4}), (5 + 6 + 7 + 8) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 5}), (5 + 6 + 7 + 8) * 100000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 9 + 10 + 11 + 12);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1}), (9 + 10 + 11 + 12) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 2}), (9 + 10 + 11 + 12) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 3}), (13 + 14 + 15 + 16) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 4}), (13 + 14 + 15 + 16) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 5}), (13 + 14 + 15 + 16) * 100000);
}

/// Utility function to test numerically the ChannelwiseQuantizedConvolution2D
/// against a floating point Convolution for different parameters.
static void testChannelwiseQuantizedConv2D(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, quantization::Schema schema, ElemKind elemQKind,
    ElemKind biasElemQKind, bool filterFloat, bool biasFloat,
    bool biasScalesExplicit) {

  std::vector<dim_t> inputDims = {5, 10, 10, 4};
  std::vector<dim_t> filterDims = {8, 3, 3, 2};
  std::vector<dim_t> biasDims = {8};
  std::vector<dim_t> outputDims = {5, 6, 6, 8};
  std::vector<unsigned_t> kernels = {3, 3};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {0, 0, 0, 0};
  dim_t group = 2;
  dim_t dilation = 2;
  dim_t qDim = 0;
  dim_t qStep = 1;

  // Create input placeholder.
  auto *inputF =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "inputF", false);
  bindings.allocate(inputF)->getHandle<float>().randomize(-1.0, 1.0,
                                                          mod.getPRNG());

  // Quantize input.
  auto inputTQP =
      quantization::chooseQuantizationParams({-1.0, 1.0}, schema, elemQKind);
  auto *inputQTy =
      mod.uniqueType(elemQKind, inputDims, inputTQP.scale, inputTQP.offset);
  auto *inputQ = F->createQuantize("inputQ", inputF, inputQTy);

  // Create float filter constant.
  auto *filterF = mod.createConstant(ElemKind::FloatTy, filterDims, "filterF");
  filterF->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                            mod.getPRNG());

  // Create float bias constant.
  auto *biasF = mod.createConstant(ElemKind::FloatTy, biasDims, "biasF");
  biasF->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                          mod.getPRNG());

  // Create quantized filter and filterScales/filterOffsets constants for
  // ChannelwiseQuantizedConvolution.
  dim_t numChannels = outputDims[3];
  Constant *filterQ =
      mod.createConstant(elemQKind, filterDims, 1.0, 0, "filterQ");
  Constant *filterScales =
      mod.createConstant(ElemKind::FloatTy, {numChannels}, "filterScales");
  Constant *filterOffsets =
      mod.createConstant(ElemKind::Int32ITy, {numChannels}, "filterOffsets");
  quantization::getTensorQuantizationParams(
      filterF->getPayload(), filterScales->getPayloadMutable(),
      filterOffsets->getPayloadMutable(), schema, elemQKind, qDim, qStep);
  filterQ->getPayloadMutable() = quantization::quantizeTensor(
      filterF->getPayload(), filterScales->getPayload(),
      filterOffsets->getPayload(), elemQKind, qDim, qStep);

  // Create quantized bias and biasScales/biasOffsets constants for
  // ChannelwiseQuantizedConvolution.
  Constant *biasQ =
      mod.createConstant(biasElemQKind, {numChannels}, 1.0, 0, "biasQ");
  Constant *biasScales =
      mod.createConstant(ElemKind::FloatTy, {numChannels}, "biasScales");
  Constant *biasOffsets =
      mod.createConstant(ElemKind::Int32ITy, {numChannels}, "biasOffsets");
  auto biasScalesH = biasScales->getPayload().getHandle<float>();
  auto biasOffsetsH = biasOffsets->getPayload().getHandle<int32_t>();
  auto filterScalesH = filterScales->getPayload().getHandle<float>();
  auto filterOffsetsH = filterOffsets->getPayload().getHandle<int32_t>();
  auto inputScale = inputQ->getResult().getType()->getScale();
  auto inputOffset = inputQ->getResult().getType()->getOffset();
  if (biasScalesExplicit) {
    quantization::getTensorQuantizationParams(
        biasF->getPayload(), biasScales->getPayloadMutable(),
        biasOffsets->getPayloadMutable(), schema, biasElemQKind, qDim, qStep);
    for (dim_t idx = 0; idx < numChannels; idx++) {
      auto biasTQPNew = specializeBiasQuantizationParams(
          {biasScalesH.raw(idx), biasOffsetsH.raw(idx)},
          {inputScale, inputOffset},
          {filterScalesH.raw(idx), filterOffsetsH.raw(idx)}, schema,
          biasElemQKind);
      biasScalesH.raw(idx) = biasTQPNew.scale;
      biasOffsetsH.raw(idx) = biasTQPNew.offset;
    }
  } else {
    for (dim_t idx = 0; idx < numChannels; idx++) {
      float filterScale = filterScalesH.raw(idx);
      biasScalesH.raw(idx) = inputScale * filterScale;
      biasOffsetsH.raw(idx) = 0;
    }
  }
  biasQ->getPayloadMutable() = quantization::quantizeTensor(
      biasF->getPayload(), biasScales->getPayload(), biasOffsets->getPayload(),
      biasElemQKind, qDim, qStep);

  // Get optimal output TQP based on inspecting the output range for the
  // particular values of the convolution parameters. If the convolution
  // sizes are changed than these parameters must be adjusted.
  auto outputTQP =
      quantization::chooseQuantizationParams({-6.0, 6.0}, schema, elemQKind);
  auto *outQTy =
      mod.uniqueType(elemQKind, outputDims, outputTQP.scale, outputTQP.offset);

  // Prepare parameters for ChannelwiseQuantizedConvolutionNode.
  Constant *filterCWQ = nullptr;
  Constant *filterScalesCWQ = nullptr;
  Constant *filterOffsetsCWQ = nullptr;
  if (filterFloat) {
    filterCWQ = filterF;
  } else {
    filterCWQ = filterQ;
    filterScalesCWQ = filterScales;
    filterOffsetsCWQ = filterOffsets;
  }
  Constant *biasCWQ = nullptr;
  Constant *biasScalesCWQ = nullptr;
  Constant *biasOffsetsCWQ = nullptr;
  if (biasFloat) {
    biasCWQ = biasF;
  } else {
    biasCWQ = biasQ;
  }
  if (biasScalesExplicit) {
    biasScalesCWQ = biasScales;
    biasOffsetsCWQ = biasOffsets;
  }

  // Create ChannelwiseQuantizedConvolution and Dequantize.
  ChannelwiseQuantizedConvolutionNode *outQ = F->createChannelwiseQuantizedConv(
      "CWQConv", inputQ, filterCWQ, biasCWQ, filterScalesCWQ, filterOffsetsCWQ,
      biasScalesCWQ, biasOffsetsCWQ, outQTy, kernels, strides, pads, group,
      dilation, /* quantizeFilter */ true, /* quantizeBias */ true, schema,
      elemQKind, biasElemQKind);
  DequantizeNode *out =
      F->createDequantize("dequantize", outQ, ElemKind::FloatTy);
  SaveNode *saveOut = F->createSave("saveOut", out);
  bindings.allocate(saveOut->getPlaceholder());

  // Create reference floating-point Convolution.
  auto *refTy = mod.uniqueType(ElemKind::FloatTy, outputDims);
  ConvolutionNode *ref = F->createConv("Conv", inputF, filterF, biasF, refTy,
                                       kernels, strides, pads, group, dilation);
  SaveNode *saveRef = F->createSave("saveRef", ref);
  bindings.allocate(saveRef->getPlaceholder());

  // Compile and run.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Extra validations.
  EXPECT_EQ(F->getNodes().size(), 6);
  EXPECT_EQ(outQ->getFilter().getElementType(), elemQKind);
  EXPECT_EQ(outQ->getBias().getElementType(), biasElemQKind);

  // Check error. If bias is carefully quantized then the bias precision does
  // not matter and so the error tolerance is the same.
  auto outH = bindings.get(saveOut->getPlaceholder())->getHandle();
  auto refH = bindings.get(saveRef->getPlaceholder())->getHandle();
  for (dim_t idx = 0; idx < refH.size(); idx++) {
    float errVal = std::abs(refH.raw(idx) - outH.raw(idx));
    EXPECT_TRUE(errVal < 0.05);
  }
}

#define TEST_CWQCONV(testName, ...)                                            \
  TEST_P(OperatorTest, testName) {                                             \
    CHECK_IF_ENABLED();                                                        \
    testChannelwiseQuantizedConv2D(bindings_, mod_, F_, EE_,                   \
                                   quantization::Schema::Asymmetric,           \
                                   __VA_ARGS__);                               \
  }

/// These unit tests prove that the bias quantization for low precision (Int8)
/// requires a special handling because if we provide a quantized bias with
/// implicit quantization parameters biasScales[i] = inputScale*filterScales[i]
/// and biasOffsets[i]=0 does not work numerically due to BIAS DATA saturation.
/// Therefore in the unit tests below we do not use the *_*FF tests.
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt8_FFT, ElemKind::Int8QTy,
             ElemKind::Int8QTy, false, false, true)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt8_FTF, ElemKind::Int8QTy,
             ElemKind::Int8QTy, false, true, false)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt8_FTT, ElemKind::Int8QTy,
             ElemKind::Int8QTy, false, true, true)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt8_TFT, ElemKind::Int8QTy,
             ElemKind::Int8QTy, true, false, true)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt8_TTF, ElemKind::Int8QTy,
             ElemKind::Int8QTy, true, true, false)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt8_TTT, ElemKind::Int8QTy,
             ElemKind::Int8QTy, true, true, true)

/// These unit tests prove that the bias quantization for high precision (Int32)
/// can work without a special handling (implicit quantization parameters).
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFF, ElemKind::Int8QTy,
             ElemKind::Int32QTy, false, false, false)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFT, ElemKind::Int8QTy,
             ElemKind::Int32QTy, false, false, true)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_FTF, ElemKind::Int8QTy,
             ElemKind::Int32QTy, false, true, false)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_FTT, ElemKind::Int8QTy,
             ElemKind::Int32QTy, false, true, true)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_TFF, ElemKind::Int8QTy,
             ElemKind::Int32QTy, true, false, false)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_TFT, ElemKind::Int8QTy,
             ElemKind::Int32QTy, true, false, true)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_TTF, ElemKind::Int8QTy,
             ElemKind::Int32QTy, true, true, false)
TEST_CWQCONV(ChannelwiseQuantizedConv2D_Int8_BiasInt32_TTT, ElemKind::Int8QTy,
             ElemKind::Int32QTy, true, true, true)
#undef TEST_CWQCONV

/// Utility function to test numerically the ChannelwiseQuantizedConvolution2D
/// against Interpreter implementation.
static FunctionTensorPair
createAndInitBasicChannelwiseConv2DTest(glow::PlaceholderBindings &bindings,
                                        glow::ExecutionEngine &EE) {

  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  std::vector<dim_t> inputDims = {5, 10, 10, 4};
  std::vector<dim_t> filterDims = {8, 3, 3, 2};
  std::vector<dim_t> biasDims = {8};
  std::vector<dim_t> outputDims = {5, 6, 6, 8};
  std::vector<unsigned_t> kernels = {3, 3};
  std::vector<unsigned_t> strides = {1, 1};
  std::vector<unsigned_t> pads = {0, 0, 0, 0};
  dim_t group = 2;
  dim_t dilation = 2;

  // Create input placeholder.
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, inputDims, "input", false);
  bindings.allocate(input)->getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());

  // Create filter constant.
  auto *filter = mod.createConstant(ElemKind::FloatTy, filterDims, "filter");
  filter->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                           mod.getPRNG());

  // Create bias constant.
  auto *bias = mod.createConstant(ElemKind::FloatTy, biasDims, "bias");
  bias->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                         mod.getPRNG());

  // Create Convolution.
  auto *outTy = mod.uniqueType(ElemKind::FloatTy, outputDims);
  ConvolutionNode *conv =
      F->createConv("Conv", input, filter, bias, outTy, kernels, strides, pads,
                    group, dilation);
  SaveNode *save = F->createSave("save", conv);
  auto *outputTensor = bindings.allocate(save->getPlaceholder());
  return std::make_pair(F, outputTensor);
}

/// Test Int8 ChannelwiseQuantizedConvolution2D with Int8 bias.
TEST_P(OperatorStatelessTest, ChannelwiseQuantizedConv2D_Int8_BiasInt8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicChannelwiseConv2DTest,
      ElemKind::FloatTy, ElemKind::Int8QTy, 0.05f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int8QTy,
      /* forceFP16AccumSLS */ false,
      PrecisionConfiguration::Float16Format::None,
      /* convertToChannelwiseQuantization */ true);
}

/// Test Int8 ChannelwiseQuantizedConvolution2D with Int32 bias.
TEST_P(OperatorStatelessTest, ChannelwiseQuantizedConv2D_Int8_BiasInt32) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicChannelwiseConv2DTest,
      ElemKind::FloatTy, ElemKind::Int8QTy, 0.05f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      quantization::Schema::Asymmetric, ElemKind::Int32QTy,
      /* forceFP16AccumSLS */ false,
      PrecisionConfiguration::Float16Format::None,
      /* convertToChannelwiseQuantization */ true);
}

/// Test the functionality of channelwise quantized group convolution using
/// ChannelwiseQuantizedConvNode.
TEST_P(OperatorTest, ChannelwiseQuantizedConv2D) {
  CHECK_IF_ENABLED();

  constexpr size_t groups = 2;
  constexpr dim_t output_channel = 4;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle<float>();
  for (size_t i = 0; i < 2 * 3 * 2; i++) {
    IH.raw(i) = i + 1;
  }

  auto *qInTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 2, 3, 2}, 1.0, 0);
  auto *qInput = F_->createQuantize("qInput", input, qInTy);

  auto filterT = Tensor(ElemKind::Int8QTy, {4, 2, 1, 1}, 1.0, 0);
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 1; k++) {
        for (dim_t l = 0; l < 1; l++) {
          filterT.getHandle<int8_t>().at({i, j, k, l}) = j + 1;
        }
      }
    }
  }
  auto *filter = mod_.createConstant("filter", std::move(filterT));

  auto biasT = Tensor(ElemKind::FloatTy, {4});
  biasT.zero();
  auto *bias = mod_.createConstant("bias", std::move(biasT));

  auto filterScalesT = Tensor(ElemKind::FloatTy, {output_channel});
  for (size_t i = 0; i < filterScalesT.size(); i++) {
    filterScalesT.getHandle<float>().raw(i) = 1;
  }
  auto *filterScales =
      mod_.createConstant("filterScales", std::move(filterScalesT));

  auto filterOffsetsT = Tensor(ElemKind::Int32ITy, {output_channel});
  filterOffsetsT.zero();
  auto *filterOffsets =
      mod_.createConstant("filterOffsets", std::move(filterOffsetsT));

  auto *outTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 1, 3, 4}, 1.0, 0);
  ChannelwiseQuantizedConvolutionNode *CQC = F_->createChannelwiseQuantizedConv(
      "channelwiseQuantizedConv", qInput, filter, bias, filterScales,
      filterOffsets, /* biasScales */ nullptr, /* biasOffsets */ nullptr, outTy,
      {2, 1}, {1, 1}, {0, 0, 0, 0}, groups);

  DequantizeNode *dq =
      F_->createDequantize("dequantize", CQC, ElemKind::FloatTy);
  SaveNode *S = F_->createSave("save", dq);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 1, 3, 4};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 15);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 15);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 21);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 21);

  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 2}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 3}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 27);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 27);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 2}), 30);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 3}), 30);
}

/// Test the functionality of channelwise quantized group convolution using
/// ChannelwiseQuantizedConvNode.
TEST_P(OperatorTest, ChannelwiseQuantizedConv3D) {
  CHECK_IF_ENABLED();

  constexpr size_t groups = 2;
  constexpr dim_t output_channel = 4;
  constexpr dim_t input_channel = 2;

  auto *input = mod_.createPlaceholder(
      ElemKind::FloatTy, {1, input_channel, 2, 3, 2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle<float>();
  for (size_t i = 0; i < input_channel * 2 * 3 * 2; i++) {
    IH.raw(i) = i + 1;
  }

  auto *qInTy =
      mod_.uniqueType(ElemKind::Int8QTy, {1, input_channel, 2, 3, 2}, 1.0, 0);
  auto *qInput = F_->createQuantize("qInput", input, qInTy);

  auto filterT = Tensor(
      ElemKind::Int8QTy,
      {output_channel / groups, input_channel / groups, 1, 1, 1}, 1.0, 0);
  for (dim_t i = 0; i < output_channel / groups; i++) {
    for (dim_t j = 0; j < input_channel / groups; j++) {
      for (dim_t t = 0; t < 1; t++) {
        for (dim_t k = 0; k < 1; k++) {
          for (dim_t l = 0; l < 1; l++) {
            filterT.getHandle<int8_t>().at({i, j, t, k, l}) = j + 1;
          }
        }
      }
    }
  }
  auto *filter = mod_.createConstant("filter", std::move(filterT));

  auto biasT = Tensor(ElemKind::FloatTy, {output_channel / groups});
  biasT.zero();
  auto *bias = mod_.createConstant("bias", std::move(biasT));

  auto scalesT = Tensor(ElemKind::FloatTy, {output_channel / groups});
  for (size_t i = 0; i < scalesT.size(); i++) {
    scalesT.getHandle<float>().raw(i) = 1;
  }
  auto *scales = mod_.createConstant("scales", std::move(scalesT));

  auto offsetsT = Tensor(ElemKind::Int32ITy, {output_channel / groups});
  offsetsT.zero();
  auto *offsets = mod_.createConstant("offsets", std::move(offsetsT));

  auto *outTy = mod_.uniqueType(ElemKind::Int8QTy,
                                {1, output_channel / groups, 2, 3, 2}, 1.0, 0);
  ChannelwiseQuantizedConvolutionNode *CQC = F_->createChannelwiseQuantizedConv(
      "channelwiseQuantizedConv", qInput, filter, bias, scales, offsets,
      /* biasScales */ nullptr, /* biasOffsets */ nullptr, outTy, {1, 1, 1},
      {1, 1, 1}, {0, 0, 0, 0, 0, 0}, groups);

  DequantizeNode *dq =
      F_->createDequantize("dequantize", CQC, ElemKind::FloatTy);
  SaveNode *S = F_->createSave("save", dq);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, output_channel / groups, 2, 3, 2};
  ASSERT_TRUE(result.dims().vec() == expectedDims);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 3);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), 5);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), 7);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 7);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 9);

  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 2}), 11);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 3}), 13);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 13);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 15);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 2}), 17);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 3}), 19);
}

TEST_P(OperatorTest, DilatedConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 1, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 3; i++)
    for (dim_t j = 0; j < 3; j++) {
      FH.at({0, i, j, 0}) = 1;
    }
  FH.at({0, 1, 1, 0}) = 0;

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 1, 1});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 3, 1, 2, 1, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 4, 1, 1};
  EXPECT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 3);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 4);
  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 0}), 1);
  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 0}), 2);
}

/// Test the functionality of channelwise quantized group convolution using
/// ChannelwiseQuantizedConvNode with non-zero offsets and biases.
void testChannelwiseQuantizedConv2DNonZero(glow::PlaceholderBindings &bindings,
                                           glow::Module &mod, glow::Function *F,
                                           glow::ExecutionEngine &EE,
                                           bool quantizeBias) {
  constexpr size_t groups = 2;
  constexpr dim_t output_channel = 4;

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 2}, "input", false);
  auto IH = bindings.allocate(input)->getHandle<float>();
  for (size_t i = 0; i < 2 * 3 * 2; i++) {
    IH.raw(i) = i + 1;
  }

  auto *qInTy = mod.uniqueType(ElemKind::Int8QTy, {1, 2, 3, 2}, 2.5, 3);
  auto *qInput = F->createQuantize("qInput", input, qInTy);

  auto filterT = Tensor(ElemKind::Int8QTy, {4, 2, 1, 1}, 1.0, 0);
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 1; k++) {
        for (dim_t l = 0; l < 1; l++) {
          filterT.getHandle<int8_t>().at({i, j, k, l}) = j + 1;
        }
      }
    }
  }
  auto *filter = mod.createConstant("filter", std::move(filterT));

  auto biasT = Tensor(ElemKind::FloatTy, {4});
  for (dim_t i = 0; i < 4; i++) {
    biasT.getHandle<float>().raw(i) = i + 1;
  }
  auto *bias = mod.createConstant("bias", std::move(biasT));

  auto filterScalesT = Tensor(ElemKind::FloatTy, {output_channel});
  for (size_t i = 0; i < filterScalesT.size(); i++) {
    filterScalesT.getHandle<float>().raw(i) = 1;
  }
  auto *filterScales =
      mod.createConstant("filterScales", std::move(filterScalesT));

  auto filterOffsetsT = Tensor(ElemKind::Int32ITy, {output_channel});
  filterOffsetsT.zero();

  auto *filterOffsets =
      mod.createConstant("filterOffsets", std::move(filterOffsetsT));

  auto *outTy = mod.uniqueType(ElemKind::Int8QTy, {1, 1, 3, 4}, 2, 2);
  ChannelwiseQuantizedConvolutionNode *CQC = F->createChannelwiseQuantizedConv(
      "channelwiseQuantizedConv", qInput, filter, bias, filterScales,
      filterOffsets, /* biasScales */ nullptr, /* biasOffsets */ nullptr, outTy,
      {2, 1}, {1, 1}, {0, 0, 0, 0}, groups, /* dilation */ 1,
      /* quantizeFilter */ false, quantizeBias);

  DequantizeNode *dq =
      F->createDequantize("dequantize", CQC, ElemKind::FloatTy);
  SaveNode *S = F->createSave("save", dq);
  bindings.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, S->getPlaceholder()});

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 1, 3, 4};
  ASSERT_TRUE(result.dims().vec() == expectedDims);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 16);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 2}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 3}), 22);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 22);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 26);

  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 2}), 28);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 3}), 30);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 26);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 28);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 2}), 32);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 3}), 36);
}

TEST_P(OperatorTest, ChannelwiseQuantizedConv2D_NonZero_FloatBias) {
  CHECK_IF_ENABLED();
  testChannelwiseQuantizedConv2DNonZero(bindings_, mod_, F_, EE_,
                                        /* quantizeBias */ false);
}

TEST_P(OperatorTest, ChannelwiseQuantizedConv2D_NonZero_QuantizedBias) {
  CHECK_IF_ENABLED();
  testChannelwiseQuantizedConv2DNonZero(bindings_, mod_, F_, EE_,
                                        /* quantizeBias */ true);
}

TEST_P(OperatorTest, GroupDilatedConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (dim_t i = 0; i < 4 * 4 * 2; i++) {
    IH.raw(i) = i;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 2; i++)
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 2; k++) {
        FH.at({i, j, k, 0}) = 1;
      }
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 4, 2});

  ConvolutionNode *CN =
      F_->createConv("Conv", input, filter, zeroBias, outTy, 2, 1, 1, 2, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 4, 4, 2};
  ASSERT_TRUE(result.dims().vec() == expectedDims);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0}), 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1}), 11);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 0}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 0, 1, 1}), 22);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 0}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 0, 2, 1}), 26);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3, 0}), 12);
  EXPECT_FLOAT_EQ(result.at({0, 0, 3, 1}), 13);

  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1}), 22);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1, 0}), 40);
  EXPECT_FLOAT_EQ(result.at({0, 1, 1, 1}), 44);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2, 0}), 48);
  EXPECT_FLOAT_EQ(result.at({0, 1, 2, 1}), 52);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3, 0}), 24);
  EXPECT_FLOAT_EQ(result.at({0, 1, 3, 1}), 26);

  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 0}), 36);
  EXPECT_FLOAT_EQ(result.at({0, 2, 0, 1}), 38);
  EXPECT_FLOAT_EQ(result.at({0, 2, 1, 0}), 72);
  EXPECT_FLOAT_EQ(result.at({0, 2, 1, 1}), 76);
  EXPECT_FLOAT_EQ(result.at({0, 2, 2, 0}), 80);
  EXPECT_FLOAT_EQ(result.at({0, 2, 2, 1}), 84);
  EXPECT_FLOAT_EQ(result.at({0, 2, 3, 0}), 40);
  EXPECT_FLOAT_EQ(result.at({0, 2, 3, 1}), 42);

  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 0}), 18);
  EXPECT_FLOAT_EQ(result.at({0, 3, 0, 1}), 19);
  EXPECT_FLOAT_EQ(result.at({0, 3, 1, 0}), 36);
  EXPECT_FLOAT_EQ(result.at({0, 3, 1, 1}), 38);
  EXPECT_FLOAT_EQ(result.at({0, 3, 2, 0}), 40);
  EXPECT_FLOAT_EQ(result.at({0, 3, 2, 1}), 42);
  EXPECT_FLOAT_EQ(result.at({0, 3, 3, 0}), 20);
  EXPECT_FLOAT_EQ(result.at({0, 3, 3, 1}), 21);
}

/// Test Conv3D with group size of 2 to make sure that group 3d convolution
/// works as expected.
TEST_P(OperatorTest, GroupConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 1, 2, 8},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < input->getType()->size(); i++) {
    IH.raw(i) = i + 1;
  }

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {6, 1, 1, 1, 4},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 6; i++)
    for (dim_t j = 0; j < 4; j++) {
      FH.at({i, 0, 0, 0, j}) = pow(10.0, i);
    }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 1, 2, 6});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, 1, 1, 0, 2);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle();

  std::vector<dim_t> expectedDims = {1, 2, 1, 2, 6};
  ASSERT_TRUE(result.dims().vec() == expectedDims);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 0}), 1 + 2 + 3 + 4);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 1}), (1 + 2 + 3 + 4) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 2}), (1 + 2 + 3 + 4) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 3}), (5 + 6 + 7 + 8) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 4}), (5 + 6 + 7 + 8) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 0, 5}), (5 + 6 + 7 + 8) * 100000);

  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 0}), 9 + 10 + 11 + 12);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 1}), (9 + 10 + 11 + 12) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 2}), (9 + 10 + 11 + 12) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 3}), (13 + 14 + 15 + 16) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 4}), (13 + 14 + 15 + 16) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 0, 0, 1, 5}), (13 + 14 + 15 + 16) * 100000);

  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 0}), 17 + 18 + 19 + 20);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 1}), (17 + 18 + 19 + 20) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 2}), (17 + 18 + 19 + 20) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 3}), (21 + 22 + 23 + 24) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 4}), (21 + 22 + 23 + 24) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 0, 5}), (21 + 22 + 23 + 24) * 100000);

  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 0}), 25 + 26 + 27 + 28);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 1}), (25 + 26 + 27 + 28) * 10);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 2}), (25 + 26 + 27 + 28) * 100);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 3}), (29 + 30 + 31 + 32) * 1000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 4}), (29 + 30 + 31 + 32) * 10000);
  EXPECT_FLOAT_EQ(result.at({0, 1, 0, 1, 5}), (29 + 30 + 31 + 32) * 100000);
}

/// Check non-square padding for convolution. The first conv has non-square
/// padding, while the second one has zero padding. The second conv's input is
/// the same as the first one's after-padding input. All other parameters of
/// the two convs are the same.
TEST_P(OperatorTest, NonSquarePaddingConvolution) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input",
                                       false, "NHWC");
  auto IH = bindings_.allocate(input)->getHandle();
  for (dim_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 1},
                                       "filter", false, "NHWC");
  auto FH = bindings_.allocate(filter)->getHandle();
  for (dim_t i = 0; i < 2 * 2 * 2; i++) {
    FH.raw(i) = pow(2.0, i);
  }
  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 8, 2});

  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {1, 1}, {0, 2, 1, 3}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Create the reference conv operator whose input is the same as the
  // after-padding-input above.
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1},
                                        "input1", false, "NHWC");
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  for (dim_t i = 0; i < 4; i++)
    for (dim_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  CN = refF->createConv("Conv1", input1, filter, zeroBias, outTy, {2, 2},
                        {1, 1}, {0, 0, 0, 0}, 1);
  S = refF->createSave("save1", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(refF, bindings_,
                                         {input, input1, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");
  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-cubic padding for conv3D. The first conv3D has non-cubic
/// padding, while the second one has zero padding. The second conv3D's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two conv3Ds are the same.
TEST_P(OperatorTest, NonCubicPaddingConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // W
    }   // H
  }     // T

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2, 2, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (size_t i = 0; i < filter->getType()->size(); i++) {
    FH.raw(i) = pow(2.0, i);
  }
  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {2}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 12, 4, 8, 2});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, {2, 2, 2},
                       {1, 1, 1}, // {0, 2, 5, 1, 3, 4},
                       {5, 4, 0, 1, 2, 3}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});

  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Create the reference conv3D operator whose input is the same as the
  // after-padding-input above.
  auto *input1 = mod_.createPlaceholder(ElemKind::FloatTy, {1, 13, 5, 9, 1},
                                        "input1", false);
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  nextVal = 1;
  for (dim_t i = 5; i < 9; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 2; k < 6; k++) {
        IH1.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // W
    }   // H
  }     // T

  Function *refF = mod_.createFunction("mainRef");
  CN = refF->createConv3D("Conv3D_1", input1, filter, zeroBias, outTy,
                          {2, 2, 2}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  S = refF->createSave("save1", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(refF, bindings_,
                                         {input, input1, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");
  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// 2D Batch Normalization in Float16
TEST_P(OperatorTest, FP16BatchNorm2D) {
  CHECK_IF_ENABLED();

  auto constFunc = [=](unsigned_t sz, std::string name,
                       std::vector<float> vals) {
    auto t = Tensor(ElemKind::Float16Ty, {sz});
    for (dim_t i = 0; i < sz; i++) {
      t.getHandle<float16_t>().raw(i) = vals[i];
    }
    auto *c = mod_.createConstant(name, std::move(t));
    return c;
  };

  // input
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 1, 3, 3}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {1., 1., 1., 1., 1.,
                                                       1., 1., 1., 1.};
  auto *bias = constFunc(1, "batchnorm_bias", {0.0});
  auto *scale = constFunc(1, "batchnorm_weights", {1.0});
  auto *mean = constFunc(1, "running_mean", {0.0});
  auto *variance = constFunc(1, "running_var", {1.0});
  unsigned_t channelIdx = 1;
  float epsilon = 0.0;
  float momentum = 1.0;

  auto *op = F_->createBatchNormalization("fp16_batch_norm2d", input, bias,
                                          scale, mean, variance, channelIdx,
                                          epsilon, momentum);
  auto *S = F_->createSave("save", op);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 1, 3, 3});
  out.getHandle<float16_t>() = {1., 1., 1., 1., 1., 1., 1., 1., 1.};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Check non-square padding for AveragePool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool op's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two convs are the same.
TEST_P(OperatorTest, NonSquarePaddingAveragePool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  Tensor &result = *bindings_.get(S->getPlaceholder());

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  for (dim_t i = 0; i < 4; i++)
    for (dim_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createAvgPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");
  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

/// Check non-square padding for MaxPool. The first pool op has non-square
/// padding, while the second one has zero padding. The second pool-op's input
/// is the same as the first one's after-padding input. All other parameters
/// of the two convs are the same.
TEST_P(OperatorTest, NonSquarePaddingMaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 2, 1, 3});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  Tensor &result = *bindings_.get(S->getPlaceholder());

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 9, 1}, "input1", false);
  bindings_.allocate(input1)->zero();
  auto IH1 = bindings_.get(input1)->getHandle();
  for (dim_t i = 0; i < 4; i++)
    for (dim_t j = 2; j < 6; j++) {
      IH1.at({0, i, j, 0}) = i * 4 + j - 2 + 1;
    }

  Function *refF = mod_.createFunction("mainRef");
  Pool = refF->createMaxPool("pool1", input1, 2, 1, 0);
  S = refF->createSave("save1", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_, "main");
  EE_.run(bindings_, "mainRef");

  Tensor &result1 = *bindings_.get(S->getPlaceholder());

  EXPECT_TRUE(result.isEqual(result1));
}

TEST_P(OperatorTest, FP16AvgPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {0., 1., 2., 3., 4.,
                                                       5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 2, 2, 1});
  out.getHandle<float16_t>() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, BFloat16AvgPool) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1},
                                       "input", false);
  bindings_.allocate(input)->getHandle<bfloat16_t>() = {0., 1., 2., 3., 4.,
                                                        5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::BFloat16Ty, {1, 2, 2, 1});
  out.getHandle<bfloat16_t>() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AvgPool operator works correctly.
TEST_P(OperatorTest, AvgPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, Int8AvgPool) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {2, 3, 5, 6};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(OperatorTest, Int8AvgPoolCountExcludePads) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createAvgPool("pool", input, {3, 3}, {2, 2}, {1, 1, 1, 1},
                                 NHWC, /* countIncludePads */ false);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {2, 3, 5, 6};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(OperatorTest, FP16AvgPool3D) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 3, 1}, // NCTHW
                             "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 0., 1., 2., 3., 4.,
      5., 6., 7., 8., 0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {2, 2, 2}, // kernel
                                 {1, 1, 1},                // stride
                                 {0, 0, 0, 0, 0, 0},       // padding
                                 NTHWC);
  auto *outputNCTHW =
      F_->createTranspose("avgpool3d_output_NTHWC2NCTHW", Pool, NTHWC2NCTHW);
  auto *S = F_->createSave("save", outputNCTHW);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 1, 2, 2, 2});
  out.getHandle<float16_t>() = {2., 3., 5., 6., 2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, BFloat16AvgPool3D) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 1, 3, 3, 3}, // NCTHW
                             "input", false);
  bindings_.allocate(input)->getHandle<bfloat16_t>() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 0., 1., 2., 3., 4.,
      5., 6., 7., 8., 0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *inputNTHWC =
      F_->createTranspose("avgpool3d_input_NCTHW2NTHWC", input, NCTHW2NTHWC);
  auto *Pool = F_->createAvgPool("pool", inputNTHWC, {2, 2, 2}, // kernel
                                 {1, 1, 1},                     // stride
                                 {0, 0, 0, 0, 0, 0},            // padding
                                 NTHWC);
  auto *outputNCTHW =
      F_->createTranspose("avgpool3d_output_NTHWC2NCTHW", Pool, NTHWC2NCTHW);
  auto *S = F_->createSave("save", outputNCTHW);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::BFloat16Ty, {1, 1, 2, 2, 2});
  out.getHandle<bfloat16_t>() = {2., 3., 5., 6., 2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, Int8AvgPool3D) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {1, 1, 3, 3, 3}, // NCTHW
                             1, 0, // scale, offset
                             "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                    0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                    0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *inputNTHWC =
      F_->createTranspose("avgpool3d_input_NCTHW2NTHWC", input, NCTHW2NTHWC);
  auto *Pool = F_->createAvgPool("avgpool3d", inputNTHWC, {2, 2, 2}, // kernel
                                 {1, 1, 1},                          // stride
                                 {0, 0, 0, 0, 0, 0},                 // padding
                                 NTHWC);
  auto *outputNCTHW =
      F_->createTranspose("avgpool3d_output_NTHWC2NCTHW", Pool, NTHWC2NCTHW);
  auto *S = F_->createSave("save", outputNCTHW);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {1, 1, 2, 2, 2}, 1, 0);
  out.getHandle<int8_t>() = {
      2, 3, 5, 6, 2, 3, 5, 6,
  };
  for (size_t i = 0; i < 2 * 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

TEST_P(OperatorTest, AvgPoolCountExcludePads) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *Pool = F_->createAvgPool("pool", input, {3, 3}, {2, 2}, {1, 1, 1, 1},
                                 NHWC, /* countIncludePads */ false);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {2., 3., 5., 6.};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly.
TEST_P(OperatorTest, AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 3, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 3, 3, 1});
  out.getHandle() = {2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with fp16.
TEST_P(OperatorTest, FP16AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 4, 4, 1}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};
  auto outTy = mod_.uniqueType(ElemKind::Float16Ty, {1, 3, 3, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 3, 3, 1});
  out.getHandle<float16_t>() = {2.5, 3.5, 4.5, 6.5, 7.5, 8.5, 10.5, 11.5, 12.5};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with bfloat16.
TEST_P(OperatorTest, BFloat16AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input = mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 4, 4, 1},
                                       "input", false);
  bindings_.allocate(input)->getHandle<bfloat16_t>() = {
      0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.};
  auto outTy = mod_.uniqueType(ElemKind::BFloat16Ty, {1, 3, 3, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::BFloat16Ty, {1, 3, 3, 1});
  out.getHandle<bfloat16_t>() = {2.5, 3.5,  4.5,  6.5, 7.5,
                                 8.5, 10.5, 11.5, 12.5};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with int8.
TEST_P(OperatorTest, Int8AdaptiveAvgPool) {
  CHECK_IF_ENABLED();
  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 4, 4, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0);
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0);
  out.getHandle<int8_t>() = {3, 4, 5, 7, 8, 9, 11, 12, 13};
  EXPECT_TRUE(out.isEqual(*result));
}

/// Verify that the AdaptiveAvgPool operator works correctly with non-square
/// inputs and outputs.
TEST_P(OperatorTest, AdaptiveAvgPoolNonSquare) {
  CHECK_IF_ENABLED();
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 5, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2.,  3.,  4.,  5.,  6., 7.,
                                            8., 9., 10., 11., 12., 13., 14.};

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 2, 1});
  auto *pool = F_->createAdaptiveAvgPool("pool", input, outTy);
  auto *S = F_->createSave("save", pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto *result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 3, 2, 1});
  out.getHandle() = {2, 3, 6.5, 7.5, 11, 12};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, MaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle() = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 2, 2, 1});
  out.getHandle() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, FP16MaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 3, 3, 1}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {0., 1., 2., 3., 4.,
                                                       5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 2, 2, 1});
  out.getHandle<float16_t>() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, BFloat16MaxPool) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 3, 3, 1},
                                       "input", false);
  bindings_.allocate(input)->getHandle<bfloat16_t>() = {0., 1., 2., 3., 4.,
                                                        5., 6., 7., 8.};
  auto *pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::BFloat16Ty, {1, 2, 2, 1});
  out.getHandle<bfloat16_t>() = {4., 5., 7., 8.};
  EXPECT_TRUE(out.isEqual(*result));
}

TEST_P(OperatorTest, Int8MaxPool) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::Int8QTy, {1, 3, 3, 1}, 1, 0,
                                       "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<int8_t>();
  Tensor out(ElemKind::Int8QTy, {2, 2}, 1, 0);
  out.getHandle<int8_t>() = {4, 5, 7, 8};
  for (size_t i = 0; i < 2 * 2; i++) {
    EXPECT_EQ(result.raw(i), out.getHandle<int8_t>().raw(i));
  }
}

#define COMPARE_UNARY_OP_FUN(_OP_NAME_, LEN, LOW, HIGH)                        \
  static FunctionTensorPair createAndInitBasic##_OP_NAME_##Test(               \
      glow::PlaceholderBindings &bindings, glow::ExecutionEngine &EE) {        \
    auto &mod = EE.getModule();                                                \
    Function *F = mod.createFunction("main");                                  \
                                                                               \
    auto *input =                                                              \
        mod.createPlaceholder(ElemKind::FloatTy, {LEN}, "input", false);       \
    bindings.allocate(input)->getHandle().randomize(LOW, HIGH, mod.getPRNG()); \
    auto *tanh = F->create##_OP_NAME_(#_OP_NAME_, input);                      \
    auto *save = F->createSave("Save", tanh);                                  \
    auto *resultTensor = bindings.allocate(save->getPlaceholder());            \
    return std::make_pair(F, resultTensor);                                    \
  }
COMPARE_UNARY_OP_FUN(Exp, 10, -1.0F, 1.0F)
COMPARE_UNARY_OP_FUN(Tanh, 10, -10.0F, 10.0F)
COMPARE_UNARY_OP_FUN(Log, 1000, 1.0F, 100.0F)
COMPARE_UNARY_OP_FUN(Sigmoid, 10, -10.0F, 10.0F)
#undef COMPARE_UNARY_OP_FUN

/// Reference ideal sigmoid implementation. Computes an fp32 sigmoid
/// and casts the result to FP16.
static float16_t refSigmoidFp16(float x) {
  float res = 1 / (1 + exp(-x));

  return (float16_t)res;
}

/// Reference ideal sigmoid implementation. Computes an fp32 sigmoid
/// and casts the result to BFloat16.
static bfloat16_t refSigmoidBFloat16(float x) {
  float res = 1 / (1 + exp(-x));

  return (bfloat16_t)res;
}

/// Test to verify that the sigmoid implementation is equal to the
/// Mirrored LUT implementation
/// Does a sweep of -15,15 and prints the outputs of the NNPI implementation
/// compared to the LUT one, the ideal sigmoid in fp16 is also provided as
/// a visual sanity check, but nothing is enforced against that last one.
static void testSigmoidFp16Sweep(glow::PlaceholderBindings &bindings,
                                 glow::Module &mod, glow::Function *F,
                                 glow::ExecutionEngine &EE) {
  constexpr dim_t N = 100;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {N}, "input", false);
  auto inputH = bindings.allocate(input)->getHandle();

  constexpr float rangeStart = -20;
  constexpr float rangeEnd = 20;
  constexpr float delta = (rangeEnd - rangeStart) / N;

  for (dim_t i = 0; i < N; i++) {
    inputH.raw(i) = rangeStart + i * delta;
  }

  auto *sigmoid = F->createSigmoid("Sigmoid", input);
  auto *save = F->createSave("Save", sigmoid);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::FP16;

  EE.compile(cctx);
  EE.run(bindings);

  auto resultH = resultTensor->getHandle();
  int numDiffs = 0;

  for (dim_t i = 0; i < N; i++) {
    float inputV = inputH.at({i});
    float refIdeal = refSigmoidFp16(inputV);
    float output = resultH.at({i});
    float absDiff = fabs(output - refIdeal);
    float relDiff = fabs(absDiff / (refIdeal + 1e-8));

    bool failed = false;
    // Relative error should be 2^-11 but we are relaxing this constraint
    // due to linear interpolation
    // Absolute error can remain 1e-5 for now
    if (absDiff > 1e-5 && relDiff > 2e-3) {
      numDiffs++;
      failed = true;
    }

    llvm::outs() << "Sigmoid " << i << " " << inputV << " Backend:" << output
                 << " ref_ideal:" << refIdeal << " relDiff:" << relDiff
                 << " absDiff:" << absDiff << " failed:" << failed << "\n";
  }
  llvm::outs() << "Number of diffs: " << numDiffs << "\n";
  llvm::outs().flush();

  EXPECT_EQ(numDiffs, 0);
}

/// Test to verify that the sigmoid implementation is equal to the
/// Mirrored LUT implementation
/// Does a sweep of -15,15 and prints the outputs of the NNPI implementation
/// compared to the LUT one, the ideal sigmoid in bfloat16 is also provided as
/// a visual sanity check, but nothing is enforced against that last one.
static void testSigmoidBFloat16Sweep(glow::PlaceholderBindings &bindings,
                                     glow::Module &mod, glow::Function *F,
                                     glow::ExecutionEngine &EE) {
  constexpr dim_t N = 100;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {N}, "input", false);
  auto inputH = bindings.allocate(input)->getHandle();

  constexpr float rangeStart = -20;
  constexpr float rangeEnd = 20;
  constexpr float delta = (rangeEnd - rangeStart) / N;

  for (dim_t i = 0; i < N; i++) {
    inputH.raw(i) = rangeStart + i * delta;
  }

  auto *sigmoid = F->createSigmoid("Sigmoid", input);
  auto *save = F->createSave("Save", sigmoid);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::BFloat16;

  EE.compile(cctx);
  EE.run(bindings);

  auto resultH = resultTensor->getHandle();
  int numDiffs = 0;

  for (dim_t i = 0; i < N; i++) {
    float inputV = inputH.at({i});
    float refIdeal = refSigmoidBFloat16(inputV);
    float output = resultH.at({i});
    float absDiff = fabs(output - refIdeal);
    float relDiff = fabs(absDiff / (refIdeal + 1e-8));

    bool failed = false;
    // Relative error should be 2^-11 but we are relaxing this constraint
    // due to linear interpolation.
    // Absolute error can remain 1e-5 for now
    if (absDiff > 1e-3 && relDiff > 2e-2) {
      numDiffs++;
      failed = true;
    }

    llvm::outs() << "Sigmoid " << i << " " << inputV << " Backend:" << output
                 << " ref_ideal:" << refIdeal << " relDiff:" << relDiff
                 << " absDiff:" << absDiff << " failed:" << failed << "\n";
  }
  llvm::outs() << "Number of diffs: " << numDiffs << "\n";
  llvm::outs().flush();

  EXPECT_EQ(numDiffs, 0);
}

TEST_P(OperatorTest, SigmoidSweep_Float16) {
  CHECK_IF_ENABLED();

  testSigmoidFp16Sweep(bindings_, mod_, F_, EE_);
}

TEST_P(OperatorTest, SigmoidSweep_BFloat16) {
  CHECK_IF_ENABLED();

  testSigmoidBFloat16Sweep(bindings_, mod_, F_, EE_);
}

/// Reference ideal tanh implementation. Computes an fp32 tanh
/// and casts the result to FP16, no denorms
static float16_t refTanHFp16(float x) {
  float res = (exp(2 * x) - 1) / (exp(2 * x) + 1);
  if (fabs(res) < 6e-5) {
    res = 0.0;
  }
  return (float16_t)res;
}

/// Reference ideal tanh implementation. Computes an fp32 tanh
/// and casts the result to BFloat16, no denorms
static bfloat16_t refTanHBFloat16(float x) {
  float res = (exp(2 * x) - 1) / (exp(2 * x) + 1);
  if (fabs(res) < 6e-5) {
    res = 0.0;
  }
  return (bfloat16_t)res;
}

/// Test to verify that the tanh implementation is close to the ideal one
/// Does a sweep of -15,15 and prints the outputs of the NNPI implementation
/// compared to the ideal tanh in fp16.
static void testTanHFp16Sweep(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE) {
  constexpr dim_t N = 100;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {N}, "input", false);
  auto inputH = bindings.allocate(input)->getHandle();

  constexpr float rangeStart = -15;
  constexpr float rangeEnd = 15;
  constexpr float delta = (rangeEnd - rangeStart) / N;

  for (dim_t i = 0; i < N; i++) {
    inputH.raw(i) = rangeStart + i * delta;
  }

  auto *sigmoid = F->createTanh("TanH", input);
  auto *save = F->createSave("Save", sigmoid);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::FP16;

  EE.compile(cctx);
  EE.run(bindings);

  auto resultH = resultTensor->getHandle();
  int count = 0;

  for (dim_t i = 0; i < N; i++) {
    float inputV = inputH.at({i});
    float refIdeal = refTanHFp16(inputV);
    float output = resultH.at({i});
    float diff = fabs(output - refIdeal);

    if (diff > 1e-6) {
      count++;
    }

    llvm::outs() << "TanH " << i << " " << inputV << " Backend:" << output
                 << " ref_ideal:" << refIdeal << " diff:" << diff << "\n";
  }
  llvm::outs().flush();

  EXPECT_EQ(count, 0);
}

/// Test to verify that the tanh implementation is close to the ideal one
/// Does a sweep of -15,15 and prints the outputs of the NNPI implementation
/// compared to the ideal tanh in fp16.
static void testTanHBFloat16Sweep(glow::PlaceholderBindings &bindings,
                                  glow::Module &mod, glow::Function *F,
                                  glow::ExecutionEngine &EE) {
  constexpr dim_t N = 100;
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {N}, "input", false);
  auto inputH = bindings.allocate(input)->getHandle();

  constexpr float rangeStart = -15;
  constexpr float rangeEnd = 15;
  constexpr float delta = (rangeEnd - rangeStart) / N;

  for (dim_t i = 0; i < N; i++) {
    inputH.raw(i) = rangeStart + i * delta;
  }

  auto *sigmoid = F->createTanh("TanH", input);
  auto *save = F->createSave("Save", sigmoid);
  auto *resultTensor = bindings.allocate(save->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::BFloat16;

  EE.compile(cctx);
  EE.run(bindings);

  auto resultH = resultTensor->getHandle();
  int count = 0;

  for (dim_t i = 0; i < N; i++) {
    float inputV = inputH.at({i});
    float refIdeal = refTanHBFloat16(inputV);
    float output = resultH.at({i});
    float diff = fabs(output - refIdeal);

    if (diff > 1e-2) {
      count++;
    }

    llvm::outs() << "TanH " << i << " " << inputV << " Backend:" << output
                 << " ref_ideal:" << refIdeal << " diff:" << diff << "\n";
  }
  llvm::outs().flush();

  EXPECT_EQ(count, 0);
}

TEST_P(OperatorTest, TanHSweep_Float16) {
  CHECK_IF_ENABLED();

  testTanHFp16Sweep(bindings_, mod_, F_, EE_);
}

TEST_P(OperatorTest, TanHSweep_BFloat16) {
  CHECK_IF_ENABLED();

  testTanHBFloat16Sweep(bindings_, mod_, F_, EE_);
}

template <typename DataType>
static void testMaxPoolWithArgmax(glow::PlaceholderBindings &bindings,
                                  glow::Module &mod, glow::Function *F,
                                  glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *input = createPlaceholderConditionallyQuantized(mod, DTy, {1, 3, 3, 1},
                                                        "input", false, "NHWC");
  bindings.allocate(input)->getHandle<DataType>() = {0, 3, 7, 6, 5, 1, 2, 8, 4};
  auto *pool = F->createMaxPool("pool", input, {2, 2}, {1, 1}, {0, 0, 0, 0});
  auto *SResult = F->createSave("save_result", pool->getResult());
  auto *SArgmax = F->createSave("save_argmax", pool->getArgmax());
  bindings.allocate(SResult->getPlaceholder());
  bindings.allocate(SArgmax->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(SResult->getPlaceholder());
  auto argmax = bindings.get(SArgmax->getPlaceholder());
  Tensor out1 = createTensorConditionallyQuantized(DTy, {1, 2, 2, 1});
  out1.getHandle<DataType>() = {6, 7, 8, 8};
  EXPECT_TRUE(out1.isEqual(*result));

  Tensor out2(ElemKind::Int64ITy, {1, 2, 2, 1});
  out2.getHandle<int64_t>() = {3, 2, 7, 7};
  EXPECT_TRUE(out2.isEqual(*argmax));
}

TEST_P(OperatorTest, FloatMaxPoolWithArgmax) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmax<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedMaxPoolWithArgmax) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmax<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

template <typename DataType>
static void
testMaxPoolWithArgmaxTransposed(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind DTy) {
  // Show that sequence Tensor(NCHW) -> Transpose(NCHWtoNHWC) ->
  // MaxPoolWithArgmax -> Transpose(NHWCtoNCHW) produces correct
  // linearization.
  auto *inputNCHW = createPlaceholderConditionallyQuantized(
      mod, DTy, {1, 3, 4, 4}, "input", false, "NCHW");
  auto inHandle = bindings.allocate(inputNCHW)->getHandle<DataType>();
  inHandle.clear(0.);
  inHandle.at({0, 0, 2, 2}) = 11;
  inHandle.at({0, 1, 2, 2}) = 22;
  inHandle.at({0, 2, 2, 2}) = 33;

  // Input NCHW to NHWC conversion.
  auto *inputNHWC =
      F->createTranspose("transposeInput", inputNCHW, {0, 2, 3, 1}, "NHWC");
  auto *pool =
      F->createMaxPool("pool", inputNHWC, {4, 4}, {4, 4}, {0, 0, 0, 0});

  // NHWC to NCHW conversion.
  auto *resultNCHW = F->createTranspose("transposeRes", pool->getResult(),
                                        {0, 3, 1, 2}, "NCHW");
  auto *argmaxNCHW = F->createTranspose("transposeArgmax", pool->getArgmax(),
                                        {0, 3, 1, 2}, "NCHW");

  auto *SResult = F->createSave("save_result", resultNCHW);
  auto *SArgmax = F->createSave("save_argmax", argmaxNCHW);
  bindings.allocate(SResult->getPlaceholder());
  bindings.allocate(SArgmax->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto result = bindings.get(SResult->getPlaceholder());
  auto argmax = bindings.get(SArgmax->getPlaceholder());
  Tensor out1 = createTensorConditionallyQuantized(DTy, {1, 3, 1, 1});
  out1.getHandle<DataType>() = {11, 22, 33};
  EXPECT_TRUE(out1.isEqual(*result));

  Tensor out2(ElemKind::Int64ITy, {1, 3, 1, 1});
  out2.getHandle<int64_t>() = {0 + 2 * 3 + 2 * 12, 1 + 2 * 3 + 2 * 12,
                               2 + 2 * 3 + 2 * 12};
  EXPECT_TRUE(out2.isEqual(*argmax));
}

TEST_P(OperatorTest, FloatMaxPoolWithArgmaxTransposed) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmaxTransposed<float>(bindings_, mod_, F_, EE_,
                                         ElemKind::FloatTy);
}

TEST_P(OperatorTest, QuantizedMaxPoolWithArgmaxTransposed) {
  CHECK_IF_ENABLED();
  testMaxPoolWithArgmaxTransposed<int8_t>(bindings_, mod_, F_, EE_,
                                          ElemKind::Int8QTy);
}

TEST_P(OperatorStatelessTest, Int8Tanh) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicTanhTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.005f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Tanh_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicTanhTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.001f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Tanh_BFloat16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicTanhTest,
                            ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.001f,
                            parCloneCountOpt);
}

/// Verify that the Tanh operator works correctly.
TEST_P(OperatorTest, Tanh) {
  CHECK_IF_ENABLED();

  constexpr dim_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  bindings_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *tanh = F_->createTanh("Tanh", input);
  auto *save = F_->createSave("Save", tanh);
  bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(save->getPlaceholder())->getHandle();
  auto inputH = bindings_.get(input)->getHandle();

  for (dim_t i = 0; i < size; i++) {
    EXPECT_NEAR(resultH.at({i}), std::tanh(inputH.at({i})), 0.001);
  }
}

TEST_P(OperatorStatelessTest, Exp_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicExpTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.005f,
                            parCloneCountOpt);
}

TEST_P(OperatorStatelessTest, Exp_BFloat16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicExpTest,
                            ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.005f,
                            parCloneCountOpt);
}

/// Verify that the Exp operator works correctly.
TEST_P(OperatorTest, Exp) {
  CHECK_IF_ENABLED();
  constexpr dim_t size = 10;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  bindings_.allocate(input)->getHandle().randomize(-10.0, 10.0, mod_.getPRNG());

  auto *expn = F_->createExp("Exp", input);
  auto *save = F_->createSave("Save", expn);
  bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(save->getPlaceholder())->getHandle();
  auto inputH = bindings_.get(input)->getHandle();

  for (dim_t i = 0; i < size; i++) {
    EXPECT_NEAR(resultH.at({i}), std::exp(inputH.at({i})), 0.001);
  }
}

/// Verify that a quantized Log works correctly.
TEST_P(OperatorStatelessTest, Int8Log) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicLogTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.1f,
                            parCloneCountOpt);
}

/// Check Non-square kernel for conv.
TEST_P(OperatorTest, NonSquareKernelConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 3, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 1 * 2 * 3; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 3, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 3}, {1, 1}, {0, 0, 0, 0}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {106, 127, 190, 211, 274, 295};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-cubic kernel for conv3D.
TEST_P(OperatorTest, NonCubicKernelConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 1; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 3; k++) {
        FH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 3, 2, 1});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, {1, 2, 3},
                       {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {106, 127, 190,  211,  274,  295,  442,  463,
                              526, 547, 610,  631,  778,  799,  862,  883,
                              946, 967, 1114, 1135, 1198, 1219, 1282, 1303};
  for (size_t i = 0; i < 4 * 3 * 2; i++) {
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
  }
}

/// Check Non-cubic kernel for conv3D with quantized input, filters, and bias.
TEST_P(OperatorTest, NonCubicKernelConv3DQuantized) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto qInType = mod_.uniqueType(ElemKind::Int16QTy, {1, 4, 4, 4, 1}, 0.1, 0);
  QuantizeNode *qInput = F_->createQuantize("q_input", input, qInType);

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 1; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 3; k++) {
        FH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // D
    }   // W
  }     // H

  auto qFilterType =
      mod_.uniqueType(ElemKind::Int16QTy, {1, 1, 2, 3, 1}, 0.1, 0);
  QuantizeNode *qFilter = F_->createQuantize("q_filter", filter, qFilterType);

  auto *bias = mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(bias)->zero();

  auto qBiasType = mod_.uniqueType(ElemKind::Int32QTy, {1}, 0.1, 0);
  QuantizeNode *qBias = F_->createQuantize("q_bias", bias, qBiasType);

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 4, 3, 2, 1});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, bias, outTy, {1, 2, 3},
                       {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);

  auto qOutTy = mod_.uniqueType(ElemKind::Int16QTy, {1, 4, 3, 2, 1}, 0.1, 0);

  Convolution3DNode *qCN =
      F_->createConv3D("q_Conv3D", qInput, qFilter, qBias, qOutTy, {1, 2, 3},
                       {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);

  SaveNode *S = F_->createSave("save", CN);

  DequantizeNode *deQ =
      F_->createDequantize("deQ_result", qCN, ElemKind::FloatTy);
  SaveNode *qS = F_->createSave("save", deQ);

  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  bindings_.allocate(mod_.getPlaceholders());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor &qResult = *bindings_.get(qS->getPlaceholder());

  for (size_t i = 0; i < 4 * 3 * 2; i++) {
    EXPECT_NEAR(qResult.getHandle().raw(i), result.getHandle().raw(i), 0.5);
  }
}

/// Test for quantized Convolution3D.
static void Conv3DQuantizedTest(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind elemKind,
                                ElemKind biaselemKind) {
  // Create floating-point network.
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1}, "input", false);
  auto *filter = mod.createPlaceholder(ElemKind::FloatTy, {1, 1, 2, 3, 1},
                                       "filter", false);
  auto *bias = mod.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  auto outTy = mod.uniqueType(ElemKind::FloatTy, {1, 4, 3, 2, 1});
  Convolution3DNode *conv3d =
      F->createConv3D("Conv3D", input, filter, bias, outTy, {1, 2, 3},
                      {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  SaveNode *save = F->createSave("save", conv3d);

  // Quantized types.
  auto inputTQP = quantization::chooseQuantizationParams(
      {-1.0, 1.0}, quantization::Schema::Asymmetric, elemKind);
  auto filterTQP = quantization::chooseQuantizationParams(
      {-1.0, 1.0}, quantization::Schema::Asymmetric, elemKind);
  auto outputTQP = quantization::chooseQuantizationParams(
      {-4.0, 4.0}, quantization::Schema::Asymmetric, elemKind);

  // Create quantized network.
  auto inputQTy = mod.uniqueType(elemKind, {1, 4, 4, 4, 1}, inputTQP.scale,
                                 inputTQP.offset);
  auto filterQTy = mod.uniqueType(elemKind, {1, 1, 2, 3, 1}, filterTQP.scale,
                                  filterTQP.offset);
  auto outQTy = mod.uniqueType(elemKind, {1, 4, 3, 2, 1}, outputTQP.scale,
                               outputTQP.offset);
  QuantizeNode *inputQ = F->createQuantize("inputQ", input, inputQTy);
  QuantizeNode *filterQ = F->createQuantize("filterQ", filter, filterQTy);
  Convolution3DNode *conv3dQ = nullptr;
  if (biaselemKind == ElemKind::FloatTy) {
    conv3dQ = F->createConv3D("Conv3DQ", inputQ, filterQ, bias, outQTy,
                              {1, 2, 3}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  } else {
    auto biasTQP = quantization::chooseQuantizationParams(
        {-1.0, 1.0}, quantization::Schema::Asymmetric, biaselemKind);
    auto biasQTy =
        mod.uniqueType(biaselemKind, {1}, biasTQP.scale, biasTQP.offset);
    QuantizeNode *biasQ = F->createQuantize("biasQ", bias, biasQTy);
    conv3dQ = F->createConv3D("Conv3DQ", inputQ, filterQ, biasQ, outQTy,
                              {1, 2, 3}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}, 1);
  }
  DequantizeNode *deQ = F->createDequantize("deQ", conv3dQ, ElemKind::FloatTy);
  SaveNode *saveQ = F->createSave("saveQ", deQ);

  // Allocate placeholders.
  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(filter)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(bias)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(save->getPlaceholder());
  bindings.allocate(saveQ->getPlaceholder());

  // Run network.
  ::glow::convertPlaceholdersToConstants(
      F, bindings, {input, save->getPlaceholder(), saveQ->getPlaceholder()});
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compare.
  Tensor &res = *bindings.get(save->getPlaceholder());
  Tensor &resQ = *bindings.get(saveQ->getPlaceholder());
  for (size_t i = 0; i < res.size(); i++) {
    EXPECT_NEAR(res.getHandle().raw(i), resQ.getHandle().raw(i), 0.03);
  }
}

/// Test Int8 Conv3D with Int8 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int8_BiasInt8) {
  ENABLED_BACKENDS("Interpreter");
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                      ElemKind::Int8QTy);
}

/// Test Int8 Conv3D with Int32 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int8_BiasInt32) {
  ENABLED_BACKENDS("Interpreter", "NNPI");
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                      ElemKind::Int32QTy);
}

/// Test Int8 Conv3D with Float32 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int8_BiasFloat) {
  ENABLED_BACKENDS("Interpreter", "NNPI");
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy,
                      ElemKind::FloatTy);
}

/// Test Int16 Conv3D with Int16 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int16_BiasInt16) {
  ENABLED_BACKENDS("Interpreter");
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int16QTy,
                      ElemKind::Int16QTy);
}

/// Test Int16 Conv3D with Int32 bias.
TEST_P(OperatorTest, Conv3DQuantizedTest_Int16_BiasInt32) {
  ENABLED_BACKENDS("Interpreter");
  Conv3DQuantizedTest(bindings_, mod_, F_, EE_, ElemKind::Int16QTy,
                      ElemKind::Int32QTy);
}

/// Check Non-square kernel for AveragePool.
TEST_P(OperatorTest, NonSquareKernelAveragePool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {4, 5, 8, 9, 12, 13};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square kernel for MaxPool.
TEST_P(OperatorTest, NonSquareKernelMaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 3}, {1, 1}, {0, 0, 0, 0});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {7, 8, 11, 12, 15, 16};
  for (size_t i = 0; i < 6; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for conv.
TEST_P(OperatorTest, NonSquareStrideConvolution) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }

  auto filter =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 1}, "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  for (size_t i = 0; i < 1 * 2 * 2; i++) {
    FH.raw(i) = i + 1;
  }

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 2, 1});
  ConvolutionNode *CN = F_->createConv("Conv", input, filter, zeroBias, outTy,
                                       {2, 2}, {3, 2}, {0, 0, 1, 1}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {44, 64, 41, 47};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-cubic stride for conv3D.
TEST_P(OperatorTest, NonCubicStrideConv3D) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 4, 1},
                                       "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  int nextVal = 1;
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 4; k++) {
        IH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // W
    }   // H
  }     // T

  auto *filter = mod_.createPlaceholder(ElemKind::FloatTy, {1, 2, 2, 2, 1},
                                        "filter", false);
  auto FH = bindings_.allocate(filter)->getHandle();
  nextVal = 1;
  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 2; k++) {
        FH.at({0, i, j, k, 0}) = static_cast<float>(nextVal++);
      } // W
    }   // H
  }     // T

  auto *zeroBias =
      mod_.createPlaceholder(ElemKind::FloatTy, {1}, "bias", false);
  bindings_.allocate(zeroBias)->zero();

  auto outTy = mod_.uniqueType(ElemKind::FloatTy, {1, 2, 2, 2, 1});

  Convolution3DNode *CN =
      F_->createConv3D("Conv3D", input, filter, zeroBias, outTy, {2, 2, 2},
                       {3, 3, 2}, //{0, 0, 0, 1, 1, 1}, 1);
                       {0, 1, 0, 1, 0, 1}, 1);
  SaveNode *S = F_->createSave("save", CN);
  bindings_.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(F_, bindings_,
                                         {input, S->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {560, 632, 366, 394, 524, 544, 185, 191};
  for (size_t i = 0; i < 8; i++) {
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
  }
}

/// Check Non-square stride for AveragePool.
TEST_P(OperatorTest, NonSquareStrideAveragePool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createAvgPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {3.5, 5.5, 6.75, 7.75};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

/// Check Non-square stride for MaxPool.
TEST_P(OperatorTest, NonSquareStrideMaxPool) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 4, 4, 1}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  for (size_t i = 0; i < 4 * 4; i++) {
    IH.raw(i) = i + 1;
  }
  auto *Pool = F_->createMaxPool("pool", input, {2, 2}, {3, 2}, {0, 0, 1, 1});
  auto *S = F_->createSave("save", Pool->getResult());
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());

  static const float ref[] = {6, 8, 14, 16};
  for (size_t i = 0; i < 4; i++)
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
}

TEST_P(OperatorTest, SigmoidOverflow) {
  CHECK_IF_ENABLED();

  auto *input = mod_.createPlaceholder(ElemKind::FloatTy, {2}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH.raw(0) = 1000;
  IH.raw(1) = -1000;

  auto *fpSigmoid = F_->createSigmoid("fpSigmoid", input);
  auto *S = F_->createSave("fpSave", fpSigmoid);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());
  static const float ref[] = {1, 0};
  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(result.getHandle().raw(i), ref[i]);
  }
}

/// This unit test exposes a problem with the CPU Sigmoid when stacking a higher
/// number of operations for extreme input values which result in NaNs.
TEST_P(OperatorTest, SigmoidOverflowCPUStacking) {
  CHECK_IF_ENABLED();
  dim_t size = 20;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH = {
      -1588.409912109375,  -460.55999755859375, -1176.9149169921875,
      -1655.9249267578125, -1580.1217041015625, -1680.279541015625,
      -1750.2677001953125, -1762.1697998046875, -1616.599365234375,
      -1725.301025390625,  +1588.409912109375,  +460.55999755859375,
      +1176.9149169921875, +1655.9249267578125, +1580.1217041015625,
      +1680.279541015625,  +1750.2677001953125, +1762.1697998046875,
      +1616.599365234375,  +1725.301025390625,
  };
  auto *fpSigmoid = F_->createSigmoid("fpSigmoid", input);
  auto *S = F_->createSave("fpSave", fpSigmoid);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());
  for (size_t i = 0; i < size; i++) {
    float ref = IH.raw(i) > 0 ? 1 : 0;
    EXPECT_NEAR(result.getHandle().raw(i), ref, 1E-6);
  }
}

/// This unit test exposes a problem with the CPU Tanh when stacking a higher
/// number of operations for extreme input values which result in NaNs.
TEST_P(OperatorTest, TanhOverflowCPUStacking) {
  CHECK_IF_ENABLED();
  dim_t size = 20;
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {size}, "input", false);
  auto IH = bindings_.allocate(input)->getHandle();
  IH = {
      -1588.409912109375,  -460.55999755859375, -1176.9149169921875,
      -1655.9249267578125, -1580.1217041015625, -1680.279541015625,
      -1750.2677001953125, -1762.1697998046875, -1616.599365234375,
      -1725.301025390625,  +1588.409912109375,  +460.55999755859375,
      +1176.9149169921875, +1655.9249267578125, +1580.1217041015625,
      +1680.279541015625,  +1750.2677001953125, +1762.1697998046875,
      +1616.599365234375,  +1725.301025390625,
  };
  auto *fpTanh = F_->createTanh("fpTanh", input);
  auto *S = F_->createSave("fpSave", fpTanh);
  bindings_.allocate(S->getPlaceholder());
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor &result = *bindings_.get(S->getPlaceholder());
  for (size_t i = 0; i < size; i++) {
    float ref = IH.raw(i) > 0 ? 1 : -1;
    EXPECT_NEAR(result.getHandle().raw(i), ref, 1E-6);
  }
}

TEST_P(OperatorStatelessTest, Int8Sigmoid) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicSigmoidTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.005f,
                            parCloneCountOpt);
}

/// Check that the batch add operator works properly.
TEST_P(OperatorTest, BatchAdd) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {13, 3, 3}, "A", false);
  bindings_.allocate(input)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3}, "slice", false);
  bindings_.allocate(slice)->getHandle<float>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float>();
  auto handleInput = bindings_.get(input)->getHandle<float>();
  auto handleSlice = bindings_.get(slice)->getHandle<float>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

/// Check that the batch add operator works properly for FP16.
TEST_P(OperatorTest, FP16BatchAdd) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {13, 3, 3}, "A", false);
  bindings_.allocate(input)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::Float16Ty, {3, 3}, "slice", false);
  bindings_.allocate(slice)->getHandle<float16_t>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<float16_t>();
  auto handleInput = bindings_.get(input)->getHandle<float16_t>();
  auto handleSlice = bindings_.get(slice)->getHandle<float16_t>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

/// Check that the batch add operator works properly for BFloat16.
TEST_P(OperatorTest, BFloat16BatchAdd) {
  CHECK_IF_ENABLED();

  PseudoRNG PRNG;

  auto *input =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {13, 3, 3}, "A", false);
  bindings_.allocate(input)->getHandle<bfloat16_t>().randomize(-3.0, 3.0, PRNG);
  auto *slice =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {3, 3}, "slice", false);
  bindings_.allocate(slice)->getHandle<bfloat16_t>().randomize(-3.0, 3.0, PRNG);
  auto *batchAdd = F_->createBatchedAdd("batchAdd", input, slice);
  auto *S = F_->createSave("save", batchAdd);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder())->getHandle<bfloat16_t>();
  auto handleInput = bindings_.get(input)->getHandle<bfloat16_t>();
  auto handleSlice = bindings_.get(slice)->getHandle<bfloat16_t>();
  ASSERT_EQ(result.size(), handleInput.size());
  for (size_t idx = 0, end = result.size(); idx != end; ++idx) {
    EXPECT_EQ(result.raw(idx),
              handleInput.raw(idx) + handleSlice.raw(idx % handleSlice.size()));
  }
}

TEST_P(OperatorTest, BroadcastAdd2x) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {10, 1}, "input", false);
  auto *bias = mod_.createConstant(ElemKind::FloatTy, {1, 1}, "bias");
  bias->getPayloadMutable().getHandle() = {42};
  auto *tile = F_->createTile("tile", bias, 10, 0);
  auto *add = F_->createAdd("add", input, tile);
  auto *save = F_->createSave("save", add);
  auto *output = save->getPlaceholder();
  bindings_.allocate(input)->getHandle() = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  bindings_.allocate(output);
  EE_.compile(CompilationMode::Infer);
  for (int i = 0; i < 2; i++) {
    Tensor expected(ElemKind::FloatTy, {10, 1});
    expected.getHandle() = {42, 43, 44, 45, 46, 47, 48, 49, 50, 51};
    EE_.run(bindings_);
    EXPECT_TRUE(bindings_.get(output)->isEqual(expected));
  }
}

/// Helper to test Sigmoid using \p DTy.
template <typename DataType>
static void testSigmoid(glow::PlaceholderBindings &bindings, glow::Module &mod,
                        glow::Function *F, glow::ExecutionEngine &EE,
                        ElemKind DTy, float allowedError = 0.001f) {
  constexpr dim_t size = 10;
  auto *input = mod.createPlaceholder(DTy, {size}, "input", false);
  bindings.allocate(input)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());

  auto *sigmoid = F->createSigmoid("sigmoid", input);
  auto *save = F->createSave("Save", sigmoid);
  bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(save->getPlaceholder())->getHandle<DataType>();
  auto inH = bindings.get(input)->getHandle<DataType>();

  for (dim_t i = 0; i < size; i++) {
    float val = 1 / (1 + std::exp(-(float)inH.at({i})));
    EXPECT_NEAR(RH.at({i}), val, allowedError);
  }
}

/// Verify that the Sigmoid operator works correctly with FloatTy.
TEST_P(OperatorTest, Sigmoid_Float) {
  CHECK_IF_ENABLED();
  testSigmoid<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the Sigmoid operator works correctly with Float16Ty.
TEST_P(OperatorTest, Sigmoid_Float16) {
  CHECK_IF_ENABLED();
  testSigmoid<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Verify that the Sigmoid operator works correctly with BFloat16Ty.
TEST_P(OperatorTest, Sigmoid_BFloat16) {
  CHECK_IF_ENABLED();
  testSigmoid<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                          0.01f);
}

/// Helper to test Swish using \p DTy.
template <typename DataType>
static void testSwish(glow::PlaceholderBindings &bindings, glow::Module &mod,
                      glow::Function *F, glow::ExecutionEngine &EE,
                      ElemKind DTy, float allowedError = 0.006f) {
  constexpr dim_t size = 10;
  auto *input = mod.createPlaceholder(DTy, {size}, "input", false);
  bindings.allocate(input)->getHandle<DataType>().randomize(-5.0, 5.0,
                                                            mod.getPRNG());

  auto *swish = F->createSwish("swish", input);
  auto *save = F->createSave("Save", swish);
  bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(save->getPlaceholder())->getHandle<DataType>();
  auto inH = bindings.get(input)->getHandle<DataType>();

  for (dim_t i = 0; i < size; i++) {
    float x = (float)inH.at({i});
    float val = x / (1 + std::exp(-x));
    EXPECT_NEAR(RH.at({i}), val, allowedError);
  }
}

/// Verify that the Swish operator works correctly with FloatTy.
TEST_P(OperatorTest, Swish_Float) {
  CHECK_IF_ENABLED();
  testSwish<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Verify that the Swish operator works correctly with Float16Ty.
TEST_P(OperatorTest, Swish_Float16) {
  CHECK_IF_ENABLED();
  testSwish<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Verify that the Swish operator works correctly with BFloat16Ty.
TEST_P(OperatorTest, Swish_BFloat16) {
  CHECK_IF_ENABLED();
  testSwish<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, 0.2f);
}

/// Verify that the Swish operator works correctly with Int8QTy.
TEST_P(OperatorStatelessTest, Swish_Int8) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(
      getBackendName(),
      [](PlaceholderBindings &bindings, ExecutionEngine &EE) {
        Module &mod = EE.getModule();
        Function *F = mod.createFunction("main");
        Placeholder *input =
            mod.createPlaceholder(ElemKind::FloatTy, {500}, "input", false);
        bindings.allocate(input)->getHandle<float>().randomize(-5.0, 5.0,
                                                               mod.getPRNG());
        SwishNode *swish = F->createSwish("swish", input);
        SaveNode *save = F->createSave("Save", swish);
        Tensor *saveTensor = bindings.allocate(save->getPlaceholder());
        return std::make_pair(F, saveTensor);
      },
      ElemKind::FloatTy, ElemKind::Int8QTy, 0.035, parCloneCountOpt);
}

TEST_P(OperatorTest, IntLookupTable) {
  CHECK_IF_ENABLED();

  constexpr dim_t size = 6;
  auto *input =
      mod_.createPlaceholder(ElemKind::Int8QTy, {size}, 1, 0, "input", false);
  bindings_.allocate(input)->getHandle<int8_t>() = {0, 1, 2, 3, 4, 5};

  auto outTy = mod_.uniqueType(ElemKind::Int8QTy, {size}, 3, 3);

  // Mapping i -> i.
  std::vector<int8_t> initValues(256);
  for (size_t i = 0; i < 256; ++i) {
    initValues[i] = i - 128;
  }

  auto *lookupTable =
      F_->createIntLookupTable("lookupTable", input, initValues, outTy);
  auto *save = F_->createSave("save", lookupTable);
  bindings_.allocate(save->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(save->getPlaceholder())->getHandle<int8_t>();
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(result.raw(i), i);
  }
}

/// Helper to test BatchAdd using \p DTy.
template <typename DataType>
static void testBatchOp(glow::PlaceholderBindings &bindings, glow::Module &mod,
                        glow::Function *F, glow::ExecutionEngine &EE,
                        ElemKind DTy, const std::string &opName) {
  CHECK(opName == "add" || opName == "mul") << "Invalid opName: " << opName;

  constexpr unsigned numSlices = 10;
  constexpr unsigned batchSize = 3;
  auto *input = mod.createPlaceholder(DTy, {batchSize * numSlices, 10, 10},
                                      "input", false);
  auto *slice = mod.createPlaceholder(DTy, {10, 10}, "slice", false);

  bindings.allocate(input)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());
  bindings.allocate(slice)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());

  std::vector<NodeValue> ops;
  for (dim_t i = 0; i < numSlices; i++) {
    auto *ex = F->createSlice("slice", input, {i * batchSize, 0, 0},
                              {(i + 1) * batchSize, 10, 10});
    if (opName == "add") {
      ops.push_back(F->createBatchedAdd("add", ex, slice)->getResult());
    } else {
      ops.push_back(F->createBatchedMul("mul", ex, slice)->getResult());
    }
  }

  auto *cc = F->createConcat("concat", ops, 0);

  // Remove the reference to the graph nodes to allow DCE to remove them.
  ops.clear();

  auto *result = F->createSave("save", cc);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(result->getPlaceholder())->getHandle<DataType>();
  auto IH = bindings.get(input)->getHandle<DataType>();
  auto SH = bindings.get(slice)->getHandle<DataType>();

  // Check that batched add works as expected.
  for (dim_t i = 0; i < numSlices * batchSize; i++) {
    for (dim_t j = 0; j < 10; j++) {
      for (dim_t k = 0; k < 10; k++) {
        if (opName == "add") {
          EXPECT_NEAR(IH.at({i, j, k}) + SH.at({j, k}), RH.at({i, j, k}),
                      0.00001);
        } else {
          EXPECT_NEAR(IH.at({i, j, k}) * SH.at({j, k}), RH.at({i, j, k}),
                      0.00001);
        }
      }
    }
  }
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(OperatorTest, testBatchAdd_Float) {
  CHECK_IF_ENABLED();
  testBatchOp<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, "add");
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(OperatorTest, testBatchAdd_Float16) {
  CHECK_IF_ENABLED();
  testBatchOp<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, "add");
}

/// Check that the sequence of extract-batchedadd-concat works.
TEST_P(OperatorTest, testBatchAdd_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchOp<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                          "add");
}

/// Check that the sequence of extract-batchedmul-concat works.
TEST_P(OperatorTest, testBatchMul_Float) {
  CHECK_IF_ENABLED();
  testBatchOp<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, "mul");
}

/// Check that the sequence of extract-batchedmul-concat works.
TEST_P(OperatorTest, testBatchMul_Float16) {
  CHECK_IF_ENABLED();
  testBatchOp<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, "mul");
}

/// Check that the sequence of extract-batchedmul-concat works.
TEST_P(OperatorTest, testBatchMul_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchOp<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                          "mul");
}

static void quantizedBatchOp(ExecutionEngine &EE, Function *F,
                             PlaceholderBindings &bindings, ElemKind Ty,
                             const std::string &opName) {
  CHECK(opName == "add" || opName == "mul") << "Invalid opName: " << opName;
  auto &mod = EE.getModule();
  constexpr unsigned numSlices = 10;
  constexpr unsigned batchSize = 3;

  auto *input = mod.createPlaceholder(
      ElemKind::FloatTy, {numSlices * batchSize, 10, 10}, "input", false);
  auto *slice =
      mod.createPlaceholder(ElemKind::FloatTy, {10, 10}, "slice", false);

  bindings.allocate(input)->getHandle().randomize(-5.0, 5.0, mod.getPRNG());
  bindings.allocate(slice)->getHandle().randomize(-5.0, 5.0, mod.getPRNG());

  // Scale the numbers in the range (-5. .. 5.) to (-50 .. 50).
  auto qInType =
      mod.uniqueType(ElemKind::Int8QTy, {numSlices * batchSize, 10, 10}, .1, 0);
  auto qSliceType2 = mod.uniqueType(Ty, {10, 10}, .1, 0);
  auto qSliceType3 =
      mod.uniqueType(ElemKind::Int8QTy, {batchSize, 10, 10}, .1, 0);

  auto *intInput = F->createQuantize("qinput", input, qInType);
  auto *intSlice = F->createQuantize("qslice", slice, qSliceType2);

  const Type *outTy;

  if (opName == "add") {
    outTy = qInType;
  } else {
    outTy = mod.uniqueType(ElemKind::Int8QTy, {batchSize, 10, 10}, 1.2, 0);
  }

  std::vector<NodeValue> ops;
  for (dim_t i = 0; i < numSlices; i++) {
    auto *ex =
        F->createSlice("slice", intInput, {i * batchSize, 0, 0}, qSliceType3);
    if (opName == "add") {
      ops.push_back(F->createBatchedAdd("add", ex, intSlice)->getResult());
    } else {
      ops.push_back(
          F->createBatchedMul("mul", outTy, ex, intSlice)->getResult());
    }
  }

  Node *cc = F->createConcat(
      "concat", ops, 0, mod.uniqueTypeWithNewShape(outTy, qInType->dims()));
  cc = F->createDequantize("dq", cc, ElemKind::FloatTy);
  auto *result = F->createSave("save", cc);
  bindings.allocate(result->getPlaceholder());

  // Remove the reference to the graph nodes to allow DCE to remove them.
  ops.clear();

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto RH = bindings.get(result->getPlaceholder())->getHandle();
  auto IH = bindings.get(input)->getHandle();
  auto SH = bindings.get(slice)->getHandle();

  // Check that batched add works as expected.
  for (dim_t i = 0; i < numSlices * batchSize; i++) {
    for (dim_t j = 0; j < 10; j++) {
      for (dim_t k = 0; k < 10; k++) {
        if (opName == "add") {
          EXPECT_NEAR(IH.at({i, j, k}) + SH.at({j, k}), RH.at({i, j, k}), 0.1);

        } else {
          EXPECT_NEAR(IH.at({i, j, k}) * SH.at({j, k}), RH.at({i, j, k}), 2.0);
        }
      }
    }
  }
}

/// Tests quantized batched-add arithmetic on Int8QTy.
TEST_P(OperatorTest, testQuantizedBatchAdd_Int8) {
  CHECK_IF_ENABLED();
  quantizedBatchOp(EE_, F_, bindings_, ElemKind::Int8QTy, "add");
}

/// Tests quantized batched-add arithmetic on Int32QTy.
TEST_P(OperatorTest, testQuantizedBatchAdd_Int32) {
  CHECK_IF_ENABLED();
  quantizedBatchOp(EE_, F_, bindings_, ElemKind::Int32QTy, "add");
}

/// Tests quantized batched-mul arithmetic on Int8QTy.
TEST_P(OperatorTest, testQuantizedBatchMul_Int8) {
  CHECK_IF_ENABLED();
  quantizedBatchOp(EE_, F_, bindings_, ElemKind::Int8QTy, "mul");
}

template <typename DataType>
static Tensor *testCumSum(glow::PlaceholderBindings &bindings,
                          glow::Module &mod, glow::Function *F,
                          glow::ExecutionEngine &EE, ElemKind DTy,
                          bool exclusive, bool reverse) {
  auto *data = mod.createPlaceholder(DTy, {4}, "data", false);
  bindings.allocate(data)->getHandle<DataType>() = {1, 2, 3, 4};

  auto *CS = F->createCumSum("CumSum", data, exclusive, reverse);
  auto *S = F->createSave("save", CS);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);
  return bindings.get(S->getPlaceholder());
}

TEST_P(OperatorTest, CumSum_Float) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [1, 3, 6, 10]
  */

  Tensor *result =
      testCumSum<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                        /*exclusive*/ false, /*reverse*/ false);
  Tensor expected(result->getType());
  expected.getHandle<float>() = {1, 3, 6, 10};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_Float16) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [1, 3, 6, 10]
  */

  Tensor *result =
      testCumSum<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                            /*exclusive*/ false, /*reverse*/ false);
  Tensor expected(result->getType());
  expected.getHandle<float16_t>() = {1, 3, 6, 10};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_BFloat16) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [1, 3, 6, 10]
  */

  Tensor *result =
      testCumSum<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                             /*exclusive*/ false, /*reverse*/ false);
  Tensor expected(result->getType());
  expected.getHandle<bfloat16_t>() = {1, 3, 6, 10};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_Int32) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [1, 3, 6, 10]
  */

  Tensor *result =
      testCumSum<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy,
                          /*exclusive*/ false, /*reverse*/ false);
  Tensor expected(result->getType());
  expected.getHandle<int32_t>() = {1, 3, 6, 10};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_Int64) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [1, 3, 6, 10]
  */

  Tensor *result =
      testCumSum<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                        /*exclusive*/ false, /*reverse*/ false);
  Tensor expected(result->getType());
  expected.getHandle<float>() = {1, 3, 6, 10};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_Exclusive) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [0, 1, 3, 6]
  */

  Tensor *result =
      testCumSum<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                        /*exclusive*/ true, /*reverse*/ false);
  Tensor expected(result->getType());
  expected.getHandle<float>() = {0, 1, 3, 6};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_Reverse) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [10, 9, 7, 4]
  */

  Tensor *result =
      testCumSum<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                            /*exclusive*/ false, /*reverse*/ true);
  Tensor expected(result->getType());
  expected.getHandle<float16_t>() = {10, 9, 7, 4};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_Reverse_BFloat16) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [10, 9, 7, 4]
  */

  Tensor *result =
      testCumSum<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                             /*exclusive*/ false, /*reverse*/ true);
  Tensor expected(result->getType());
  expected.getHandle<bfloat16_t>() = {10, 9, 7, 4};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_ExclusiveReverse) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [1, 2, 3, 4]
    OUTPUT = [9, 7, 4, 0]
  */

  Tensor *result =
      testCumSum<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy,
                          /*exclusive*/ true, /*reverse*/ true);
  Tensor expected(result->getType());
  expected.getHandle<int32_t>() = {9, 7, 4, 0};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, CumSum_WithZeroes) {
  CHECK_IF_ENABLED();
  /*
    DATA  = [0, 0, 1, 0, 0, 2, 0, 0, 3]
    OUTPUT = [0, 0, 1, 1, 1, 3, 3, 3, 6]
  */

  auto *data = mod_.createPlaceholder(ElemKind::Int64ITy, {9}, "data", false);
  bindings_.allocate(data)->getHandle<int64_t>() = {0, 0, 1, 0, 0, 2, 0, 0, 3};

  auto *CS = F_->createCumSum("CumSum", data);
  auto *S = F_->createSave("save", CS);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  Tensor *result = bindings_.get(S->getPlaceholder());
  Tensor expected(result->getType());
  expected.getHandle<int64_t>() = {0, 0, 1, 1, 1, 3, 3, 3, 6};

  EXPECT_TRUE(expected.isEqual(*result));
}

TEST_P(OperatorTest, LengthsSum) {
  CHECK_IF_ENABLED();

  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 3.7],
        [3.0, 2.9],
        [1.1, 1.4],
        [2.8, 8.4],
    ]
    LENGTHS = [2, 0, 3, 1]
    OUTPUT = [
        [3.3, 4.6],
        [0.0, 0.0],
        [8.6, 8.0],
        [2.8, 8.4],
    ]
  */
  auto *data = mod_.createPlaceholder(ElemKind::FloatTy, {6, 2}, "data", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings_.allocate(data)->getHandle() = {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 3.7f,
                                           3.0f, 2.9f, 1.1f, 1.4f, 2.8f, 8.4f};
  bindings_.allocate(lengths)->getHandle<int32_t>() = {2, 0, 3, 1};

  auto *R = F_->createLengthsSum("LS", data, lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4, 2});
  expected.getHandle() = {3.3f, 4.6f, 0.0f, 0.0f, 8.6f, 8.0f, 2.8f, 8.4f};

  EXPECT_TRUE(expected.isEqual(result));
}

/// Helper to test SLS using \p DTy.
template <typename DataType, typename IndexType>
static void testSLS(glow::PlaceholderBindings &bindings, glow::Module &mod,
                    glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
                    ElemKind ITy, float allowedError) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  auto *data = mod.createPlaceholder(DTy, {3, 2}, "data", false);
  auto *indices = mod.createPlaceholder(ITy, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {5}, "lengths", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };
  bindings.allocate(indices)->getHandle<IndexType>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F->createSparseLengthsSum("SLS", data, indices, lengths);

  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test that SLS is correctly supported in FloatTy with int64 indices.
TEST_P(OperatorTest, SparseLengthsSum_Float) {
  CHECK_IF_ENABLED();
  testSLS<float, int64_t>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                          ElemKind::Int64ITy, 0.0001);
}

/// Test that SLS is correctly supported in FloatTy with int32 indices.
TEST_P(OperatorTest, SparseLengthsSum_Float_Int32) {
  CHECK_IF_ENABLED();
  testSLS<float, int32_t>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                          ElemKind::Int32ITy, 0.0001);
}

/// Test that SLS is correctly supported in Float16Ty with int64 indices.
TEST_P(OperatorTest, SparseLengthsSum_Float16) {
  CHECK_IF_ENABLED();
  testSLS<float16_t, int64_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              ElemKind::Int64ITy, 0.002);
}

/// Test that SLS is correctly supported in BFloat16Ty with int64 indices.
TEST_P(OperatorTest, SparseLengthsSum_BFloat16) {
  CHECK_IF_ENABLED();
  testSLS<bfloat16_t, int64_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                               ElemKind::Int64ITy, 0.05);
}

/// Test that SLS is correctly supported in Float16Ty with int32 indices.
TEST_P(OperatorTest, SparseLengthsSum_Float16_Int32) {
  CHECK_IF_ENABLED();
  testSLS<float16_t, int32_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              ElemKind::Int32ITy, 0.05);
}

/// Test that SLS is correctly supported in BFloat16Ty with int32 indices.
TEST_P(OperatorTest, SparseLengthsSum_BFloat16_Int32) {
  CHECK_IF_ENABLED();
  testSLS<bfloat16_t, int32_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                               ElemKind::Int32ITy, 0.05);
}

TEST_P(OperatorTest, SparseLengthsSumI8) {
  CHECK_IF_ENABLED();

  /*
    DATA  = [
        [11, 13],
        [24, 35],
        [46, 58],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [56, 70],
        [ 1,  1],
        [69, 92],
        [11, 13],
        [31, 37],
    ]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3, 2}, 0.1f, 1, "data", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {5}, "lengths", false);

  bindings_.allocate(data)->getHandle<int8_t>() = {
      11, 13, 24, 35, 46, 58,
  };
  bindings_.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int8QTy, {5, 2}, 0.1f, 1);
  expected.getHandle<int8_t>() = {
      56, 70, 1, 1, 69, 92, 11, 13, 31, 37,
  };
  EXPECT_TRUE(expected.isEqual(result));
}

/// Test SparseLengthsWeightedSum with an N-dimension embedding table.
template <typename DataType>
static void testSLWS(glow::PlaceholderBindings &bindings, glow::Module &mod,
                     glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
                     float allowedError, size_t ndims) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  ShapeVector idims(ndims, 1);
  ShapeVector odims(ndims, 1);
  idims[0] = 3;
  odims[0] = 4;

  auto *data = mod.createPlaceholder(DTy, idims, "data", false);
  auto *weights = mod.createPlaceholder(DTy, {8}, "weights", false);
  auto *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      2.0,
      -0.5,
      13,
  };
  bindings.allocate(weights)->getHandle<DataType>() = {
      3, 1, 0, 0, 0, 0, 2, -0.5,
  };
  bindings.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                              lengths);
  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, odims);
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test that SLWS is correctly supported in FloatTy in 1D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_1D_Float) {
  CHECK_IF_ENABLED();
  testSLWS<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                  /* ndims */ 1);
}

/// Test that SLWS is correctly supported in FloatTy in 2D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_2D_Float) {
  CHECK_IF_ENABLED();
  testSLWS<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                  /* ndims */ 2);
}

/// Test that SLWS is correctly supported in Float16Ty in 1D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_1D_Float16) {
  CHECK_IF_ENABLED();
  testSLWS<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.0001,
                      /* ndims */ 1);
}

/// Test that SLWS is correctly supported in BFloat16Ty in 1D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_1D_BFloat16) {
  CHECK_IF_ENABLED();
  testSLWS<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, 0.0001,
                       /* ndims */ 1);
}

/// Test that SLWS is correctly supported in Float16Ty in 2D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_2D_Float16) {
  CHECK_IF_ENABLED();
  testSLWS<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.0001,
                      /* ndims */ 2);
}

/// Test that SLWS is correctly supported in BFloat16Ty in 2D.
TEST_P(OperatorTest, SparseLengthsWeightedSum_2D_BFloat16) {
  CHECK_IF_ENABLED();
  testSLWS<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty, 0.0001,
                       /* ndims */ 2);
}

TEST_P(OperatorTest, SparseLengthsWeightedSumI8) {
  CHECK_IF_ENABLED();

  /*
    DATA  =   [4, -1, 26]
    WEIGHTS = [6, 2, 0, 0, 0, 0, 4, -1]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [1, 0, 0, 50]
  */
  auto *data =
      mod_.createPlaceholder(ElemKind::Int8QTy, {3}, 0.5, 0, "data", false);
  auto *weights =
      mod_.createPlaceholder(ElemKind::Int8QTy, {8}, 0.5, 0, "weights", false);
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings_.allocate(data)->getHandle<int8_t>() = {
      4,
      -1,
      26,
  };
  bindings_.allocate(weights)->getHandle<int8_t>() = {
      6, 2, 0, 0, 0, 0, 4, -1,
  };
  bindings_.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F_->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                               lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int8QTy, {4}, 0.5, 0);
  expected.getHandle<int8_t>() = {
      1,
      0,
      0,
      50,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Helper function to construct indices/offsets pair for EmbeddingBag
/// and EmbeddingBagByteRowwiseOffsets
template <typename DataType>
static void addEmbeddingBagPartialInputs(
    glow::PlaceholderBindings &bindings, glow::Module &mod, ElemKind DTy,
    Placeholder *&weights, Placeholder *&indices, Placeholder *&offsets,
    bool hasEndOffset, bool partialInput = false) {

  if (hasEndOffset) {
    Tensor weightsTensorReal(DTy, {10});
    Tensor indicesTensorReal(ElemKind::Int64ITy, {10});
    Tensor offsetsTensorReal(ElemKind::Int64ITy, {5});

    weightsTensorReal.getHandle<DataType>() = {
        3, 1, 0, 0, 0, 0, 2, -0.5, 42.0, 42.0,
    };
    indicesTensorReal.getHandle<int64_t>() = {
        1, 0, 2, 0, 1, 2, 2, 0, 13, 10,
    };
    offsetsTensorReal.getHandle<int64_t>() = {
        0, 3, 3, 6,
        8, // extra end offset
    };

    if (partialInput) {
      weights = mod.createPlaceholder(DTy, {20}, "weights", false);
      indices =
          mod.createPlaceholder(ElemKind::Int64ITy, {20}, "indices", false);
      offsets =
          mod.createPlaceholder(ElemKind::Int64ITy, {6}, "offsets", false);

      // If we use partial weights, it will cause problems when it added as a
      // Constant. So here we pad it with zeros.
      Tensor weightsTensorPadded(weights->getType());
      memcpy(weightsTensorPadded.getUnsafePtr(),
             weightsTensorReal.getUnsafePtr(),
             weightsTensorReal.getSizeInBytes());
      memset(weightsTensorPadded.getUnsafePtr() +
                 weightsTensorReal.getSizeInBytes(),
             0,
             weightsTensorPadded.getSizeInBytes() -
                 weightsTensorReal.getSizeInBytes());

      Tensor indicesTensorPartial(indicesTensorReal.getUnsafePtr(),
                                  indices->getType(),
                                  indicesTensorReal.getSizeInBytes());
      Tensor offsetsTensorPartial(offsetsTensorReal.getUnsafePtr(),
                                  offsets->getType(),
                                  offsetsTensorReal.getSizeInBytes());
      bindings.insert(weights, std::move(weightsTensorPadded));
      bindings.insert(indices, indicesTensorPartial.clone());
      bindings.insert(offsets, offsetsTensorPartial.clone());
    } else {
      weights = mod.createPlaceholder(DTy, {10}, "weights", false);
      indices =
          mod.createPlaceholder(ElemKind::Int64ITy, {10}, "indices", false);
      offsets =
          mod.createPlaceholder(ElemKind::Int64ITy, {5}, "offsets", false);

      bindings.insert(weights, std::move(weightsTensorReal));
      bindings.insert(indices, std::move(indicesTensorReal));
      bindings.insert(offsets, std::move(offsetsTensorReal));
    }
  } else {
    // We assume no partial inputs will be used if hasEndOffset is false
    Tensor weightsTensorReal(DTy, {8});
    Tensor indicesTensorReal(ElemKind::Int64ITy, {8});
    Tensor offsetsTensorReal(ElemKind::Int64ITy, {4});

    weightsTensorReal.getHandle<DataType>() = {
        3, 1, 0, 0, 0, 0, 2, -0.5,
    };
    indicesTensorReal.getHandle<int64_t>() = {
        1, 0, 2, 0, 1, 2, 2, 0,
    };
    offsetsTensorReal.getHandle<int64_t>() = {0, 3, 3, 6};

    weights = mod.createPlaceholder(DTy, {8}, "weights", false);
    indices = mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
    offsets = mod.createPlaceholder(ElemKind::Int64ITy, {4}, "offsets", false);

    bindings.insert(weights, std::move(weightsTensorReal));
    bindings.insert(indices, std::move(indicesTensorReal));
    bindings.insert(offsets, std::move(offsetsTensorReal));
  }
}

/// Test EmbeddingBag with an N-dimension embedding table.
template <typename DataType>
static void testEmbeddingBag(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy,
                             float allowedError, dim_t ndims, bool hasEndOffset,
                             bool partialInput = false) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    OFFSETS = [0, 3, 3, 6]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  ShapeVector idims(ndims, 1);
  ShapeVector odims(ndims, 1);
  idims[0] = 3;
  odims[0] = partialInput ? 5 : 4;

  auto *data = mod.createPlaceholder(DTy, idims, "data", false);

  bindings.allocate(data)->getHandle<DataType>() = {
      2.0,
      -0.5,
      13,
  };

  // If hasEndOffset then add some additional junk to the end of indices and
  // weights and an extra offset to offsets.
  Placeholder *weights;
  Placeholder *indices;
  Placeholder *offsets;

  addEmbeddingBagPartialInputs<DataType>(bindings, mod, DTy, weights, indices,
                                         offsets, hasEndOffset, partialInput);

  auto *R = F->createEmbeddingBag("EB", data, weights, indices, offsets,
                                  hasEndOffset);
  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, odims);
  if (partialInput) {
    expected.getHandle<DataType>() = {
        0.5, 0, 0, 25, 0,
    };
  } else {
    expected.getHandle<DataType>() = {
        0.5,
        0,
        0,
        25,
    };
  }

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test that EB is correctly supported in FloatTy in 1D.
TEST_P(OperatorTest, EmbeddingBag_1D_Float) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 1, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in FloatTy in 1D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_1D_Float_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 1, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in FloatTy in 2D.
TEST_P(OperatorTest, EmbeddingBag_2D_Float) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 2, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in FloatTy in 2D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_2D_Float_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 2, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in Float16Ty in 1D.
TEST_P(OperatorTest, EmbeddingBag_1D_Float16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 1, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in BFloat16Ty in 1D.
TEST_P(OperatorTest, EmbeddingBag_1D_BFloat16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                               0.0001,
                               /* ndims */ 1, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in Float16Ty in 1D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_1D_Float16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 1, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in BFloat16Ty in 1D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_1D_BFloat16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                               0.0001,
                               /* ndims */ 1, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in Float16Ty in 2D.
TEST_P(OperatorTest, EmbeddingBag_2D_Float16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 2, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in BFloat16Ty in 2D.
TEST_P(OperatorTest, EmbeddingBag_2D_BFloat16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                               0.0001,
                               /* ndims */ 2, /* hasEndOffset */ false);
}

/// Test that EB is correctly supported in Float16Ty in 2D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_2D_Float16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                              0.0001,
                              /* ndims */ 2, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in BFloat16Ty in 2D with an end offset.
TEST_P(OperatorTest, EmbeddingBag_2D_BFloat16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                               0.0001,
                               /* ndims */ 2, /* hasEndOffset */ true);
}

/// Test that EB is correctly supported in FloatTy in 1D with an end offset and
/// partial inputs.
TEST_P(OperatorTest, EmbeddingBag_1D_Float_End_Offset_Partial) {
  CHECK_IF_ENABLED();
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 1, /* hasEndOffset */ true,
                          /* partialInput */ true);
}

/// Test that EB is correctly supported in Float16Ty in 1D with an end offset
/// and partial inputs.
TEST_P(OperatorTest, EmbeddingBag_2D_Float_End_Offset_Partial) {
  CHECK_IF_ENABLED();
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testEmbeddingBag<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001,
                          /* ndims */ 2, /* hasEndOffset */ true,
                          /* partialInput */ true);
}

/// Helper to test EmbeddingBagByteRowwiseOffsets using \p DTy.
template <typename DataType>
static void testEmbeddingBagByteRowwiseOffsets(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind fusedDTy, float allowedError,
    bool useFP16Accumulation, bool hasEndOffset, bool partialInput = false) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    OFFSETS = [0, 3, 3, 6]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  const bool fusedData = isFusedQuantizedElemKind(fusedDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(fusedDTy) : fusedDTy;
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  // If hasEndOffset then add some additional junk to the end of indices and
  // weights and an extra offset to offsets.
  // Note that weights here needs to be Constant instead of Placeholder for
  // EmbeddingBagByteRowwiseOffsets, so we need to convert it later on
  Placeholder *weights;
  Placeholder *indices;
  Placeholder *offsets;

  addEmbeddingBagPartialInputs<DataType>(bindings, mod, DTy, weights, indices,
                                         offsets, hasEndOffset, partialInput);

  auto *R = F->createEmbeddingBagByteRowwiseOffsets(
      "EBBRO", data, weights, indices, offsets, fusedDTy, useFP16Accumulation,
      hasEndOffset);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  ::glow::convertPlaceholdersToConstants(
      F, bindings, {indices, offsets, S->getPlaceholder()});

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  ShapeVector odims(2, 1);
  odims[0] = partialInput ? 5 : 4;
  Tensor expected(DTy, odims);
  if (partialInput) {
    expected.getHandle<DataType>() = {
        0.5, 0, 0, 25, 0,
    };
  } else {
    expected.getHandle<DataType>() = {
        0.5,
        0,
        0,
        25,
    };
  }

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test EmbeddingBagByteRowwiseOffsets in Float.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ false);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float with end offset.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float with end offset and partial
/// inputs.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float_End_Offset_Partial) {
  CHECK_IF_ENABLED();
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testEmbeddingBagByteRowwiseOffsets<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true,
      /* partialInputs */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float accumulation.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ false);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float accumulation.
/// Has end offset.
TEST_P(OperatorTest,
       EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float accumulation.
/// Has end offset and using partial inputs.
TEST_P(OperatorTest,
       EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset_Partial) {
  CHECK_IF_ENABLED();
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true,
      /* partialInputs */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest, EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ true, /* hasEndOffset */ false);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float16 accumulation.
/// Has end offset.
TEST_P(OperatorTest,
       EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset) {
  CHECK_IF_ENABLED();
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ true, /* hasEndOffset */ true);
}

/// Test EmbeddingBagByteRowwiseOffsets in Float16. Uses Float16 accumulation.
/// Has end offset and using partial inputs.
TEST_P(OperatorTest,
       EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset_Partial) {
  CHECK_IF_ENABLED();
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testEmbeddingBagByteRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.0001,
      /* useFP16Accumulation */ false, /* hasEndOffset */ true,
      /* partialInputs */ true);
}

/// Helper to test EmbeddingBag4BitRowwiseOffsets.
template <typename DataType>
static void testEmbeddingBag4BitRowwiseOffsets(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, bool useFP16Accumulation, bool hasEndOffset,
    float allowedError) {
  /*
    DATA  =   [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], // First Slice.
               [-3, -2, -1., 0], [0, -1, -2, -3],  // Second Slice.
               [2, 2, 2, 2,], [2, 2, 2, 2]  // Third Slice.
               ]
    WEIGHTS = [1, 2, 3, 2, 0.5, -0.5, 2]
    INDICES = [0, 1, 2, 4, 3, 5, 6]
    OFFSETS = [
        0, // This slice contains numbers >= 0.
        3, // This slice contains numbers <= 0.
        5, // This slice contains numbers which are all the same.
        7, // Empty slice.
    ]
    OUTPUT =  [[0, 6, 12, 18], // Output row per slice.
               [-1.5, -3, -4.5, -6],
               [3, 3, 3, 3]
               [0, 0, 0, 0]]
  */
  Tensor data(ElemKind::FloatTy, {7, 4});
  data.getHandle() = {
      0.,  1., 2., 3.,  0.,  1.,  2., 3., 0., 1., 2., 3., -3., -2.,
      -1., 0., 0., -1., -2., -3., 2., 2., 2., 2., 2., 2., 2.,  2.,
  };

  // If hasEndOffset then add some additional junk to the end of indices and
  // weights and an extra offset to offsets.
  Constant *weights;
  Placeholder *indices;
  Placeholder *offsets;
  if (hasEndOffset) {
    weights = mod.createConstant(ElemKind::Float16Ty, {9}, "weights");
    weights->getPayloadMutable().getHandle<DataType>() = {
        1.,
        2.,
        3.,
        2,
        0.5,
        -0.5,
        2,
        -42.0 /* A dummy weight for end offset. */,
        42.0 /* A dummy weight for end offset. */,
    };

    indices = mod.createPlaceholder(ElemKind::Int64ITy, {9}, "indices",
                                    /* isTrainable */ false);
    offsets = mod.createPlaceholder(ElemKind::Int64ITy, {5}, "offsets",
                                    /* isTrainable */ false);

    bindings.allocate(indices)->getHandle<int64_t>() = {
        0,
        1,
        2,
        4,
        3,
        5,
        6,
        100 /* A dummy indice for end offset. */,
        200 /* A dummy indice for end offset. */,
    };

    bindings.allocate(offsets)->getHandle<int64_t>() = {
        0, // This slice contains numbers >= 0.
        3, // This slice contains numbers <= 0.
        5, // This slice contains numbers which are all the same.
        7, // Empty slice.
        7, // Dummy end offset.
    };

  } else {
    weights = mod.createConstant(ElemKind::Float16Ty, {7}, "weights");
    weights->getPayloadMutable().getHandle<DataType>() = {
        1., 2., 3., 2, 0.5, -0.5, 2,
    };

    indices = mod.createPlaceholder(ElemKind::Int64ITy, {7}, "indices",
                                    /* isTrainable */ false);
    offsets = mod.createPlaceholder(ElemKind::Int64ITy, {4}, "offsets",
                                    /* isTrainable */ false);

    bindings.allocate(indices)->getHandle<int64_t>() = {
        0, 1, 2, 4, 3, 5, 6,
    };
    bindings.allocate(offsets)->getHandle<int64_t>() = {
        0, // This slice contains numbers >= 0.
        3, // This slice contains numbers <= 0.
        5, // This slice contains numbers which are all the same.
        7, // Empty slice.
    };
  }

  auto *R = F->createEmbeddingBagByteRowwiseOffsets(
      "EBBRO", data, weights, indices, offsets, ElemKind::UInt4FusedFP16QTy,
      useFP16Accumulation, hasEndOffset);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(ElemKind::Float16Ty, {4, 4});
  expected.getHandle<DataType>() = {0., 6., 12., 18., -1.5, -3., -4.5, -6,
                                    3., 3., 3.,  3.,  0.,   0.,  0.,   0.};

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

TEST_P(OperatorTest, EmbeddingBag4BitRowwiseOffsets_Float16) {
  CHECK_IF_ENABLED();
  testEmbeddingBag4BitRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_,
      /* useFP16Accumulation */ false, /* hasEndOffset */ false, 0.005);
}

TEST_P(OperatorTest, EmbeddingBag4BitRowwiseOffsets_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testEmbeddingBag4BitRowwiseOffsets<float16_t>(
      bindings_, mod_, F_, EE_,
      /* useFP16Accumulation */ true, /* hasEndOffset */ false, 0.005);
}

TEST_P(OperatorTest, EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset) {
  CHECK_IF_ENABLED();
  testEmbeddingBag4BitRowwiseOffsets<float16_t>(bindings_, mod_, F_, EE_,
                                                /* useFP16Accumulation */ false,
                                                /* hasEndOffset */ true, 0.005);
}

TEST_P(OperatorTest,
       EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset_AccumFloat) {
  CHECK_IF_ENABLED();
  testEmbeddingBag4BitRowwiseOffsets<float16_t>(bindings_, mod_, F_, EE_,
                                                /* useFP16Accumulation */ true,
                                                /* hasEndOffset */ true, 0.005);
}

/// Helper to test RowwiseQuantizedSparseLengthsWeightedSum using \p DTy.
template <typename DataType, typename IndexType>
static void testRowwiseQuantizedSparseLengthsWeightedSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind DTy, ElemKind ITy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  =   [2.0, -0.5, 13]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  Tensor data(ElemKind::FloatTy, {3});
  data.getHandle<float>() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(DTy, {8}, "weights");
  weights->getPayloadMutable().getHandle<DataType>() = {
      3., 1., 0., 0., 0., 0., 2., -0.5,
  };

  Placeholder *indices = mod.createPlaceholder(ITy, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<IndexType>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths,
      quantization::Schema::Asymmetric, DTy, useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {4});
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test RWQ-SLWS with Float Weights, Scales, Offsets, and Output.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsWeightedSum_Float) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::FloatTy, ElemKind::Int64ITy, 0.0001);
}

/// Test RWQ-SLWS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float accumulation.
TEST_P(OperatorTest,
       RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int64ITy, 0.0001,
      /* useFP16Accumulation */ false);
}

/// Test RWQ-SLWS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float16 accumulation.
TEST_P(OperatorTest,
       RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int64ITy, 0.0001,
      /* useFP16Accumulation */ true);
}

/// Test RWQ-SLWS with Float Weights, Scales, Offsets, and Output. Int32
/// indices.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsWeightedSum_Float_Int32) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::FloatTy, ElemKind::Int32ITy, 0.0001);
}

/// Test RWQ-SLWS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float accumulation. Int32 indices.
TEST_P(OperatorTest,
       RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat_Int32) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int32ITy, 0.0001,
      /* useFP16Accumulation */ false);
}

/// Test RWQ-SLWS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float16 accumulation. Int32 indices.
TEST_P(OperatorTest,
       RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16_Int32) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, ElemKind::Int32ITy, 0.0001,
      /* useFP16Accumulation */ true);
}

static FunctionTensorPair
createAndInitRWQSLWSAllSame(glow::PlaceholderBindings &bindings,
                            glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  Tensor data(ElemKind::FloatTy, {20, 2});
  data.getHandle<float>() = {
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {21}, "weights");
  weights->getPayloadMutable().getHandle<float>() = {
      0.44419134, 0.3419154,  0.28775468, 0.47224975, 0.05422213, 0.14346851,
      0.05846643, 0.3750175,  0.09190885, 0.3335992,  0.09665264, 0.4560224,
      0.2244578,  0.44881952, 0.42696562, 0.33007848, 0.4511249,  0.11568925,
      0.02629679, 0.33864713, 0.42614424};

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {21}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {2}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      11, 8, 19, 8, 4, 11, 4, 19, 6, 18, 2, 6, 15, 5, 14, 14, 15, 13, 4, 6, 5,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {15, 6};

  auto *R = F->createRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths,
      quantization::Schema::Asymmetric, ElemKind::FloatTy,
      /* useFP16Accumulation */ false);
  SaveNode *S = F->createSave("save", R);
  Tensor *resultT = bindings.allocate(S->getPlaceholder());

  return std::make_pair(F, resultT);
}

TEST_P(OperatorStatelessTest, RWQSLWSAllSame_Float16_AccumFP16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitRWQSLWSAllSame, ElemKind::Float16Ty,
      ElemKind::Float16Ty, 0.0005, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      /*schema */ quantization::Schema::Asymmetric,
      /* biasElemKind */ ElemKind::Int32QTy, /* forceFP16AccumSLS */ true);
}

TEST_P(OperatorStatelessTest, RWQSLWSAllSame_Float16_AccumFP32) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitRWQSLWSAllSame, ElemKind::Float16Ty,
      ElemKind::Float16Ty, 1e-6, parCloneCountOpt,
      /* convertToRowwiseQuantization */ false,
      /*schema */ quantization::Schema::Asymmetric,
      /* biasElemKind */ ElemKind::Int32QTy, /* forceFP16AccumSLS */ false);
}

/// Helper to test RowwiseQuantizedSparseLengthsWeightedSum using \p DTy.
template <typename DataType>
static void testRowwiseQuantizedSparseLengthsSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind DTy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  Tensor data(ElemKind::FloatTy, {3, 2});
  data.getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F->createRowwiseQuantizedSparseLengthsSum(
      "RQSLWS", data, indices, lengths, quantization::Schema::Asymmetric, DTy,
      useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test RWQ-SLS with Float Weights, Scales, Offsets, and Output.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsSum_Float) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsSum<float>(bindings_, mod_, F_, EE_,
                                              ElemKind::FloatTy, 0.015);
}

/// Test RWQ-SLS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float accumulation.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.02,
      /* useFP16Accumulation */ false);
}

/// Test RWQ-SLS with Float16 Weights, Scales, Offsets, and Output. Uses
/// Float16 accumulation.
TEST_P(OperatorTest, RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.02,
      /* useFP16Accumulation */ true);
}

TEST_P(OperatorTest, RepeatedSLSWithPartialTensors) {
  CHECK_IF_ENABLED();

  // This test is only meaningful if the backend supports partial tensors.
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());

  constexpr dim_t embeddingRows = 1275;
  constexpr dim_t numLengths = 20;
  constexpr dim_t maxIndices = 20000;
  constexpr dim_t numIndices = 20; // Must be less than sum(lengths).
  constexpr dim_t iterations = 33;

  auto *data =
      mod_.createConstant(ElemKind::FloatTy, {embeddingRows, 1}, "data");
  data->getPayloadMutable().getHandle<float>().randomize(-1.0, 1.0,
                                                         mod_.getPRNG());
  auto *indices = mod_.createPlaceholder(ElemKind::Int64ITy, {maxIndices},
                                         "indices", false);
  auto *lengths = mod_.createPlaceholder(ElemKind::Int32ITy, {numLengths},
                                         "lengths", false);
  auto *SLS = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *save = F_->createSave("save", SLS);
  auto *outPH = save->getPlaceholder();
  EE_.compile(CompilationMode::Infer);

  Tensor indicesReal(ElemKind::Int64ITy, {numIndices});
  indicesReal.getHandle<int64_t>().randomize(0, embeddingRows - 1,
                                             mod_.getPRNG());
  Tensor indicesPartial(indicesReal.getUnsafePtr(), indices->getType(),
                        indicesReal.getSizeInBytes());
  Tensor indicesPadded(indices->getType());
  indicesPadded.zero();
  memcpy(indicesPadded.getUnsafePtr(), indicesReal.getUnsafePtr(),
         numIndices * sizeof(int64_t));

  Tensor lengthsReal(ElemKind::Int32ITy, {numLengths});
  lengthsReal.getHandle<int32_t>().clear(1);
  Tensor lengthsPartial(lengthsReal.getUnsafePtr(), lengths->getType(),
                        lengthsReal.getSizeInBytes());
  Tensor lengthsPadded(ElemKind::Int32ITy, {numLengths});
  lengthsPadded.assign(&lengthsReal);

  bindings_.insert(indices, std::move(indicesPartial));
  bindings_.insert(lengths, std::move(lengthsPartial));
  bindings_.allocate(outPH);

  PlaceholderBindings paddedBindings;
  paddedBindings.insert(indices, std::move(indicesPadded));
  paddedBindings.insert(lengths, std::move(lengthsPadded));
  paddedBindings.allocate(outPH);

  for (dim_t i = 0; i < iterations; i++) {
    EE_.run(bindings_);
    EE_.run(paddedBindings);
    ASSERT_TRUE(bindings_.get(outPH)->isEqual(*paddedBindings.get(outPH)));
  }

  // Keep these around so their memory is not freed at the end of the
  // test/scope. This is so that inside TearDown during import/export testing
  // the data is still around.
  unownedTensors_.push_back(std::move(indicesReal));
  unownedTensors_.push_back(std::move(lengthsReal));
}

/// Helper to test gathers using partial inputs using \p ITy.
template <typename IndicesType>
static void
testPartialGather(glow::PlaceholderBindings &bindings, glow::Module &mod,
                  glow::Function *F, glow::ExecutionEngine &EE,
                  std::vector<Tensor> &unownedTensors, ElemKind ITy) {
  /*
    The acutal input we care about has the following shape/result:

    DATA  = [1.0, 2.3, 4.5]
    INDICES = [0, 1, 0, 1, 2, 0]
    OUTPUT = [1.0, 2.3, 1.0, 2.3, 4.5, 1.0]

    However, we are going to create a larger INDICES input that is only
    partially filled, and expect a larger OUTPUT that we expect will have data
    we do not care about.
  */

  Placeholder *data = mod.createPlaceholder(ElemKind::FloatTy, {3}, "data",
                                            /* isTrainable */ false);
  Placeholder *indices =
      mod.createPlaceholder(ITy, {10000}, "indices", /* isTrainable */ false);

  bindings.allocate(data)->getHandle<float>() = {1.0f, 2.3f, 4.5f};

  Tensor indicesReal(ITy, {6});
  indicesReal.getHandle<IndicesType>() = {0, 1, 0, 1, 2, 0};
  Tensor indicesPartial(indicesReal.getUnsafePtr(), indices->getType(),
                        indicesReal.getSizeInBytes());
  bindings.insert(indices, std::move(indicesPartial));

  auto *R = F->createGather("gather", data, indices);

  auto *result = F->createSave("save", R);
  Tensor *resultT = bindings.allocate(result->getPlaceholder());

  // Result should be 10000, even though we only care about the first 6
  // results.
  EXPECT_EQ(resultT->getType().dims().size(), 1);
  EXPECT_EQ(resultT->getType().dims()[0], 10000);

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor expectedT(ElemKind::FloatTy, {6});
  auto expectedH = expectedT.getHandle<float>();
  expectedH = {1.0, 2.3, 1.0, 2.3, 4.5, 1.0};
  auto resultH = resultT->getHandle<float>();

  for (dim_t i = 0; i < 6; ++i) {
    EXPECT_EQ(expectedH.at({i}), resultH.at({i}));
  }

  // Keep this around so their memory is not freed at the end of the
  // test/scope. This is so that inside TearDown during import/export testing
  // the data is still around.
  unownedTensors.push_back(std::move(indicesReal));
}

TEST_P(OperatorTest, GatherWithInt64PartialTensors) {
  CHECK_IF_ENABLED();
  // This test is only meaningful if the backend supports partial tensors.
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testPartialGather<int64_t>(bindings_, mod_, F_, EE_, unownedTensors_,
                             ElemKind::Int64ITy);
}

TEST_P(OperatorTest, GatherWithInt32PartialTensors) {
  CHECK_IF_ENABLED();
  // This test is only meaningful if the backend supports partial tensors.
  ASSERT_TRUE(EE_.getBackend(getBackendName()).supportsPartialTensors());
  testPartialGather<int32_t>(bindings_, mod_, F_, EE_, unownedTensors_,
                             ElemKind::Int32ITy);
}

/// Helper to test FusedRowwiseQuantizedSparseLengthsWeightedSum using \p DTy.
template <typename DataType, typename IndexType>
static void testFusedRowwiseQuantizedSparseLengthsWeightedSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind fusedDTy, ElemKind ITy,
    float allowedError, bool useFP16Accumulation = false) {
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [[0.5, 0, 0, 25]]
  */
  const bool fusedData = isFusedQuantizedElemKind(fusedDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(fusedDTy) : fusedDTy;
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(DTy, {8}, "weights");
  weights->getPayloadMutable().getHandle<DataType>() = {
      3., 1., 0., 0., 0., 0., 2., -0.5,
  };

  Placeholder *indices = mod.createPlaceholder(ITy, {8}, "indices",
                                               /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<IndexType>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths, fusedDTy, useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {4, 1});
  expected.getHandle<DataType>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test Fused-RWQ-SLWS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSparseLengthsWeightedSum_Float) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, ElemKind::Int64ITy,
      0.0001);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, ElemKind::Int64ITy,
      0.0001,
      /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int64_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, ElemKind::Int64ITy,
      0.0001,
      /* useFP16Accumulation */ true);
}

/// Test Fused-RWQ-SLWS in Float. Int32 indices.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_Float_Int32) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, ElemKind::Int32ITy,
      0.0001);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float accumulation. Int32 indices.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat_Int32) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, ElemKind::Int32ITy,
      0.0001,
      /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float16 accumulation. Int32 indices.
TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16_Int32) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsWeightedSum<float16_t, int32_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, ElemKind::Int32ITy,
      0.0001,
      /* useFP16Accumulation */ true);
}

static void testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, float allowedError, bool convertFusedToFP16,
    bool useFP16AccumSLS) {
  CHECK_IF_ENABLED();
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [[0.5, 0, 0, 25]]
  */
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod.createConstant(ElemKind::FloatTy, {8}, "weights");
  weights->getPayloadMutable().getHandle<float>() = {
      3., 1., 0., 0., 0., 0., 2., -0.5,
  };

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                            /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = convertFusedToFP16;
  cctx.precisionConfig.forceFP16AccumSLS = useFP16AccumSLS;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::FP16;

  EE.compile(cctx);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4, 1});
  expected.getHandle<float>() = {
      0.5,
      0,
      0,
      25,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test Fused-RWQ-SLWS in where the weights are in Fp16, data
/// inputs are UInt8FusedQTy.
TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_NoFusedConvert) {
  CHECK_IF_ENABLED();
  return testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
      bindings_, mod_, F_, EE_, 0.02,
      /* convertFusedToFP16*/ false, /* useFP16AccumSLS */ true);
}

TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_NoFusedConvert_FP32Accum) {
  CHECK_IF_ENABLED();
  return testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
      bindings_, mod_, F_, EE_, 0.02,
      /* convertFusedToFP16*/ false, /* useFP16AccumSLS */ false);
}

TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16) {
  CHECK_IF_ENABLED();
  return testRowwiseQuantizedSparseLengthsSum_ConvertedFloat16(
      bindings_, mod_, F_, EE_, 0.02,
      /* convertFusedToFP16*/ true, /* useFP16AccumSLS */ true);
}

TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_back) {
  CHECK_IF_ENABLED();
  /*
    DATA  =   [[2.0, -0.5, 13]]
    WEIGHTS = [1]
    INDICES = [0]
    LENGTHS = [0, 0, 0, 1] and then [1, 0, 0, 0]
    OUTPUT =  [[0, 0, 0, 0.2]] and then [[2.0, 0, 0, 0]]
  */
  Tensor data(ElemKind::FloatTy, {3, 1});
  data.getHandle() = {
      2.0,
      -0.5,
      13,
  };

  Constant *weights = mod_.createConstant(ElemKind::FloatTy, {1}, "weights");
  weights->getPayloadMutable().getHandle<float>() = {1.};

  Placeholder *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1}, "indices",
                             /* isTrainable */ false);
  Placeholder *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths",
                             /* isTrainable */ false);

  bindings_.allocate(indices)->getHandle<int64_t>() = {
      0,
  };
  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      0,
      0,
      0,
      1,
  };

  auto *R = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths);
  SaveNode *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::FP16;

  EE_.compile(cctx);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {4, 1});
  expected.getHandle<float>() = {
      0,
      0,
      0,
      2.0,
  };

  EXPECT_TRUE(expected.isEqual(result, 0.02));

  // Send another inference
  bindings_.get(lengths)->getHandle<int32_t>() = {
      1,
      0,
      0,
      0,
  };
  EE_.run(bindings_);

  Tensor &result1 = *bindings_.get(S->getPlaceholder());
  Tensor expected1(ElemKind::FloatTy, {4, 1});
  expected1.getHandle<float>() = {
      2.0,
      0,
      0,
      0,
  };
  EXPECT_TRUE(expected1.isEqual(result1, 0.02));
}

TEST_P(
    OperatorTest,
    FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_back2) {
  CHECK_IF_ENABLED();

  Tensor data(ElemKind::FloatTy, {10000, 64});
  data.getHandle().randomize(-1, 1, mod_.getPRNG());

  Placeholder *weights =
      mod_.createPlaceholder(ElemKind::FloatTy, {10000}, "weights",
                             /* isTrainable */ false);

  Placeholder *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {10000}, "indices",
                             /* isTrainable */ false);
  Placeholder *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {32}, "lengths",
                             /* isTrainable */ false);

  Tensor *wT = bindings_.allocate(weights);
  wT->zero();
  wT->getHandle<float>().at({0}) = 4.18067;

  Tensor *iT = bindings_.allocate(indices);
  iT->zero();
  iT->getHandle<int64_t>().at({0}) = 4124;

  bindings_.allocate(lengths)->getHandle<int32_t>() = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0};

  auto *R = F_->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
      "RQSLWS", data, weights, indices, lengths);
  SaveNode *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  CompilationContext cctx;
  cctx.precisionConfig.convertToFP16 = true;
  cctx.precisionConfig.convertFusedToFP16 = true;
  cctx.precisionConfig.float16Format =
      PrecisionConfiguration::Float16Format::FP16;

  EE_.compile(cctx);
  EE_.run(bindings_);

  // This is the result for the first inference. We expect the result in the
  // second last row or raw location 30 * 64 to 31 * 64 -1. The rest of the
  // rows should be all 0.
  Tensor &result = *bindings_.get(S->getPlaceholder());

  // Send another inference
  result.zero();
  // set new indices.
  iT = bindings_.get(indices);
  iT->zero();
  iT->getHandle<int64_t>().at({0}) = 1256;
  // set new lengths.
  bindings_.get(lengths)->getHandle<int32_t>() = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

  };
  EE_.run(bindings_);

  // We now expect the second to last row to be all 0.
  Tensor &result1 = *bindings_.get(S->getPlaceholder());
  float *d = reinterpret_cast<float *>(result1.getUnsafePtr());
  for (size_t i = 30 * 64; i < 31 * 64; ++i) {
    EXPECT_EQ(0, d[i]);
  }
}

/// Helper to test FusedRowwiseQuantizedSparseLengthsSum using \p fusedDTy.
template <typename DataType>
static void testFusedRowwiseQuantizedSparseLengthsSum(
    glow::PlaceholderBindings &bindings, glow::Module &mod, glow::Function *F,
    glow::ExecutionEngine &EE, ElemKind fusedDTy, float allowedError,
    bool useFP16Accumulation = false) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    OUTPUT = [
        [5.5, 6.9],
        [0.0, 0.0],
        [6.8, 9.1],
        [1.0, 1.2],
        [3.0, 3.6],
    ]
  */
  const bool fusedData = isFusedQuantizedElemKind(fusedDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(fusedDTy) : fusedDTy;

  Tensor data(ElemKind::FloatTy, {3, 2});
  data.getHandle() = {
      1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f,
  };

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };

  auto *R = F->createFusedRowwiseQuantizedSparseLengthsSum(
      "RQSLWS", data, indices, lengths, fusedDTy, useFP16Accumulation);
  SaveNode *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      5.5f, 6.9f, 0.0f, 0.0f, 6.8f, 9.1f, 1.0f, 1.2f, 3.0f, 3.6f,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test Fused-RWQ-SLS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSparseLengthsSum_Float) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.015);
}

/// Test Fused-RWQ-SLS in Float16. Uses Float accumulation.
TEST_P(OperatorTest, FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.02,
      /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.02,
      /* useFP16Accumulation */ true);
}

/// Test Fused-RWQ-SLS in Float16 wth 4-bit quantization for the embedding.
/// Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSparseLengthsSum_Fused4Bit_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testFusedRowwiseQuantizedSparseLengthsSum<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt4FusedFP16QTy, 0.15,
      /* useFP16Accumulation */ true);
}

/// Helper to test all variants of SLWS wiith all lengths as one, with
/// precision \p DTy, and precision for data \p dataDTy.
template <typename DataType>
static void testSLWSTwoColumn(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind dataDTy,
                              float allowedError,
                              bool useFP16Accumulation = false) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0, 0, 0, 0]
    LENGTHS = [2, 0, 2, 1, 3]
    WEIGHTS = [1, -1, 1.5, 0.5, -1.5, 2, -2, -0.5]
    OUTPUT = [
        [3.5, 4.5],
        [0.0, 0.0],
        [5.7, 7.95],
        [-1.5, -1.8],
        [-0.5, -0.6],
    ]
  */
  const bool fusedData = isFusedQuantizedElemKind(dataDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(dataDTy) : dataDTy;

  Tensor data(fusedData ? ElemKind::FloatTy : DTy, {3, 2});
#define floatData                                                              \
  { 1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f, }
  if (fusedData) {
    data.getHandle<float>() = floatData;
  } else {
    data.getHandle<DataType>() = floatData;
  }

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);
  Placeholder *weights =
      mod.createPlaceholder(DTy, {8}, "weights", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0, 0, 0, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      2, 0, 2, 1, 3,
  };
  bindings.allocate(weights)->getHandle<DataType>() = {
      1, -1, 1.5, 0.5, -1.5, 2, -2, -0.5,
  };

  Node *SLWS = nullptr;
  if (fusedData) {
    SLWS = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "RQSLWS", data, weights, indices, lengths, dataDTy,
        useFP16Accumulation);
  } else {
    Placeholder *dataP = mod.createPlaceholder(&data.getType(), "data",
                                               /* isTrainable */ false);
    bindings.insert(dataP, std::move(data));
    SLWS = F->createSparseLengthsWeightedSum("SLWS", dataP, weights, indices,
                                             lengths);
  }
  SaveNode *S = F->createSave("save", SLWS);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  Tensor expected(DTy, {5, 2});
  expected.getHandle<DataType>() = {
      3.5, 4.5, 0.0, 0.0, 5.7, 7.95, -1.5, -1.8, -0.5, -0.6,
  };

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test SLWS in Float.
TEST_P(OperatorTest, SLWSTwoColumn_Float) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.0001);
}

/// Test SLWS in Float16.
TEST_P(OperatorTest, SLWSTwoColumn_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                               0.005,
                               /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSTwoColumn_Float) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float>(bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy,
                           0.015);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float accumulation.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_,
                               ElemKind::UInt8FusedFP16QTy, 0.015,
                               /* useFP16Accumulation */ false);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_,
                               ElemKind::UInt8FusedFP16QTy, 0.015,
                               /* useFP16Accumulation */ true);
}

/// Test Fused-RWQ-SLWS in Float16 wth 4-bit quantization for the embedding.
/// Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSLWSTwoColumn_Fused4Bit_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testSLWSTwoColumn<float16_t>(bindings_, mod_, F_, EE_,
                               ElemKind::UInt4FusedFP16QTy, 0.1,
                               /* useFP16Accumulation */ true);
}

/// Helper to test SLWS with different lengths modes, with precision \p DTy,
/// and precision for data \p dataDTy.
template <typename DataType>
static void testSLWSLengthsMode(glow::PlaceholderBindings &bindings,
                                glow::Module &mod, glow::Function *F,
                                glow::ExecutionEngine &EE, ElemKind dataDTy,
                                float allowedError, bool useFP16Accumulation,
                                LengthsMode lengthsMode) {
  /*
    DATA  = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ]
    INDICES = [2, 0, 1, 2, 0]
    LENGTHS = [1, 1, 1, 1, 1]
    WEIGHTS = [1, -1, 1.5, 0.5, -1.5]
    OUTPUT = [
        [4.5, 5.7],
        [-1.0, -1.2],
        [3.45, 5.1],
        [2.25, 2.85],
        [-1.5, -1.8],
    ]
  */
  const bool fusedData = isFusedQuantizedElemKind(dataDTy);
  const ElemKind DTy =
      fusedData ? getScaleOffsetElemKindFromFused(dataDTy) : dataDTy;

  Tensor data(fusedData ? ElemKind::FloatTy : DTy, {3, 2});
#define floatData                                                              \
  { 1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f, }
  if (fusedData) {
    data.getHandle<float>() = floatData;
  } else {
    data.getHandle<DataType>() = floatData;
  }

  Placeholder *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {5}, "indices",
                            /* isTrainable */ false);
  Placeholder *lengths = mod.createPlaceholder(
      ElemKind::Int32ITy, {5}, "lengths", /* isTrainable */ false);
  Placeholder *weights =
      mod.createPlaceholder(DTy, {5}, "weights", /* isTrainable */ false);

  bindings.allocate(indices)->getHandle<int64_t>() = {
      2, 0, 1, 2, 0,
  };
  auto LH = bindings.allocate(lengths)->getHandle<int32_t>();
  Tensor expected(DTy, {5, 2});
  LH = {1, 1, 1, 1, 1};
  expected.getHandle<DataType>() = {
      4.5, 5.7, -1.0, -1.2, 3.45, 5.1, 2.25, 2.85, -1.5, -1.8,
  };
  bindings.allocate(weights)->getHandle<DataType>() = {
      1, -1, 1.5, 0.5, -1.5,
  };

  Node *SLWS = nullptr;
  if (fusedData) {
    SLWS = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "RQSLWS", data, weights, indices, lengths, dataDTy, useFP16Accumulation,
        lengthsMode);
  } else {
    Placeholder *dataP = mod.createPlaceholder(&data.getType(), "data",
                                               /* isTrainable */ false);
    bindings.insert(dataP, std::move(data));
    SLWS = F->createSparseLengthsWeightedSum("SLWS", dataP, weights, indices,
                                             lengths, lengthsMode);
  }
  SaveNode *S = F->createSave("save", SLWS);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());

  EXPECT_TRUE(expected.isEqual(result, allowedError));
}

/// Test SLWS in Float.
TEST_P(OperatorTest, SLWSAllLengthsOne_Float) {
  CHECK_IF_ENABLED();
  testSLWSLengthsMode<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy,
                             0.0001, /* useFP16Accumulation */ false,
                             LengthsMode::AllOne);
}

/// Test SLWS in Float16.
TEST_P(OperatorTest, SLWSAllLengthsOne_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testSLWSLengthsMode<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::Float16Ty, 0.005,
      /* useFP16Accumulation */ false, LengthsMode::AllOne);
}

/// Test Fused-RWQ-SLWS in Float.
TEST_P(OperatorTest, FusedRowwiseQuantizedSLWSAllLengthsOne_Float) {
  CHECK_IF_ENABLED();
  testSLWSLengthsMode<float>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedQTy, 0.015,
      /* useFP16Accumulation */ false, LengthsMode::AllOne);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat) {
  CHECK_IF_ENABLED();
  testSLWSLengthsMode<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.015,
      /* useFP16Accumulation */ false, LengthsMode::AllOne);
}

/// Test Fused-RWQ-SLWS in Float16. Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testSLWSLengthsMode<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt8FusedFP16QTy, 0.015,
      /* useFP16Accumulation */ true, LengthsMode::AllOne);
}

/// Test Fused-RWQ-SLWS in Float16 wth 4-bit quantization for the embedding.
/// Uses Float16 accumulation.
TEST_P(OperatorTest,
       FusedRowwiseQuantizedSLWSAllLengthsOne_Fused4Bit_Float16_AccumFloat16) {
  CHECK_IF_ENABLED();
  testSLWSLengthsMode<float16_t>(
      bindings_, mod_, F_, EE_, ElemKind::UInt4FusedFP16QTy, 0.1,
      /* useFP16Accumulation */ true, LengthsMode::AllOne);
}

/// Test SLS when some input tensors are constants.
TEST_P(OperatorTest, ConstantSLS) {
  CHECK_IF_ENABLED();

  auto *data = mod_.createConstant(ElemKind::FloatTy, {1024, 32}, "data");
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {314}, "indices", false);
  auto *lengths = mod_.createConstant(ElemKind::Int32ITy, {20}, "lengths");

  // data
  auto DH = data->getPayload().getHandle();
  for (dim_t i = 0; i < 1024; i++) {
    for (dim_t j = 0; j < 32; j++) {
      DH.at({i, j}) = (float)i;
    }
  }

  // indices
  auto IH = bindings_.allocate(indices)->getHandle<int64_t>();
  std::iota(IH.begin(), IH.end(), 0);

  // lengths
  auto LH = lengths->getHandle<int32_t>();
  LH.clear(16);
  for (dim_t ldx : {1, 2, 6, 13, 14, 19}) {
    LH.at({ldx}) = 15;
  }

  auto *R = F_->createSparseLengthsSum("SLS", data, indices, lengths);
  auto *S = F_->createSave("save", R);
  auto *out = bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  std::vector<float> expected = {120,  345,  570,  856,  1112, 1368, 1515,
                                 1864, 2120, 2376, 2632, 2888, 3144, 3180,
                                 3405, 3880, 4136, 4392, 4648, 4590};
  auto OH = out->getHandle();
  for (dim_t i = 0; i < 20; i++) {
    for (dim_t j = 0; j < 32; j++) {
      EXPECT_EQ(OH.at({i, j}), expected[i]);
    }
  }
}

/// Test SLS when some "lengths" inputs are zero.
TEST_P(OperatorStatelessTest, SLSWithZeroLengths) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(
      getBackendName(),
      [](PlaceholderBindings &bindings, ExecutionEngine &EE) {
        auto &mod = EE.getModule();
        auto *F = mod.createFunction("main");
        constexpr dim_t embedWidth = 1000;
        Tensor data(ElemKind::FloatTy, {embedWidth, 8});
        data.getHandle().randomize(-1, 1, mod.getPRNG());
        Constant *weights =
            mod.createConstant(ElemKind::FloatTy, {3000}, "weights");
        weights->getPayloadMutable().getHandle().clear(1.0f);
        auto *indices =
            mod.createPlaceholder(ElemKind::Int64ITy, {3000}, "indices", false);
        auto *lengths =
            mod.createPlaceholder(ElemKind::Int32ITy, {1000}, "lengths", false);
        bindings.allocate(indices)->getHandle<int64_t>().randomize(
            0, embedWidth - 1, mod.getPRNG());
        auto LH = bindings.allocate(lengths)->getHandle<int32_t>();
        LH.clear(0);
        auto it = LH.begin();
        for (int i = 0; i < 13; ++i, ++it) {
          *it = 20;
        }

        auto *R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
            "RQSLWS", data, weights, indices, lengths);
        auto *S = F->createSave("save", R);
        auto *res = bindings.allocate(S->getPlaceholder());
        return std::make_pair(F, res);
      },
      ElemKind::FloatTy, ElemKind::FloatTy);
}

/// Helper to create an SLS test with all zero lengths, with and without fused
/// rowwise quantization based on \p convertToRowwiseQuantization.
static FunctionTensorPair
createAndInitZeroLengthsSLSTest(glow::PlaceholderBindings &bindings,
                                glow::ExecutionEngine &EE,
                                bool convertToRowwiseQuantization) {
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("main");
  constexpr dim_t embedWidth = 1000;
  auto dataTy = mod.uniqueType(ElemKind::FloatTy, {embedWidth, 8});
  Tensor data(dataTy);
  data.getHandle().randomize(-1, 1, mod.getPRNG());
  Constant *weights = mod.createConstant(ElemKind::FloatTy, {3000}, "weights");
  weights->getPayloadMutable().getHandle().clear(1.0f);
  auto *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {3000}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {1000}, "lengths", false);
  bindings.allocate(indices)->getHandle<int64_t>().randomize(0, embedWidth - 1,
                                                             mod.getPRNG());
  auto LH = bindings.allocate(lengths)->getHandle<int32_t>();
  LH.clear(0);

  Node *R = nullptr;
  if (convertToRowwiseQuantization) {
    R = F->createFusedRowwiseQuantizedSparseLengthsWeightedSum(
        "RQSLWS", data, weights, indices, lengths);
  } else {
    Placeholder *dataP =
        mod.createPlaceholder(dataTy, "data", /* isTrainable */ false);
    bindings.insert(dataP, std::move(data));
    R = F->createSparseLengthsWeightedSum("SLWS", dataP, weights, indices,
                                          lengths);
  }
  auto *S = F->createSave("save", R);
  auto *res = bindings.allocate(S->getPlaceholder());
  return std::make_pair(F, res);
}

/// Test Fused RWQ-SLS when all "lengths" inputs are zero in FloatTy.
TEST_P(OperatorStatelessTest, FusedRWQSLSAllZeroLengths_Float) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(getBackendName(),
                            std::bind(createAndInitZeroLengthsSLSTest,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      /* convertToRowwiseQuantization */ true),
                            ElemKind::FloatTy, ElemKind::FloatTy);
}

/// Test Fused RWQ-SLS when all "lengths" inputs are zero in Float16Ty.
TEST_P(OperatorStatelessTest, FusedRWQSLSAllZeroLengths_Float16) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(getBackendName(),
                            std::bind(createAndInitZeroLengthsSLSTest,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      /* convertToRowwiseQuantization */ true),

                            ElemKind::Float16Ty, ElemKind::Float16Ty);
}

/// Test SLS when all "lengths" inputs are zero in FloatTy.
TEST_P(OperatorStatelessTest, SLSAllZeroLengths_Float) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(getBackendName(),
                            std::bind(createAndInitZeroLengthsSLSTest,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      /* convertToRowwiseQuantization */ false),
                            ElemKind::FloatTy, ElemKind::FloatTy);
}

/// Test SLS when all "lengths" inputs are zero in Float16Ty.
TEST_P(OperatorStatelessTest, SLSAllZeroLengths_Float16) {
  CHECK_IF_ENABLED();

  compareAgainstInterpreter(getBackendName(),
                            std::bind(createAndInitZeroLengthsSLSTest,
                                      std::placeholders::_1,
                                      std::placeholders::_2,
                                      /* convertToRowwiseQuantization */ false),

                            ElemKind::Float16Ty, ElemKind::Float16Ty);
}

template <typename DataType>
static void testSparseToDense(glow::PlaceholderBindings &bindings,
                              glow::Module &mod, glow::Function *F,
                              glow::ExecutionEngine &EE, ElemKind DTy) {

  // Create and initialize inputs. Make input 3D to make sure
  // multidimensional values are handled properly.
  constexpr dim_t kNumIndices = 4;
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 5;
  constexpr dim_t kMaxIndex = 10;

  auto *indices = mod.createPlaceholder(ElemKind::Int64ITy, {kNumIndices},
                                        "indices", false);
  auto *values =
      mod.createPlaceholder(DTy, {kNumIndices, kRows, kCols}, "data", false);
  auto *dataToInferDim = mod.createPlaceholder(ElemKind::FloatTy, {kMaxIndex},
                                               "dataToInferDim", false);

  auto IH = bindings.allocate(indices)->getHandle<int64_t>();
  auto VH = bindings.allocate(values)->getHandle<DataType>();

  // Duplicate one index to test that the corresponding values are added.
  IH = {1, 3, 1, 9};
  VH.randomize(-3.0, 3.0, mod.getPRNG());

  auto *STDN = F->createSparseToDense("STDN", indices, values, dataToInferDim);
  auto *S = F->createSave("save", STDN);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());

  // Compute expected output.
  Tensor expected(DTy, {kMaxIndex, kRows, kCols});
  auto EH = expected.getHandle<DataType>();

  expected.zero();
  for (dim_t i = 0; i < kNumIndices; ++i) {
    dim_t idx = IH.at({i});
    for (dim_t j = 0; j < kRows; ++j) {
      for (dim_t k = 0; k < kCols; ++k) {
        EH.at({idx, j, k}) += VH.at({i, j, k});
      }
    }
  }

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(OperatorTest, SparseToDense_Float) {
  CHECK_IF_ENABLED();
  testSparseToDense<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, SparseToDense_Int64) {
  CHECK_IF_ENABLED();
  testSparseToDense<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

TEST_P(OperatorTest, SparseToDenseMask1) {
  CHECK_IF_ENABLED();

  /*
    INDICES = [4, 42, 13, 0, 100, 13]
    VALUES = [-5.5, 0.7, 11, 1e6, 2, 3.5]
    DEFAULTVALUE = 1.1
    LENGTHS = [4, 2]
    MASK = [2, 1, 0, 13, 42, 43]
    OUTPUT =  [[1.1, 1.1, 1e6, 11, 0.7, 1.1], [1.1, 1.1, 1.1, 3.5, 1.1, 1.1]]
  */
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {6}, "indices", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {6}, "values", false);
  auto *defaultValue =
      mod_.createPlaceholder(ElemKind::FloatTy, {}, "default_value", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {2}, "lengths", false);
  std::vector<dim_t> mask{2, 1, 0, 13, 42, 43};

  bindings_.allocate(indices)->getHandle<int64_t>() = {4, 42, 13, 0, 100, 13};
  bindings_.allocate(values)->getHandle<float>() = {-5.5, 0.7, 11, 1e6, 2, 3.5};
  bindings_.allocate(defaultValue)->getHandle<float>().raw(0) = 1.1;
  bindings_.allocate(lengths)->getHandle<int32_t>() = {4, 2};

  auto *R = F_->createSparseToDenseMask("STDM", indices, values, defaultValue,
                                        lengths, mask);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {2, 6});
  expected.getHandle<float>() = {
      1.1, 1.1, 1e6, 11, 0.7, 1.1, 1.1, 1.1, 1.1, 3.5, 1.1, 1.1,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(OperatorTest, SparseToDenseMask2) {
  CHECK_IF_ENABLED();

  /*
    INDICES = [300, 100, 101, 299]
    VALUES = [[[-0.1, -0.2], [-0.3, -0.4]], [[2, -2], [2, 9]],
              [[15, 4.2], [10.3, 30.4]], [[0, 2], [3, 4.4]]]
    DEFAULTVALUE = [[0.1, 0.2], [0.3, 0.4]]
    LENGTHS = []
    MASK = [100, 300, 1]
    OUTPUT =  [[[2, -2], [2, 9]], [[-0.1, -0.2], [-0.3, -0.4]],
               [[0.1, 0.2], [0.3, 0.4]]]
  */
  auto *indices =
      mod_.createPlaceholder(ElemKind::Int64ITy, {4}, "indices", false);
  auto *values =
      mod_.createPlaceholder(ElemKind::FloatTy, {4, 2, 2}, "values", false);
  auto *defaultValue =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "default_value", false);
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {}, "lengths", false);
  std::vector<dim_t> mask{100, 300, 1};

  bindings_.allocate(indices)->getHandle<int64_t>() = {300, 100, 101, 299};
  bindings_.allocate(values)->getHandle<float>() = {
      -0.1, -0.2, -0.3, -0.4, 2, -2, 2, 9, 15, 4.2, 10.3, 30.4, 0, 2, 3, 4.4};
  bindings_.allocate(defaultValue)->getHandle<float>() = {0.1, 0.2, 0.3, 0.4};
  bindings_.allocate(lengths)->getHandle<int32_t>() = {4};

  auto *R = F_->createSparseToDenseMask("STDM", indices, values, defaultValue,
                                        lengths, mask);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::FloatTy, {3, 2, 2});
  expected.getHandle<float>() = {
      2, -2, 2, 9, -0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

TEST_P(OperatorTest, FP16Reshape) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::Float16Ty, {20, 13}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle<float16_t>();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createReshape("tr", A, {13, 20, 1});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle =
      bindings_.get(result->getPlaceholder())->getHandle<float16_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

TEST_P(OperatorTest, BoolReshape) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BoolTy, {4, 3}, "A", false);
  bindings_.allocate(A)->getHandle<bool>() = {false, true,  false, true,
                                              true,  false, false, false,
                                              true,  true,  true,  true};
  auto *tr = F_->createReshape("tr", A, {3, 4, 1});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle =
      bindings_.get(result->getPlaceholder())->getHandle<bool>();
  auto inputBoolHandle = bindings_.get(A)->getHandle<bool>();
  ASSERT_EQ(outputHandle.size(), inputBoolHandle.size());
  for (size_t idx = 0, end = inputBoolHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputBoolHandle.raw(idx), outputHandle.raw(idx));
  }
}

TEST_P(OperatorTest, BFloat16Reshape) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BFloat16Ty, {20, 13}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle<bfloat16_t>();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *tr = F_->createReshape("tr", A, {13, 20, 1});
  auto *result = F_->createSave("saveTranspose", tr);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle =
      bindings_.get(result->getPlaceholder())->getHandle<bfloat16_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Reshape operator works correctly.
TEST_P(OperatorTest, Reshape) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {5, 7}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle();
  inputHandle.randomize(-3.0, 3.0, mod_.getPRNG());

  auto *RN = F_->createReshape("reshape", A, {7, 5, 1});
  auto *result = F_->createSave("saveReshape", RN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle = bindings_.get(result->getPlaceholder())->getHandle();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  ASSERT_EQ(outputHandle.dims().size(), 3);
  EXPECT_EQ(outputHandle.dims()[0], 7);
  EXPECT_EQ(outputHandle.dims()[1], 5);
  EXPECT_EQ(outputHandle.dims()[2], 1);

  // Check values are still in the same order.
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Reshape operator works correctly with Int64ITy.
TEST_P(OperatorTest, ReshapeInt) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::Int64ITy, {5, 7}, "A", false);
  auto inputHandle = bindings_.allocate(A)->getHandle<int64_t>();
  inputHandle.randomize<int64_t>(0, 100, mod_.getPRNG());

  auto *RN = F_->createReshape("reshape", A, {7, 5, 1});
  auto *result = F_->createSave("saveReshape", RN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto outputHandle =
      bindings_.get(result->getPlaceholder())->getHandle<int64_t>();
  ASSERT_EQ(outputHandle.size(), inputHandle.size());
  ASSERT_EQ(outputHandle.dims().size(), 3);
  EXPECT_EQ(outputHandle.dims()[0], 7);
  EXPECT_EQ(outputHandle.dims()[1], 5);
  EXPECT_EQ(outputHandle.dims()[2], 1);

  // Check values are still in the same order.
  for (size_t idx = 0, end = inputHandle.size(); idx != end; ++idx) {
    EXPECT_EQ(inputHandle.raw(idx), outputHandle.raw(idx));
  }
}

/// Verify that the Select operator works correctly.
TEST_P(OperatorTest, Select) {
  CHECK_IF_ENABLED();

  auto *A = mod_.createPlaceholder(ElemKind::BoolTy, {5}, "A", false);
  bindings_.allocate(A)->getHandle<bool>() = {false, true, true, false, false};

  auto SNTy = mod_.uniqueType(ElemKind::FloatTy, {5});
  SplatNode *SN10 = F_->createSplat("zero", SNTy, 10.0);
  SplatNode *SN20 = F_->createSplat("zero", SNTy, 20.0);

  auto *SN = F_->createSelect("select", A, SN10, SN20);
  auto *result = F_->createSave("saveSelect", SN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resH = bindings_.get(result->getPlaceholder())->getHandle();
  EXPECT_EQ(resH.at({0}), 20.0);
  EXPECT_EQ(resH.at({1}), 10.0);
  EXPECT_EQ(resH.at({2}), 10.0);
  EXPECT_EQ(resH.at({3}), 20.0);
  EXPECT_EQ(resH.at({4}), 20.0);
}

/// Verify that the CmpLTE operator works correctly.
TEST_P(OperatorTest, CmpLTE) {
  CHECK_IF_ENABLED();

  Placeholder *A = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "A", false);
  Placeholder *B = mod_.createPlaceholder(ElemKind::FloatTy, {5}, "B", false);
  bindings_.allocate(A)->getHandle<float>() = {0.0, 1.0, 2.0, 3.0, 4.0};
  bindings_.allocate(B)->getHandle<float>() = {0.0, 1.1, 1.5, 10.1, -1.0};

  auto *CMPLTE = F_->createCmpLTE("select", A, B);
  auto *result = F_->createSave("saveCMPLTE", CMPLTE);
  Tensor *resultT = bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resH = resultT->getHandle<bool>();
  EXPECT_TRUE(resH.at({0}));
  EXPECT_TRUE(resH.at({1}));
  EXPECT_FALSE(resH.at({2}));
  EXPECT_TRUE(resH.at({3}));
  EXPECT_FALSE(resH.at({4}));
}

/// Helper to test SliceReshape using \p DTy.
template <typename DataType>
static void testSliceReshape(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  auto *X =
      createPlaceholderConditionallyQuantized(mod, DTy, {3, 3}, "X", false);

  auto XH = bindings.allocate(X)->getHandle<DataType>();
  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 3; j++) {
      XH.at({i, j}) = i * 3 + j;
    }
  }

  // Do an assortment of slices/reshapes stacked on top of each other.
  auto *SX = F->createSlice("sliceX", X, {2, 0}, {3, 3});
  auto *RSX = F->createReshape("reshapeSX", SX, {3});
  auto *SSX = F->createSlice("sliceSliceX", SX, {0, 2}, {1, 3});
  auto *RSSX = F->createReshape("reshapeSliceSliceX", SSX, {1});

  auto *resultSX = F->createSave("saveSX", SX);
  auto *resultRSX = F->createSave("saveRSX", RSX);
  auto *resultSSX = F->createSave("saveSSX", SSX);
  auto *resultRSSX = F->createSave("saveRSSX", RSSX);

  bindings.allocate(resultSX->getPlaceholder());
  bindings.allocate(resultRSX->getPlaceholder());
  bindings.allocate(resultSSX->getPlaceholder());
  bindings.allocate(resultRSSX->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  // Verify the slice has the same data as the original X.
  auto SXH = bindings.get(resultSX->getPlaceholder())->getHandle<DataType>();
  for (dim_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), XH.at({2, i}), 1E-5);
  }

  // Verify the reshaped slice has the same data as the slice.
  auto RSXH = bindings.get(resultRSX->getPlaceholder())->getHandle<DataType>();
  for (dim_t i = 0; i < 3; i++) {
    EXPECT_NEAR(SXH.at({0, i}), RSXH.at({i}), 1E-5);
  }

  // Verify the slice of the slice has the same data as the slice.
  auto SSXH = bindings.get(resultSSX->getPlaceholder())->getHandle<DataType>();
  EXPECT_NEAR(SXH.at({0, 2}), SSXH.at({0, 0}), 1E-5);

  // Verify the reshape of the slice of the slice has the same data as the
  // slice of the slice.
  auto RSSXH =
      bindings.get(resultRSSX->getPlaceholder())->getHandle<DataType>();
  EXPECT_NEAR(RSSXH.at({0}), SSXH.at({0, 0}), 1E-5);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in FloatTy.
TEST_P(OperatorTest, sliceReshape_Float) {
  CHECK_IF_ENABLED();

  testSliceReshape<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in Float16Ty.
TEST_P(OperatorTest, sliceReshape_Float16) {
  CHECK_IF_ENABLED();
  testSliceReshape<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in BFloat16Ty.
TEST_P(OperatorTest, sliceReshape_BFloat16) {
  CHECK_IF_ENABLED();
  testSliceReshape<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in Int8QTy.
TEST_P(OperatorTest, sliceReshape_Int8) {
  CHECK_IF_ENABLED();
  testSliceReshape<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Stack many slices/reshapes together. Some of these may be turned into
/// tensor views stacked onto each other. Test in Int32QTy.
TEST_P(OperatorTest, sliceReshape_Int32) {
  CHECK_IF_ENABLED();
  testSliceReshape<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

/// Helper to test Flatten using \p DTy.
template <typename DataType>
static void testFlatten(glow::PlaceholderBindings &bindings, glow::Module &mod,
                        glow::Function *F, glow::ExecutionEngine &EE,
                        ElemKind DTy) {
  auto *tensor4D = createPlaceholderConditionallyQuantized(
      mod, DTy, {3, 2, 4, 3}, "4D", false, "NHWC");
  bindings.allocate(tensor4D)->getHandle<DataType>().randomize(0, 100,
                                                               mod.getPRNG());

  NodeValue reshape4Dto2DAxis1 = F->createFlatten("flat4Dto2Da1", tensor4D, 1);
  EXPECT_EQ(reshape4Dto2DAxis1.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis1.dims()[0], 3);
  EXPECT_EQ(reshape4Dto2DAxis1.dims()[1], 24);

  NodeValue reshape4Dto2DAxis2 = F->createFlatten("flat4Dto2Da2", tensor4D, 2);
  EXPECT_EQ(reshape4Dto2DAxis2.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis2.dims()[0], 6);
  EXPECT_EQ(reshape4Dto2DAxis2.dims()[1], 12);

  NodeValue reshape4Dto2DAxis3 = F->createFlatten("flat4Dto2Da3", tensor4D, 3);
  EXPECT_EQ(reshape4Dto2DAxis3.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis3.dims()[0], 24);
  EXPECT_EQ(reshape4Dto2DAxis3.dims()[1], 3);

  // Now, let us do the fifth (4) axis.
  // This comes straight from caffe2 because flattening is
  // supported for every axis up and including the rank of a tensor.
  // The rank of this tensor is 4, so axis 4 is fine.
  NodeValue reshape4Dto2DAxis4 = F->createFlatten("flat4Dto2Da4", tensor4D, 4);
  EXPECT_EQ(reshape4Dto2DAxis4.dims().size(), 2);
  EXPECT_EQ(reshape4Dto2DAxis4.dims()[0], 72);
  EXPECT_EQ(reshape4Dto2DAxis4.dims()[1], 1);

  // This one is weird because we flatten something that is already flat, but
  // again because flattening is supported for every axis up and including the
  // rank of a tensor, 1D vector means we can flatten it on axis 1.
  auto *tensor1D =
      createPlaceholderConditionallyQuantized(mod, DTy, {15}, "1D", false, "N");
  bindings.allocate(tensor1D)->getHandle<DataType>().randomize(0, 100,
                                                               mod.getPRNG());

  NodeValue reshape1Dto2DAxis1 = F->createFlatten("flat1Dto2D", tensor1D, 1);
  EXPECT_EQ(reshape1Dto2DAxis1.dims().size(), 2);
  EXPECT_EQ(reshape1Dto2DAxis1.dims()[0], 15);
  EXPECT_EQ(reshape1Dto2DAxis1.dims()[1], 1);

  // Save all the reshapes so that the optimizations won't kill the network.
  auto *save1Dto2D = F->createSave("save1Dto2D", reshape1Dto2DAxis1);
  auto *save4Dto2Da1 = F->createSave("save4Dto2Da1", reshape4Dto2DAxis1);
  auto *save4Dto2Da2 = F->createSave("save4Dto2Da2", reshape4Dto2DAxis2);
  auto *save4Dto2Da3 = F->createSave("save4Dto2Da3", reshape4Dto2DAxis3);
  auto *save4Dto2Da4 = F->createSave("save4Dto2Da4", reshape4Dto2DAxis4);

  bindings.allocate(save1Dto2D->getPlaceholder());
  bindings.allocate(save4Dto2Da1->getPlaceholder());
  bindings.allocate(save4Dto2Da2->getPlaceholder());
  bindings.allocate(save4Dto2Da3->getPlaceholder());
  bindings.allocate(save4Dto2Da4->getPlaceholder());

  EE.compile(CompilationMode::Infer);

  EE.run(bindings);

  // Verify the reshapes have the same data as the original value.
  auto tensor4DH = bindings.get(tensor4D)->getHandle<DataType>();
  auto save4Dto2Da1H =
      bindings.get(save4Dto2Da1->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da1H.raw(i), 1E-5);
  }

  auto save4Dto2Da2H =
      bindings.get(save4Dto2Da2->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da2H.raw(i), 1E-5);
  }

  auto save4Dto2Da3H =
      bindings.get(save4Dto2Da3->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da3H.raw(i), 1E-5);
  }

  auto save4Dto2Da4H =
      bindings.get(save4Dto2Da4->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 72; i++) {
    EXPECT_NEAR(tensor4DH.raw(i), save4Dto2Da4H.raw(i), 1E-5);
  }

  auto tensor1DH = bindings.get(tensor1D)->getHandle<DataType>();
  auto save1Dto2DH =
      bindings.get(save1Dto2D->getPlaceholder())->getHandle<DataType>();
  for (size_t i = 0; i < 15; i++) {
    EXPECT_NEAR(tensor1DH.raw(i), save1Dto2DH.raw(i), 1E-5);
  }
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using FloatTy.
TEST_P(OperatorTest, Flatten_FloatTy) {
  CHECK_IF_ENABLED();
  testFlatten<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using Float16Ty.
TEST_P(OperatorTest, Flatten_Float16Ty) {
  CHECK_IF_ENABLED();
  testFlatten<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using BFloat16Ty.
TEST_P(OperatorTest, Flatten_BFloat16Ty) {
  CHECK_IF_ENABLED();
  testFlatten<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Check that the flatten operator produces 2D tensors of the right
/// dimensions, using Int8QTy.
TEST_P(OperatorTest, Flatten_Int8) {
  CHECK_IF_ENABLED();
  testFlatten<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Check that div on Int64ITy/size_t works.
TEST_P(OperatorTest, DivSizeT) {
  CHECK_IF_ENABLED();

  auto *LHS = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "LHS", false);
  auto *RHS = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 2}, "RHS", false);
  auto LHSH = bindings_.allocate(LHS)->getHandle<int64_t>();
  auto RHSH = bindings_.allocate(RHS)->getHandle<int64_t>();

  LHSH = {10, 20, 30, 40, 50, 60};
  RHSH = {2, 20, 100, 41, 3, 59};

  auto *R = F_->createDiv("div", LHS, RHS);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  // Disabling this so that  division of Int64ITy/size_t can be tested.
  cctx.optimizationOpts.enableTypeDemotion = false;
  EE_.compile(cctx);
  EE_.run(bindings_);

  auto H = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 2; j++) {
      EXPECT_EQ(LHSH.at({i, j}) / RHSH.at({i, j}), H.at({i, j}));
    }
  }
}

TEST_P(OperatorTest, SigmoidCrossEntropyWithLogits) {
  CHECK_IF_ENABLED();

  /*
    LOGITS  = [
      [
        [1.0, 1.2, -0.5],
        [0.1, 0.6, 0.5],
      ],
      [
        [-0.1, -2., 0.3],
        [1, 2, 3],
      ],
    ]
    TARGETS = [
      [
        [0.7, 0.7, 0.7],
        [-0.7, -0.99, 1.0],
      ],
      [
        [0, 0, 0],
        [1, 2, 3],
      ],
    ]
    OUTPUT = [
      [ 0.68687367,  0.97332054],
      [ 0.5418933,  -2.50374103],
    ]
  */
  auto *logits =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3}, "logits", false);
  auto *targets =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 3}, "targets", false);

  bindings_.allocate(logits)->getHandle() = {
      1.0f, 1.2f, -0.5f, 0.1f, 0.6f, 0.5f, -0.1f, -2.f, 0.3f, 1.f, 2.f, 3.f};
  bindings_.allocate(targets)->getHandle() = {
      0.7f, 0.7f, 0.7f, -0.7f, -0.99f, 1.0f, 0.f, 0.f, 0.f, 1.f, 2.f, 3.f};

  auto *R = F_->createSigmoidCrossEntropyWithLogits("SCEL", logits, targets);

  auto *result = F_->createSave("save", R);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor expected(ElemKind::FloatTy, {2, 2});
  expected.getHandle() = {
      0.68687367f,
      0.97332054f,
      0.5418933f,
      -2.50374103f,
  };

  EXPECT_TRUE(expected.isEqual(*bindings_.get(result->getPlaceholder())));
}

/// Test the InsertTensor node works correctly.
TEST_P(OperatorTest, insertTensorTest) {
  CHECK_IF_ENABLED();

  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  // 0 0 0 0 0 0
  auto *SN0 = mod_.createPlaceholder(ElemKind::FloatTy, {4, 6}, "SN0", false);
  bindings_.allocate(SN0)->init(Tensor::InitKind::Broadcast, 0, mod_.getPRNG());

  // 1 1
  // 1 1
  auto *SN1 = mod_.createPlaceholder(ElemKind::FloatTy, {2, 2}, "SN1", false);
  bindings_.allocate(SN1)->init(Tensor::InitKind::Broadcast, 1, mod_.getPRNG());

  // 0 0 0 0 0 0
  // 0 1 1 1 1 0
  // 0 1 1 1 1 0
  // 0 0 0 0 0 0
  Node *IN = F_->createInsertTensor("insert", SN0, SN1, /* start */ {1, 1},
                                    /* count */ 2, /* axis */ 1);
  SaveNode *result = F_->createSave("result", IN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  // Verify the output looks as expected (pictured above).
  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < 4; i++) {
    for (dim_t j = 0; j < 6; j++) {
      int64_t expected = 1;
      if (i == 0 || i == 3 || j == 0 || j == 5)
        expected = 0;
      EXPECT_EQ(resultH.at({i, j}), expected);
    }
  }
}

/// Test the InsertTensor node works correctly for 3 dimensions.
TEST_P(OperatorTest, insertTensorTest3D) {
  CHECK_IF_ENABLED();

  // 0 0 0 0 0 0 | 0 0 0 0 0 0
  // 0 0 0 0 0 0 | 0 0 0 0 0 0
  // 0 0 0 0 0 0 | 0 0 0 0 0 0
  // 0 0 0 0 0 0 | 0 0 0 0 0 0
  auto *SN0 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 4, 6}, "SN0", false);
  bindings_.allocate(SN0)->init(Tensor::InitKind::Broadcast, 0, mod_.getPRNG());

  // 1 1 | 1 1
  // 1 1 | 1 1
  auto *SN1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {2, 2, 2}, "SN1", false);
  bindings_.allocate(SN1)->init(Tensor::InitKind::Broadcast, 1, mod_.getPRNG());

  // 0 0 0 0 0 0 | 0 0 0 0 0 0
  // 0 1 1 1 1 0 | 0 1 1 1 1 0
  // 0 1 1 1 1 0 | 0 1 1 1 1 0
  // 0 0 0 0 0 0 | 0 0 0 0 0 0
  Node *IN = F_->createInsertTensor("insert", SN0, SN1, /* start */ {0, 1, 1},
                                    /* count */ 2, /* axis */ 2);
  SaveNode *result = F_->createSave("result", IN);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  // Verify the output looks as expected (pictured above).
  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < 2; i++) {
    for (dim_t j = 0; j < 4; j++) {
      for (dim_t k = 0; k < 6; k++) {
        int64_t expected = 1;
        if (j == 0 || j == 3 || k == 0 || k == 5)
          expected = 0;
        EXPECT_EQ(resultH.at({i, j, k}), expected);
      }
    }
  }
}

/// Test that the InsertTensor operator works correctly when crossing outer
/// dimensions.
TEST_P(OperatorTest, insertTensorCrossDimensions) {
  CHECK_IF_ENABLED();

  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  auto *SN0 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2, 5}, "SN0", false);
  bindings_.allocate(SN0)->init(Tensor::InitKind::Broadcast, 0, mod_.getPRNG());

  // 1 1 1 1 1 1 (T)
  auto *SN1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2, 1}, "SN1", false);
  bindings_.allocate(SN1)->init(Tensor::InitKind::Broadcast, 1, mod_.getPRNG());

  // 2 2 | 2 2
  // 2 2 | 2 2
  // 2 2 | 2 2
  auto *SN2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2, 2}, "SN2", false);
  bindings_.allocate(SN2)->init(Tensor::InitKind::Broadcast, 2, mod_.getPRNG());

  // 1 0 2 2 0
  // 1 0 2 2 0
  // 1 0 2 2 0
  // 1 0 2 2 0
  // 1 0 2 2 0
  // 1 0 2 2 0
  Node *IN = F_->createInsertTensor("insert", SN0, SN1, /* start */ {0, 0, 0},
                                    /* count */ 1, /* axis */ 2);
  Node *IN2 = F_->createInsertTensor("insert", IN, SN2, /* start */ {0, 0, 2},
                                     /* count */ 1, /* axis */ 2);
  SaveNode *result = F_->createSave("result", IN2);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);

  // Verify the output looks as expected (pictured above).
  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 2; j++) {
      for (dim_t k = 0; k < 5; k++) {
        int64_t expected = 0;
        if (k == 0)
          expected = 1;
        if (k == 2 || k == 3)
          expected = 2;
        EXPECT_EQ(resultH.at({i, j, k}), expected);
      }
    }
  }
}

/// Test the InsertTensor operator works correctly when inserting across an
/// outer dimension where the inner dimensions have different sizes.
TEST_P(OperatorTest, insertTensorPartialSliceInnerDim) {
  CHECK_IF_ENABLED();

  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  // 0 0 0 0 0
  auto *SN0 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 3, 5}, "SN0", false);
  bindings_.allocate(SN0)->init(Tensor::InitKind::Broadcast, 0, mod_.getPRNG());

  // 1 1
  // 1 1
  // 1 1
  auto *SN1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 2}, "SN1", false);
  bindings_.allocate(SN1)->init(Tensor::InitKind::Broadcast, 1, mod_.getPRNG());

  // 2 2 2
  // 2 2 2
  // 2 2 2
  auto *SN2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 1, 3}, "SN2", false);
  bindings_.allocate(SN2)->init(Tensor::InitKind::Broadcast, 2, mod_.getPRNG());

  // 1 1 0 0 0
  // 0 2 2 2 0
  // 0 0 0 0 0
  // 1 1 0 0 0
  // 0 2 2 2 0
  // 0 0 0 0 0
  // 1 1 0 0 0
  // 0 2 2 2 0
  // 0 0 0 0 0
  Node *IN = F_->createInsertTensor("insert", SN0, SN1, /* start */ {0, 0, 0},
                                    /* count */ 1, /* axis */ 2);
  Node *IN2 = F_->createInsertTensor("insert", IN, SN2, /* start */ {0, 1, 1},
                                     /* count */ 1, /* axis */ 2);
  SaveNode *result = F_->createSave("result", IN2);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);
  // Verify the output looks as expected (pictured above).
  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<float>();
  for (dim_t i = 0; i < 3; i++) {
    for (dim_t j = 0; j < 3; j++) {
      for (dim_t k = 0; k < 5; k++) {
        int64_t expected = 0;
        if (j == 0 && k <= 1)
          expected = 1;
        if (j == 1 && k >= 1 && k <= 3)
          expected = 2;
        EXPECT_EQ(resultH.at({i, j, k}), expected);
      }
    }
  }
}

static FunctionTensorPair
createAndInitBasicRowwiseFCTest(glow::PlaceholderBindings &bindings,
                                glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  // In this test we subtract the outputs of a row-wise quantized FC and a
  // floating-point FC and ensure that the error is below some low value.
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {2, 100}, "in", false);
  auto *fc = F->createFullyConnected(bindings, "FC", input, 5);

  auto *weights = llvm::cast<Placeholder>(fc->getWeights());
  auto *bias = llvm::cast<Placeholder>(fc->getBias());

  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.get(bias)->getHandle().randomize(0, 0.1, mod.getPRNG());
  bindings.get(weights)->getHandle().randomize(-1.1, 1.1, mod.getPRNG());

  auto *res = F->createSave("save", fc);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Test Int8 RowwiseQuantizedFullyConnected Node with Int8 bias.
TEST_P(OperatorStatelessTest, rowwiseQuantizedFCTest_Int8_BiasInt8) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.06f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Asymmetric,
      ElemKind::Int8QTy);
}

/// Test Int8 RowwiseQuantizedFullyConnected Node with Int32 bias.
TEST_P(OperatorStatelessTest, rowwiseQuantizedFCTest_Int8_BiasInt32) {
  ENABLED_BACKENDS("Interpreter", "CPU");
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.06f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Asymmetric,
      ElemKind::Int32QTy);
}

/// Test RowwiseQuantizedFullyConnected Node with Symmetric quantization.
TEST_P(OperatorStatelessTest, rowwiseQuantizedFCTestSymmetric) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.07f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Symmetric);
}

TEST_P(OperatorStatelessTest,
       rowwiseQuantizedFCTestSymmetric_Int8_BiasFloat32) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.07f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Symmetric,
      /*biasElemKind*/ ElemKind::Int32QTy,
      /*forceFP16AccumSLS*/ false, PrecisionConfiguration::Float16Format::None,
      /*convertToChannelwiseQuantization*/ false,
      /*skipQuantizeFCBias*/ true);
}

TEST_P(OperatorStatelessTest,
       rowwiseQuantizedFCTestAsymmetric_Int8_BiasFloat32) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(
      getBackendName(), createAndInitBasicRowwiseFCTest, ElemKind::FloatTy,
      ElemKind::Int8QTy, 0.06f, parCloneCountOpt,
      /* convertToRowwiseQuantization */ true, quantization::Schema::Asymmetric,
      /*biasElemKind*/ ElemKind::Int32QTy,
      /*forceFP16AccumSLS*/ false, PrecisionConfiguration::Float16Format::None,
      /*convertToChannelwiseQuantization*/ false,
      /*skipQuantizeFCBias*/ true);
}

static FunctionTensorPair
createAndInitBasicSLWSTest(glow::PlaceholderBindings &bindings,
                           glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  /*
    DATA  =   [2.0, -0.5, 13]
    WEIGHTS = [3, 1, 0, 0, 0, 0, 2, -0.5]
    INDICES = [1, 0, 2, 0, 1, 2, 2, 0]
    LENGTHS = [3, 0, 3, 2]
    OUTPUT =  [0.5, 0, 0, 25]
  */
  auto *data = mod.createPlaceholder(ElemKind::FloatTy, {3}, "data", false);
  auto *weights =
      mod.createPlaceholder(ElemKind::FloatTy, {8}, "weights", false);
  auto *indices =
      mod.createPlaceholder(ElemKind::Int64ITy, {8}, "indices", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings.allocate(data)->getHandle() = {
      2.0,
      -0.5,
      13,
  };
  bindings.allocate(weights)->getHandle() = {
      3, 1, 0, 0, 0, 0, 2, -0.5,
  };
  bindings.allocate(indices)->getHandle<int64_t>() = {
      1, 0, 2, 0, 1, 2, 2, 0,
  };
  bindings.allocate(lengths)->getHandle<int32_t>() = {
      3,
      0,
      3,
      2,
  };

  auto *SLWS = F->createSparseLengthsWeightedSum("SLWS", data, weights, indices,
                                                 lengths);
  auto *res = F->createSave("save", SLWS);
  ::glow::convertPlaceholdersToConstants(
      F, bindings, {indices, lengths, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Test RowwiseQuantizedSLWS Node.
TEST_P(OperatorStatelessTest, rowwiseQuantizedSLWSTest) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitBasicSLWSTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.01f,
                            parCloneCountOpt,
                            /* convertToRowwiseQuantization */ true);
}

static SaveNode *setupBucketNode(Function *F, PlaceholderBindings &bindings,
                                 Placeholder *input,
                                 const std::string &suffix) {
  std::vector<float> boundaries = {0.1, 2.5};

  auto *bucketize =
      F->createBucketizeNode("bucketize" + suffix, input, boundaries);
  auto *save = F->createSave("save" + suffix, bucketize);
  bindings.allocate(save->getPlaceholder());
  return save;
}

/// Check the correctness of the bucketize operator.
TEST_P(OperatorTest, Bucketize) {
  CHECK_IF_ENABLED();

  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3}, "input1", false);
  bindings_.allocate(input1)->getHandle<float>() = {2.0, 4.0, 1.0};
  auto *save1 =
      setupBucketNode(F_, bindings_, input1, /* suffix */ std::to_string(1));

  auto *input2 =
      mod_.createPlaceholder(ElemKind::FloatTy, {3, 2}, "input2", false);
  bindings_.allocate(input2)->getHandle<float>() = {2.0, 3.0, 4.0,
                                                    1.0, 2.0, 5.0};
  auto *save2 =
      setupBucketNode(F_, bindings_, input2, /* suffix */ std::to_string(2));

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Check the result of the first op:
  Tensor *result1 = bindings_.get(save1->getPlaceholder());
  Tensor expected1(ElemKind::Int32ITy, {3});
  expected1.getHandle<int32_t>() = {1, 2, 1};
  EXPECT_TRUE(expected1.isEqual(*result1));

  // Check the result of the second op:
  Tensor *result2 = bindings_.get(save2->getPlaceholder());
  Tensor expected2(ElemKind::Int32ITy, {3, 2});
  expected2.getHandle<int32_t>() = {1, 2, 2, 1, 1, 2};
  EXPECT_TRUE(expected2.isEqual(*result2));
}

/// Check the correctness of the SoftMax operator.
/// The semantic of SoftMax is
/// res_i = exp(input_i) / (exp(input_0) + ... + exp(input_N)).
TEST_P(OperatorTest, SoftMax) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 6}, "input", false);
  bindings_.allocate(input)->getHandle<float>() = {1., 3., 2.5, 5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::FloatTy, {1, 6});
  // Expected results are:
  // sum = exp(input_0) + ... + exp(input_N) = ~245.387
  // res_0 = exp(1) / sum = ~0.011
  // res_1 = exp(3) / sum = ~0.082
  // And so on.
  out.getHandle<float>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Check that the softmax operator works properly with FP16.
/// See the test that check the SoftMax operator for more details.
TEST_P(OperatorTest, FP16SoftMax) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::Float16Ty, {1, 6}, "input", false);
  bindings_.allocate(input)->getHandle<float16_t>() = {1., 3., 2.5, 5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::Float16Ty, {1, 6});
  out.getHandle<float16_t>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Check that the softmax operator works properly with BFloat16.
/// See the test that check the SoftMax operator for more details.
TEST_P(OperatorTest, BFloat16SoftMax) {
  CHECK_IF_ENABLED();

  auto *input =
      mod_.createPlaceholder(ElemKind::BFloat16Ty, {1, 6}, "input", false);
  bindings_.allocate(input)->getHandle<bfloat16_t>() = {1., 3., 2.5,
                                                        5., 4., 2.};
  auto *selected =
      mod_.createPlaceholder(ElemKind::Int64ITy, {1, 1}, "expected", false);
  auto *Pool = F_->createSoftMax("pool", input, selected);
  auto *S = F_->createSave("save", Pool);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto result = bindings_.get(S->getPlaceholder());
  Tensor out(ElemKind::BFloat16Ty, {1, 6});
  out.getHandle<bfloat16_t>() = {0.011f, 0.082f, 0.05f, 0.605f, 0.222f, 0.03f};
  EXPECT_TRUE(out.isEqual(*result, 0.001));
}

/// Verify that Quantize, Rescale, Dequantize work correctly together.
static void quantizeSimpleTest(glow::PlaceholderBindings &bindings_,
                               glow::Module &mod_, glow::Function *F_,
                               glow::ExecutionEngine &EE_, ElemKind QTy) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 1}, "input", true);
  bindings_.allocate(input)->init(Tensor::InitKind::Broadcast, 21,
                                  mod_.getPRNG());

  auto *Q =
      F_->createQuantize("quant", input, mod_.uniqueType(QTy, {1, 1}, 0.25, 4));
  auto *RS = F_->createRescaleQuantized("rescale", Q,
                                        mod_.uniqueType(QTy, {1, 1}, 0.5, 11));
  auto *D = F_->createDequantize("dequantize", RS, ElemKind::FloatTy);
  auto *save = F_->createSave("ret", D);
  auto *result = bindings_.allocate(save->getPlaceholder());

  EXPECT_EQ(F_->getNodes().size(), 4);
  EE_.compile(CompilationMode::Infer);

  EE_.run(bindings_);
  EXPECT_EQ(F_->getNodes().size(), 1);

  auto RH = result->getHandle();
  EXPECT_NEAR(RH.at({0, 0}), 21.0, 0.001);
}

TEST_P(OperatorTest, QuantizeSimpleInt8) {
  CHECK_IF_ENABLED();
  quantizeSimpleTest(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}
TEST_P(OperatorTest, QuantizeSimpleInt16) {
  CHECK_IF_ENABLED();
  quantizeSimpleTest(bindings_, mod_, F_, EE_, ElemKind::Int16QTy);
}
TEST_P(OperatorTest, QuantizeSimpleInt32) {
  CHECK_IF_ENABLED();
  quantizeSimpleTest(bindings_, mod_, F_, EE_, ElemKind::Int32QTy);
}

TEST_P(OperatorTest, LengthsToRanges) {
  CHECK_IF_ENABLED();

  /*
    LENGTHS = [1, 3, 0, 2]
    OUTPUT =  [[0, 1], [1, 3], [4, 0], [4, 2]]
  */
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {4}, "lengths", false);

  bindings_.allocate(lengths)->getHandle<int32_t>() = {1, 3, 0, 2};

  auto *R = F_->createLengthsToRanges("LTR", lengths);
  auto *S = F_->createSave("save", R);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int32ITy, {4, 2});
  expected.getHandle<int32_t>() = {
      0, 1, 1, 3, 4, 0, 4, 2,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test that LengthsRangeFill works.
TEST_P(OperatorTest, LengthsRangeFill) {
  CHECK_IF_ENABLED();

  /*
    LENGTHS = [4, 3, 1]
    OUTPUT =  [0, 1, 2, 3, 0, 1, 2, 0]
  */
  auto *lengths =
      mod_.createPlaceholder(ElemKind::Int32ITy, {3}, "lengths", false);

  bindings_.allocate(lengths)->getHandle<int32_t>() = {4, 3, 1};

  auto *LRF = F_->createLengthsRangeFill("LRF", lengths, /* maxOutputSize */ 8);
  auto *S = F_->createSave("save", LRF);
  bindings_.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  Tensor &result = *bindings_.get(S->getPlaceholder());
  Tensor expected(ElemKind::Int32ITy, {8});
  expected.getHandle<int32_t>() = {0, 1, 2, 3, 0, 1, 2, 0};

  EXPECT_TRUE(expected.isEqual(result));
}

/// Helper for testing BatchOneHot with different \p DTy.
template <typename DataType>
void batchOneHotTest(glow::PlaceholderBindings &bindings, glow::Module &mod,
                     glow::Function *F, glow::ExecutionEngine &EE,
                     ElemKind DTy) {
  /*
    DATA = [[5, 0], [11, 3], [0, 5]]
    LENGTHS = [4, 2]
    VALUES = [5, 0, 11, 0, 5, 0]
    OUTPUT =  [[1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0]]
  */
  auto *data =
      createPlaceholderConditionallyQuantized(mod, DTy, {3, 2}, "data", false);
  auto *lengths =
      mod.createPlaceholder(ElemKind::Int32ITy, {2}, "lengths", false, "N");
  auto *values = createPlaceholderConditionallyQuantized(mod, DTy, {6},
                                                         "values", false, "N");

  bindings.allocate(data)->getHandle<DataType>() = {5, 0, 11, 3, 0, 5};
  bindings.allocate(lengths)->getHandle<int32_t>() = {4, 2};
  bindings.allocate(values)->getHandle<DataType>() = {5, 0, 11, 0, 5, 0};

  auto *R = F->createBatchOneHot("BOH", data, lengths, values);
  auto *S = F->createSave("save", R);
  bindings.allocate(S->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  Tensor &result = *bindings.get(S->getPlaceholder());
  auto expected = createTensorConditionallyQuantized(DTy, {3, 6});
  expected.getHandle<DataType>() = {
      1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0,
  };

  EXPECT_TRUE(expected.isEqual(result));
}

/// Test BatchOneHot with Float data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataFloat) {
  CHECK_IF_ENABLED();
  batchOneHotTest<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test BatchOneHot with Float16 data and Int32 Lengths
TEST_P(OperatorTest, BatchOneHotDataFloat16) {
  CHECK_IF_ENABLED();
  batchOneHotTest<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test BatchOneHot with BFloat16 data and Int32 Lengths
TEST_P(OperatorTest, BatchOneHotDataBFloat16) {
  CHECK_IF_ENABLED();
  batchOneHotTest<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Test BatchOneHot with Int64 data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataInt64) {
  CHECK_IF_ENABLED();
  batchOneHotTest<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy);
}

/// Test BatchOneHot with Int32 data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataInt32) {
  CHECK_IF_ENABLED();
  batchOneHotTest<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy);
}

/// Test BatchOneHot with Int8 data and Int32 Lengths.
TEST_P(OperatorTest, BatchOneHotDataInt8) {
  CHECK_IF_ENABLED();
  batchOneHotTest<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Modulo with Int64 Tensors with SignFollowDivisor off.
TEST_P(OperatorTest, ModuloInt64NoSignFollow) {
  CHECK_IF_ENABLED();

  auto *src = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int64_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = false;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  std::vector<int64_t> expectedResults = {-1, 0, -2, -1, 0, -2, -1, 0,
                                          1,  2, 0,  1,  2, 0,  1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Modulo with Int64 Tensors with SignFollowDivisor on.
TEST_P(OperatorTest, ModuloInt64SignFollow) {
  CHECK_IF_ENABLED();

  auto *src = mod_.createPlaceholder(ElemKind::Int64ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int64_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = true;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int64_t>();

  std::vector<int64_t> expectedResults = {2, 0, 1, 2, 0, 1, 2, 0,
                                          1, 2, 0, 1, 2, 0, 1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Modulo with Int32 Tensors with SignFollowDivisor off.
TEST_P(OperatorTest, ModuloInt32NoSignFollow) {
  CHECK_IF_ENABLED();
#define TENSORTYPE int32_t
  auto *src = mod_.createPlaceholder(ElemKind::Int32ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int32_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = false;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int32_t>();

  std::vector<int32_t> expectedResults = {-1, 0, -2, -1, 0, -2, -1, 0,
                                          1,  2, 0,  1,  2, 0,  1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Modulo with Int32 Tensors with SignFollowDivisor off.
TEST_P(OperatorTest, ModuloInt32SignFollow) {
  CHECK_IF_ENABLED();

  auto *src = mod_.createPlaceholder(ElemKind::Int32ITy, {3, 5}, "src", false);
  auto srcH = bindings_.allocate(src)->getHandle<int32_t>();

  srcH = {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

  int64_t divisor = 3;
  bool signFollowDivisor = true;

  auto *modulo = F_->createModulo("mod", src, divisor, signFollowDivisor);
  auto *result = F_->createSave("save", modulo);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resultH = bindings_.get(result->getPlaceholder())->getHandle<int32_t>();

  std::vector<int32_t> expectedResults = {2, 0, 1, 2, 0, 1, 2, 0,
                                          1, 2, 0, 1, 2, 0, 1};
  ASSERT_EQ(expectedResults.size(), resultH.size());

  for (size_t i = 0, end = expectedResults.size(); i < end; ++i) {
    EXPECT_EQ(resultH.raw(i), expectedResults.at(i));
  }
}

/// Helper to test DotProduct1D using \p DTy.
template <typename DataType>
static void testDotProduct1D(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  // Input tensors.
  constexpr dim_t kDataSize = 10;
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, {kDataSize}, "X",
                                                    false, "N");
  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, {kDataSize}, "Y",
                                                    false, "N");
  auto XH = bindings.allocate(X)->getHandle<DataType>();
  auto YH = bindings.allocate(Y)->getHandle<DataType>();

  // Fill inputs with random values.
  XH.randomize(-10.0, 10.0, mod.getPRNG());
  YH.randomize(-10.0, 10.0, mod.getPRNG());

  // Compute expected output.
  auto expected = createTensorConditionallyQuantized(DTy, {kDataSize});
  auto expectedH = expected.getHandle<DataType>();

  for (dim_t i = 0; i < kDataSize; ++i) {
    expectedH.at({i}) = XH.at({i}) * YH.at({i});
  }

  // Compile and run the model.
  auto *dotProduct = F->createDotProduct("prod", X, Y);
  auto *result = F->createSave("save", dotProduct);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto actualH = bindings.get(result->getPlaceholder())->getHandle<DataType>();

  // Check that the output tensor is the same as the expected output.
  EXPECT_EQ(actualH.size(), expectedH.size());
  for (std::size_t i = 0; i < actualH.size(); ++i) {
    EXPECT_NEAR(actualH.raw(i), expectedH.raw(i), 0.00001);
  }
}

/// Test a DotProduct operator with 1D inputs, using FloatTy.
TEST_P(OperatorTest, dotProduct1D_Float) {
  CHECK_IF_ENABLED();
  testDotProduct1D<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

/// Test a DotProduct operator with 1D inputs, using Float16Ty.
TEST_P(OperatorTest, dotProduct1D_Float16) {
  CHECK_IF_ENABLED();
  testDotProduct1D<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

/// Test a DotProduct operator with 1D inputs, using Float16Ty.
TEST_P(OperatorTest, dotProduct1D_BFloat16) {
  CHECK_IF_ENABLED();
  testDotProduct1D<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

/// Test a DotProduct operator with 1D inputs, using Int8Ty.
TEST_P(OperatorTest, dotProduct1D_Int8) {
  CHECK_IF_ENABLED();
  testDotProduct1D<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

// Test a BatchedPairwiseDotProduct operator.
TEST_P(OperatorTest, batchedPairwiseDotProduct) {
  CHECK_IF_ENABLED();

  // Input tensors.
  constexpr dim_t kBatchSize = 2;
  constexpr dim_t kVectorSize = 6;

  auto *W = createPlaceholderConditionallyQuantized(
      mod_, ElemKind::FloatTy, {kBatchSize, kVectorSize}, "X", false);
  auto *X = createPlaceholderConditionallyQuantized(
      mod_, ElemKind::FloatTy, {kBatchSize, kVectorSize}, "X", false);
  auto *Y = createPlaceholderConditionallyQuantized(
      mod_, ElemKind::FloatTy, {kBatchSize, kVectorSize}, "Y", false);
  auto *Z = createPlaceholderConditionallyQuantized(
      mod_, ElemKind::FloatTy, {kBatchSize, kVectorSize}, "Z", false);
  auto WH = bindings_.allocate(W)->getHandle();
  auto XH = bindings_.allocate(X)->getHandle();
  auto YH = bindings_.allocate(Y)->getHandle();
  auto ZH = bindings_.allocate(Z)->getHandle();

  // Fill inputs with random values.

  WH = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
  XH = {2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3};
  YH = {3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4};
  ZH = {4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5};

  // Compute expected output.
  auto expected =
      createTensorConditionallyQuantized(ElemKind::FloatTy, {kBatchSize, 6});
  auto expectedH = expected.getHandle();

  expectedH = {12, 18, 36, 24, 48, 72, 36, 48, 72, 60, 90, 120};

  // Compile and run the model.
  auto *pairwiseDotProduct =
      F_->createBatchedPairwiseDotProduct("prod", {W, X, Y, Z});
  auto *result = F_->createSave("save", pairwiseDotProduct);
  bindings_.allocate(result->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto actualH = bindings_.get(result->getPlaceholder())->getHandle();

  // Check that the output tensor is the same as the expected output.
  EXPECT_TRUE(actualH.size() == expectedH.size());
  EXPECT_TRUE(actualH.getType().isEqual(expectedH.getType()));
  for (std::size_t i = 0; i < actualH.size(); ++i) {
    EXPECT_NEAR(actualH.raw(i), expectedH.raw(i), 0.00001);
  }
}

// Test an ElementwiseLinear operator with both axis = 0 and axis = 1
// arguments.
TEST_P(OperatorTest, elementwiseLinear) {
  CHECK_IF_ENABLED();

  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 20;

  // Create and allocate input placeholders.
  auto *X =
      mod_.createPlaceholder(ElemKind::FloatTy, {kCols, kRows}, "X", false);
  auto *w = mod_.createPlaceholder(ElemKind::FloatTy, {kCols}, "w", false);
  auto *b = mod_.createPlaceholder(ElemKind::FloatTy, {kCols}, "b", false);

  auto XH = bindings_.allocate(X)->getHandle();
  auto wH = bindings_.allocate(w)->getHandle();
  auto bH = bindings_.allocate(b)->getHandle();

  // Fill inputs with random values.
  XH.randomize(-3.0, 3.0, mod_.getPRNG());
  wH.randomize(-3.0, 3.0, mod_.getPRNG());
  bH.randomize(-3.0, 3.0, mod_.getPRNG());

  // Create two separate models to test behaviour when axis = 0 and axis = 1.
  // For the test with axis = 0, the 0th dimension of X, w, and b must match.
  auto *elementwiseLinearAxisZero =
      F_->createElementwiseLinear("elAxisZero", X, w, b, /*axis=*/0);
  auto *resultAxisZero =
      F_->createSave("saveAxisZero", elementwiseLinearAxisZero);
  bindings_.allocate(resultAxisZero->getPlaceholder());

  // For the test with axis = 1, the 1st dimension of X must match the 0th
  // dimension of w and b must match, so a transpose is needed.
  auto *XT = F_->createTranspose("XT", X, {1, 0});
  auto *elementwiseLinearAxisOne =
      F_->createElementwiseLinear("elAxisOne", XT, w, b, /*axis=*/1);
  auto *resultAxisOne = F_->createSave("saveAxisOne", elementwiseLinearAxisOne);
  bindings_.allocate(resultAxisOne->getPlaceholder());

  // Compile and run the model.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  auto resAxisZeroH =
      bindings_.get(resultAxisZero->getPlaceholder())->getHandle();
  auto resAxisOneH =
      bindings_.get(resultAxisOne->getPlaceholder())->getHandle();

  // Results should be the same shape as X/XT.
  ASSERT_EQ(resAxisZeroH.size(), XH.size());
  ASSERT_EQ(resAxisOneH.size(), (XT->getResult().getType())->size());

  // Compute the expected output and check that the model outputs match.
  for (dim_t i = 0; i < resAxisZeroH.dims()[0]; ++i) {
    for (dim_t j = 0; j < resAxisZeroH.dims()[1]; ++j) {
      float expected = (XH.at({i, j}) * wH.at({i})) + bH.at({i});
      EXPECT_NEAR(resAxisZeroH.at({i, j}), expected, 0.00001);
      EXPECT_NEAR(resAxisOneH.at({j, i}), expected, 0.00001);
    }
  }
}

/// Helper to test DotProduct2D using \p DTy.
template <typename DataType>
static void testDotProduct2D(glow::PlaceholderBindings &bindings,
                             glow::Module &mod, glow::Function *F,
                             glow::ExecutionEngine &EE, ElemKind DTy) {
  // Input tensors.
  constexpr dim_t kRows = 10;
  constexpr dim_t kCols = 14;
  auto *X = createPlaceholderConditionallyQuantized(mod, DTy, {kRows, kCols},
                                                    "X", false);
  auto *Y = createPlaceholderConditionallyQuantized(mod, DTy, {kRows, kCols},
                                                    "Y", false);
  auto XH = bindings.allocate(X)->getHandle<DataType>();
  auto YH = bindings.allocate(Y)->getHandle<DataType>();

  // Fill inputs with random values.
  XH.randomize(-3.0, 3.0, mod.getPRNG());
  YH.randomize(-3.0, 3.0, mod.getPRNG());

  // Compute expected output.
  auto expected = createTensorConditionallyQuantized(DTy, {kRows});
  auto expectedH = expected.getHandle<DataType>();

  for (dim_t i = 0; i < kRows; ++i) {
    DataType dotProduct = 0.0f;

    // Compute dot product of the i-th row of X and Y.
    for (dim_t j = 0; j < kCols; ++j) {
      dotProduct += (XH.at({i, j}) * YH.at({i, j}));
    }

    expectedH.at({i}) = dotProduct;
  }

  // Compile and run the model.
  auto *dotProduct = F->createDotProduct("prod", X, Y);
  auto *result = F->createSave("save", dotProduct);
  bindings.allocate(result->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto actualH = bindings.get(result->getPlaceholder())->getHandle<DataType>();

  // Check that the output tensor is the same as the expected output.
  EXPECT_EQ(actualH.size(), expectedH.size());
  for (std::size_t i = 0; i < actualH.size(); ++i) {
    EXPECT_NEAR(actualH.raw(i), expectedH.raw(i), 0.00001);
  }
}

// Test a DotProduct operator with 2D inputs, using FloatTy.
TEST_P(OperatorTest, dotProduct2D_Float) {
  CHECK_IF_ENABLED();
  testDotProduct2D<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

// Test a DotProduct operator with 2D inputs, using Float16Ty.
TEST_P(OperatorTest, dotProduct2D_Float16) {
  CHECK_IF_ENABLED();
  testDotProduct2D<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

// Test a DotProduct operator with 2D inputs, using BFloat16Ty.
TEST_P(OperatorTest, dotProduct2D_BFloat16) {
  CHECK_IF_ENABLED();
  testDotProduct2D<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty);
}

// Test a DotProduct operator with 2D inputs, using Int8QTy.
TEST_P(OperatorTest, dotProduct2D_Int8) {
  CHECK_IF_ENABLED();
  testDotProduct2D<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

/// Helper to test BatchBoxCox using \p DTy.
template <typename DataType>
static void testBatchBoxCox(glow::PlaceholderBindings &bindings,
                            glow::Module &mod, glow::Function *F,
                            glow::ExecutionEngine &EE, ElemKind DTy,
                            float allowedError = 0.0001f, float maxRange = 5.0f,
                            float maxLambda2 = 2.0f) {
  // Input tensors.
  const dim_t kRows = 10;
  const dim_t kCols = 5;
  auto *data = mod.createPlaceholder(DTy, {kRows, kCols}, "data",
                                     /* isTrainable */ false);
  auto *lambda1 = mod.createPlaceholder(DTy, {kCols}, "lambda1",
                                        /* isTrainable */ false);
  auto *lambda2 = mod.createPlaceholder(DTy, {kCols}, "lambda2",
                                        /* isTrainable */ false);
  auto dataH = bindings.allocate(data)->getHandle<DataType>();
  auto lambda1H = bindings.allocate(lambda1)->getHandle<DataType>();
  auto lambda2H = bindings.allocate(lambda2)->getHandle<DataType>();

  // Fill inputs with random values.
  dataH.randomize(0.0, maxRange, mod.getPRNG());
  lambda1H.randomize(1.0, 2.0, mod.getPRNG());
  lambda2H.randomize(1.0, maxLambda2, mod.getPRNG());

  // Zero out every other element to lambda1 to test that case of the transform.
  for (dim_t i = 0; i < kCols; i += 2) {
    lambda1H.at({i}) = 0;
  }

  const float epsilon = std::is_same<float, DataType>::value
                            ? std::numeric_limits<float>::min()
                            : 1e-6f;

  // Construct the graph for the backend to run.
  auto *BBC = F->createBatchBoxCox("bbc", data, lambda1, lambda2, epsilon);
  auto *save = F->createSave("save", BBC);
  auto resultH =
      bindings.allocate(save->getPlaceholder())->getHandle<DataType>();

  // Compile and run the model, setting results in tensor backed by resultH.
  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  // Compute expected output here on the host to compare results.
  Tensor expected(DTy, {kRows, kCols});
  auto expectedH = expected.getHandle<DataType>();

  for (dim_t i = 0; i < kRows; ++i) {
    for (dim_t j = 0; j < kCols; ++j) {
      float d = dataH.at({i, j});
      float l1 = lambda1H.at({j});
      float l2 = lambda2H.at({j});

      // Compute elementwise Box-Cox transform.
      float tmp = std::max(d + l2, 1e-6f);
      if (l1 == 0) {
        // Clip argument to log and pow at 1e-6 to avoid saturation.
        expectedH.at({i, j}) = std::log(tmp);
      } else {
        expectedH.at({i, j}) = (std::pow(tmp, l1) - 1) / l1;
      }
    }
  }

  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0; i < resultH.size(); ++i) {
    EXPECT_NEAR(resultH.raw(i), expectedH.raw(i), allowedError);
  }
}

/// Test that the BatchBoxCox operator works as expected in FloatTy.
TEST_P(OperatorTest, BatchBoxCox_Float) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, 0.001f);
}

/// Test that the BatchBoxCox operator works as expected in Float16Ty.
TEST_P(OperatorTest, BatchBoxCox_Large_Float16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                             0.032f, 5.0f);
}
TEST_P(OperatorTest, BatchBoxCox_Medium_Float16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                             0.016f, 3.0f);
}
TEST_P(OperatorTest, BatchBoxCox_Small_Float16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<float16_t>(bindings_, mod_, F_, EE_, ElemKind::Float16Ty,
                             0.003f, 1.0f, 1.001f);
}

/// Test that the BatchBoxCox operator works as expected in BFloat16Ty.
TEST_P(OperatorTest, BatchBoxCox_Large_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                              0.32f, 5.0f);
}
TEST_P(OperatorTest, BatchBoxCox_Medium_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                              0.16f, 3.0f);
}
TEST_P(OperatorTest, BatchBoxCox_Small_BFloat16) {
  CHECK_IF_ENABLED();
  testBatchBoxCox<bfloat16_t>(bindings_, mod_, F_, EE_, ElemKind::BFloat16Ty,
                              0.03f, 1.0f, 1.001f);
}

/// Test that Arithmetic ops work.
#define TEST_ARITH_OP_FLOAT(OP_NAME_, OP_)                                     \
  TEST_P(OperatorTest, OP_NAME_##ArithFloatTest) {                             \
    CHECK_IF_ENABLED();                                                        \
    constexpr dim_t size = 50;                                                 \
    auto *A = mod_.createPlaceholder(ElemKind::FloatTy, {size}, "A", false);   \
    auto *B = mod_.createPlaceholder(ElemKind::FloatTy, {size}, "B", false);   \
    auto *AT = bindings_.allocate(A);                                          \
    auto *BT = bindings_.allocate(B);                                          \
    auto AH = AT->getHandle();                                                 \
    auto BH = BT->getHandle();                                                 \
    AH.randomize(-10.0f, 10.0f, mod_.getPRNG());                               \
    BH.randomize(0.01f, 10.0f, mod_.getPRNG());                                \
                                                                               \
    auto *N = F_->create##OP_NAME_("op", A, B);                                \
    auto *save = F_->createSave("save", N);                                    \
    auto resultH = bindings_.allocate(save->getPlaceholder())->getHandle();    \
                                                                               \
    EE_.compile(CompilationMode::Infer);                                       \
    EE_.run(bindings_);                                                        \
                                                                               \
    for (size_t i = 0; i < size; i++) {                                        \
      EXPECT_FLOAT_EQ(resultH.raw(i), OP_(AH.raw(i), BH.raw(i)));              \
    }                                                                          \
  }

TEST_ARITH_OP_FLOAT(Add, [](float a, float b) { return a + b; })
TEST_ARITH_OP_FLOAT(Sub, [](float a, float b) { return a - b; })
TEST_ARITH_OP_FLOAT(Mul, [](float a, float b) { return a * b; })
TEST_ARITH_OP_FLOAT(Div, [](float a, float b) { return a / b; })
TEST_ARITH_OP_FLOAT(Min, [](float a, float b) { return std::min(a, b); })
TEST_ARITH_OP_FLOAT(Max, [](float a, float b) { return std::max(a, b); })

/// Helper to test ConvertTo casting from \p STy to \p DTy.
template <typename SourceType, typename DestType>
static void testConvertTo(glow::PlaceholderBindings &bindings_,
                          glow::Module &mod_, glow::Function *F_,
                          glow::ExecutionEngine &EE_, ElemKind STy,
                          ElemKind DTy) {
  // Input tensor in source type.
  dim_t shape[] = {5, 3, 20};
  auto *data = mod_.createPlaceholder(STy, shape, "data",
                                      /* isTrainable */ false);
  auto dataH = bindings_.allocate(data)->getHandle<SourceType>();
  if (STy == ElemKind::BoolTy) {
    for (dim_t i = 0; i < dataH.size(); i++) {
      dataH.raw(i) = static_cast<bool>(i % 2 == 0);
    }
  } else {
    dataH.randomize(-1000, 1000, mod_.getPRNG());
  }

  // Construct the graph for the backend to run, converting to dest type.
  auto OT = mod_.uniqueType(DTy, shape);
  auto *convert = F_->createConvertTo("convert", data, OT);
  auto *save = F_->createSave("save", convert);
  auto resultH =
      bindings_.allocate(save->getPlaceholder())->getHandle<DestType>();

  // Compile and run the model, setting results in tensor backed by resultH.
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Compute expected output here on the host to compare results.
  Tensor expected(DTy, shape);
  auto expectedH = expected.getHandle<DestType>();
  for (size_t i = 0, e = expectedH.size(); i < e; ++i) {
    expectedH.raw(i) = static_cast<DestType>(dataH.raw(i));
  }

  // Check that the output tensor is the same as the expected output.
  for (size_t i = 0, e = resultH.size(); i < e; i++) {
    const DestType exp = expectedH.raw(i);
    const DestType res = resultH.raw(i);
    if (DTy == ElemKind::FloatTy) {
      EXPECT_FLOAT_EQ(exp, res);
    } else {
      EXPECT_EQ(exp, res);
    }
  }
}

/// Test that ConvertTo operator casts correctly from one type to another.
#define TEST_CONVERT_TO(T_FROM, T_TO, DTY_FROM, DTY_TO)                        \
  TEST_P(OperatorTest, ConvertFrom_##DTY_FROM##_To_##DTY_TO) {                 \
    CHECK_IF_ENABLED();                                                        \
    testConvertTo<T_FROM, T_TO>(bindings_, mod_, F_, EE_, ElemKind::DTY_FROM,  \
                                ElemKind::DTY_TO);                             \
  }
TEST_CONVERT_TO(float, float, FloatTy, FloatTy)
TEST_CONVERT_TO(float, float16_t, FloatTy, Float16Ty)
TEST_CONVERT_TO(float, bfloat16_t, FloatTy, BFloat16Ty)
TEST_CONVERT_TO(float, int32_t, FloatTy, Int32ITy)
TEST_CONVERT_TO(float, int64_t, FloatTy, Int64ITy)
TEST_CONVERT_TO(float, bool, FloatTy, BoolTy)
TEST_CONVERT_TO(float16_t, float, Float16Ty, FloatTy)
TEST_CONVERT_TO(float16_t, float16_t, Float16Ty, Float16Ty)
TEST_CONVERT_TO(float16_t, bfloat16_t, Float16Ty, BFloat16Ty)
TEST_CONVERT_TO(float16_t, int32_t, Float16Ty, Int32ITy)
TEST_CONVERT_TO(float16_t, int64_t, Float16Ty, Int64ITy)
TEST_CONVERT_TO(bfloat16_t, float, BFloat16Ty, FloatTy)
TEST_CONVERT_TO(bfloat16_t, float16_t, BFloat16Ty, Float16Ty)
TEST_CONVERT_TO(bfloat16_t, bfloat16_t, BFloat16Ty, BFloat16Ty)
TEST_CONVERT_TO(bfloat16_t, int32_t, BFloat16Ty, Int32ITy)
TEST_CONVERT_TO(bfloat16_t, int64_t, BFloat16Ty, Int64ITy)
TEST_CONVERT_TO(int32_t, float, Int32ITy, FloatTy)
TEST_CONVERT_TO(int32_t, float16_t, Int32ITy, Float16Ty)
TEST_CONVERT_TO(int32_t, bfloat16_t, Int32ITy, BFloat16Ty)
TEST_CONVERT_TO(int32_t, int32_t, Int32ITy, Int32ITy)
TEST_CONVERT_TO(int32_t, int64_t, Int32ITy, Int64ITy)
TEST_CONVERT_TO(int64_t, float, Int64ITy, FloatTy)
TEST_CONVERT_TO(int64_t, float16_t, Int64ITy, Float16Ty)
TEST_CONVERT_TO(int64_t, bfloat16_t, Int64ITy, BFloat16Ty)
TEST_CONVERT_TO(int64_t, int32_t, Int64ITy, Int32ITy)
TEST_CONVERT_TO(int64_t, int64_t, Int64ITy, Int64ITy)
TEST_CONVERT_TO(bool, float, BoolTy, FloatTy)
TEST_CONVERT_TO(bool, float16_t, BoolTy, Float16Ty)
TEST_CONVERT_TO(bool, bfloat16_t, BoolTy, BFloat16Ty)

#undef TEST_CONVERT_TO

/// Helper to test ConvertTo casting from \p STy to \p DTy and back.
template <typename SourceType, typename DestType>
static void testConvertToAndBack(glow::PlaceholderBindings &bindings_,
                                 glow::Module &mod_, glow::Function *F_,
                                 glow::ExecutionEngine &EE_, ElemKind STy,
                                 ElemKind DTy, bool castIsNoOp) {
  // Input tensor in source type.
  dim_t shape[] = {5, 3, 20};
  auto *data = mod_.createPlaceholder(STy, shape, "data",
                                      /* isTrainable */ false);
  auto dataH = bindings_.allocate(data)->getHandle<SourceType>();
  dataH.randomize(-1000, 1000, mod_.getPRNG());

  // Construct the graph for the backend to run, converting to dest type and
  // back.
  auto IT = mod_.uniqueType(STy, shape);
  auto OT = mod_.uniqueType(DTy, shape);
  auto *convert = F_->createConvertTo("convert_forth", data, OT);
  auto *convertBack = F_->createConvertTo("convert_back", convert, IT);
  auto *save = F_->createSave("save", convertBack);
  auto resultH =
      bindings_.allocate(save->getPlaceholder())->getHandle<SourceType>();

  // Compile and run the model, setting results in tensor backed by resultH.
  EXPECT_EQ(F_->getNodes().size(), 3);
  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);
  EXPECT_EQ(F_->getNodes().size(), size_t(castIsNoOp ? 1 : 3));

  for (size_t i = 0, e = resultH.size(); i < e; i++) {
    const SourceType res = resultH.raw(i);
    const SourceType expected =
        static_cast<SourceType>(static_cast<DestType>(dataH.raw(i)));
    EXPECT_EQ(res, expected);
  }
}

/// Test that ConvertTo operator casts correctly from one type to another.
#define TEST_CAST_2WAYS(T_FROM, T_TO, DTY_FROM, DTY_TO, NOOP_CAST)             \
  TEST_P(OperatorTest, ConvertFrom_##DTY_FROM##_To_##DTY_TO##_AndBack) {       \
    CHECK_IF_ENABLED();                                                        \
    testConvertToAndBack<T_FROM, T_TO>(bindings_, mod_, F_, EE_,               \
                                       ElemKind::DTY_FROM, ElemKind::DTY_TO,   \
                                       NOOP_CAST);                             \
  }
TEST_CAST_2WAYS(float, float, FloatTy, FloatTy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(float, float16_t, FloatTy, Float16Ty, /* castIsNoOp */ false)
// FIXME: Should this test succeed?
TEST_CAST_2WAYS(float, bfloat16_t, FloatTy, BFloat16Ty, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float, int32_t, FloatTy, Int32ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float, int64_t, FloatTy, Int64ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(float16_t, float, Float16Ty, FloatTy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(float16_t, float16_t, Float16Ty, Float16Ty,
                /* castIsNoOp */ true)
TEST_CAST_2WAYS(float16_t, bfloat16_t, Float16Ty, BFloat16Ty,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(float16_t, int32_t, Float16Ty, Int32ITy,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(float16_t, int64_t, Float16Ty, Int64ITy,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(bfloat16_t, float, BFloat16Ty, FloatTy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(bfloat16_t, float16_t, BFloat16Ty, Float16Ty,
                /* castIsNoOp */ true)
TEST_CAST_2WAYS(bfloat16_t, bfloat16_t, BFloat16Ty, BFloat16Ty,
                /* castIsNoOp */ true)
TEST_CAST_2WAYS(bfloat16_t, int32_t, BFloat16Ty, Int32ITy,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(bfloat16_t, int64_t, BFloat16Ty, Int64ITy,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, float, Int32ITy, FloatTy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, float16_t, Int32ITy, Float16Ty,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, bfloat16_t, Int32ITy, BFloat16Ty,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(int32_t, int32_t, Int32ITy, Int32ITy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(int32_t, int64_t, Int32ITy, Int64ITy, /* castIsNoOp */ true)
TEST_CAST_2WAYS(int64_t, float, Int64ITy, FloatTy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, float16_t, Int64ITy, Float16Ty,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, bfloat16_t, Int64ITy, BFloat16Ty,
                /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, int32_t, Int64ITy, Int32ITy, /* castIsNoOp */ false)
TEST_CAST_2WAYS(int64_t, int64_t, Int64ITy, Int64ITy, /* castIsNoOp */ true)

#undef TEST_CAST_2WAYS

TEST_P(OperatorTest, ConvertFusedToFusedFP16) {
  CHECK_IF_ENABLED();

  // First create float data.
  Tensor fData(ElemKind::FloatTy, {20, 30});
  fData.getHandle().randomize(-10.0f, 10.0f, mod_.getPRNG());

  // Convert the float data to RWQ, with float scale/offset.
  Tensor rwqData(ElemKind::UInt8FusedQTy, {20, 30 + 2 * (dim_t)sizeof(float)},
                 1.0, 0);
  quantization::tensorFusedRowwiseQuantization<float>(fData, rwqData);

  // Create graph where we convert to using float16_t scale/offset.
  Placeholder *rwqDataPH =
      mod_.createPlaceholder(mod_.uniqueType(rwqData.getType()), "lhs", false);
  auto OT = mod_.uniqueType(ElemKind::UInt8FusedFP16QTy,
                            {20, 30 + 2 * (dim_t)sizeof(float16_t)}, 1.0, 0);
  auto *convert = F_->createConvertTo("convert", rwqDataPH, OT);
  auto *save = F_->createSave("save", convert);
  auto *resultT = bindings_.allocate(save->getPlaceholder());
  bindings_.insert(rwqDataPH, std::move(rwqData));

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings_);

  // Dequantize the resulting RWQ w/ float16_t scale/offset, and compare to the
  // original float data we started with.
  Tensor dequantResult =
      quantization::dequantizeTensor(*resultT, ElemKind::FloatTy);
  EXPECT_TRUE(dequantResult.isEqual(fData, 0.05));
}

template <typename DataType>
glow::Handle<DataType>
mulHelper(glow::PlaceholderBindings &bindings, glow::Module &mod,
          glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
          llvm::ArrayRef<DataType> lhsValues,
          llvm::ArrayRef<DataType> rhsValues, llvm::ArrayRef<dim_t> lhsDims,
          llvm::ArrayRef<dim_t> rhsDims) {
  auto *lhs = mod.createPlaceholder(DTy, lhsDims, "lhs", false);
  auto *rhs = mod.createPlaceholder(DTy, rhsDims, "rhs", false);
  bindings.allocate(lhs)->getHandle<DataType>() = lhsValues;
  bindings.allocate(rhs)->getHandle<DataType>() = rhsValues;

  auto *N = F->createMul("Mul", lhs, rhs);
  auto *save = F->createSave("save", N);
  auto *saveTensor = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  return saveTensor->getHandle<DataType>();
}

/// Check that the Mul operator behaves correctly with int32.
TEST_P(OperatorTest, mul_int32) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<int32_t, 16> xValues = {
      3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<int32_t, 16> yValues = {
      3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<int32_t> saveH =
      mulHelper<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy, xValues,
                         yValues, xDims, yDims);

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_EQ(xValues[counter] * yValues[counter],
                    saveH.at({i, j, k, f}));
          ++counter;
        }
      }
    }
  }
}

/// Check that the Mul operator behaves correctly with int64.
TEST_P(OperatorTest, mul_int64) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<int64_t, 16> xValues = {
      3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<int64_t, 16> yValues = {
      3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<int64_t> saveH =
      mulHelper<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy, xValues,
                         yValues, xDims, yDims);

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_EQ(xValues[counter] * yValues[counter],
                    saveH.at({i, j, k, f}));
          ++counter;
        }
      }
    }
  }
}
/// Check that the Mul operator behaves correctly with float.
TEST_P(OperatorTest, mul_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<float, 16> xValues = {
      3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<float, 16> yValues = {
      3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<float> saveH =
      mulHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                       yValues, xDims, yDims);

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(xValues[counter] * yValues[counter],
                          saveH.at({i, j, k, f}));
          ++counter;
        }
      }
    }
  }
}

template <typename DataType>
glow::Handle<DataType>
addHelper(glow::PlaceholderBindings &bindings, glow::Module &mod,
          glow::Function *F, glow::ExecutionEngine &EE, ElemKind DTy,
          llvm::ArrayRef<DataType> lhsValues,
          llvm::ArrayRef<DataType> rhsValues, llvm::ArrayRef<dim_t> lhsDims,
          llvm::ArrayRef<dim_t> rhsDims) {
  auto *lhs = mod.createPlaceholder(DTy, lhsDims, "lhs", false);
  auto *rhs = mod.createPlaceholder(DTy, rhsDims, "rhs", false);
  bindings.allocate(lhs)->getHandle<DataType>() = lhsValues;
  bindings.allocate(rhs)->getHandle<DataType>() = rhsValues;

  auto *N = F->createAdd("Add", lhs, rhs);
  auto *save = F->createSave("save", N);
  auto *saveTensor = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  return saveTensor->getHandle<DataType>();
}

/// Check that the Mul operator behaves correctly with int32.
TEST_P(OperatorTest, add_int32) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<int32_t, 16> xValues = {
      3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<int32_t, 16> yValues = {
      3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<int32_t> saveH =
      addHelper<int32_t>(bindings_, mod_, F_, EE_, ElemKind::Int32ITy, xValues,
                         yValues, xDims, yDims);

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_EQ(xValues[counter] + yValues[counter],
                    saveH.at({i, j, k, f}));
          ++counter;
        }
      }
    }
  }
}

/// Check that the Mul operator behaves correctly with int32.
TEST_P(OperatorTest, add_int64) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<int64_t, 16> xValues = {
      3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<int64_t, 16> yValues = {
      3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<int64_t> saveH =
      addHelper<int64_t>(bindings_, mod_, F_, EE_, ElemKind::Int64ITy, xValues,
                         yValues, xDims, yDims);

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_EQ(xValues[counter] + yValues[counter],
                    saveH.at({i, j, k, f}));
          ++counter;
        }
      }
    }
  }
}
/// Check that the Mul operator behaves correctly with int32.
TEST_P(OperatorTest, add_float) {
  CHECK_IF_ENABLED();

  llvm::SmallVector<float, 16> xValues = {
      3, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1,

      1, 2, 3, 6, 4, 5, 6, 3, 7, 8, 9, 2, 3, 5, 7, 1};

  llvm::SmallVector<float, 16> yValues = {
      3, 4, 5, 7, 2, 5, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6,

      3, 4, 5, 7, 2, 1, 0, 6, 4, 2, 1, 8, 5, 9, 2, 6};

  llvm::SmallVector<dim_t, 4> xDims = {2, 2, 4, 4};
  llvm::SmallVector<dim_t, 4> yDims = {2, 2, 4, 4};

  Handle<float> saveH =
      addHelper<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy, xValues,
                       yValues, xDims, yDims);

  int counter = 0;
  for (dim_t i = 0; i < saveH.dims()[0]; ++i) {
    for (dim_t j = 0; j < saveH.dims()[1]; ++j) {
      for (dim_t k = 0; k < saveH.dims()[2]; ++k) {
        for (dim_t f = 0; f < saveH.dims()[3]; ++f) {
          EXPECT_FLOAT_EQ(xValues[counter] + yValues[counter],
                          saveH.at({i, j, k, f}));
          ++counter;
        }
      }
    }
  }
}

static FunctionTensorPair
createAndInitLayerNormTest(glow::PlaceholderBindings &bindings,
                           glow::ExecutionEngine &EE) {
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");

  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 4, 5, 5}, "in", false);

  Tensor scaleT(ElemKind::FloatTy, {5, 5});
  scaleT.getHandle().randomize(0.0f, 1.0f, mod.getPRNG());
  Constant *scaleC = mod.createConstant("scale", std::move(scaleT));
  Tensor biasT(ElemKind::FloatTy, {5, 5});
  biasT.getHandle().randomize(0.0f, 1.0f, mod.getPRNG());
  Constant *biasC = mod.createConstant("bias", std::move(biasT));

  LayerNormalizationNode *LNN =
      F->createLayerNormalization("LN", input, scaleC, biasC, 1e-5);

  bindings.allocate(input)->getHandle().randomize(0.0f, 1.0f, mod.getPRNG());

  auto *res = F->createSave("save", LNN);
  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  auto *resultTensor = bindings.allocate(res->getPlaceholder());

  return std::make_pair(F, resultTensor);
}

/// Test LayerNorm with FloatTy.
TEST_P(OperatorStatelessTest, LayerNorm_Float) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitLayerNormTest,
                            ElemKind::FloatTy, ElemKind::FloatTy, 0.0001f,
                            parCloneCountOpt);
}

/// Test LayerNorm with Float16Ty.
TEST_P(OperatorStatelessTest, LayerNorm_Float16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitLayerNormTest,
                            ElemKind::FloatTy, ElemKind::Float16Ty, 0.01f,
                            parCloneCountOpt);
}

/// Test LayerNorm with BFloat16Ty.
TEST_P(OperatorStatelessTest, LayerNorm_BFloat16) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitLayerNormTest,
                            ElemKind::FloatTy, ElemKind::BFloat16Ty, 0.01f,
                            parCloneCountOpt);
}

/// Test LayerNorm with Int8Ty.
TEST_P(OperatorStatelessTest, LayerNorm_Int8) {
  CHECK_IF_ENABLED();
  compareAgainstInterpreter(getBackendName(), createAndInitLayerNormTest,
                            ElemKind::FloatTy, ElemKind::Int8QTy, 0.04f,
                            parCloneCountOpt);
}

static void testDequantizeFRWQ(glow::PlaceholderBindings &bindings,
                               glow::Module &mod, glow::Function *F,
                               glow::ExecutionEngine &EE, ElemKind destTy) {
  Tensor FT(ElemKind::FloatTy, {10, 20});
  FT.getHandle().randomize(-0.5, 0.5, mod.getPRNG());
  TypeRef RWQTy = mod.uniqueType(ElemKind::UInt8FusedQTy,
                                 {10, 20 + 2 * sizeof(float)}, 1.0, 0);
  Tensor RWQT(RWQTy);
  quantization::tensorFusedRowwiseQuantization<float>(FT, RWQT);

  auto *input = mod.createPlaceholder(RWQTy, "input", false);
  bindings.insert(input, std::move(RWQT));

  auto *D = F->createDequantize("dequantize", input, destTy);
  auto *save = F->createSave("ret", D);
  auto *result = bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  if (destTy == ElemKind::Float16Ty) {
    FT.convertToType(destTy);
  }
  EXPECT_TRUE(FT.isEqual(*result, 0.002f));
}

TEST_P(OperatorTest, DequantizeFRWQ_Float) {
  CHECK_IF_ENABLED();
  testDequantizeFRWQ(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}
TEST_P(OperatorTest, DequantizeFRWQ_Float16) {
  CHECK_IF_ENABLED();
  testDequantizeFRWQ(bindings_, mod_, F_, EE_, ElemKind::Float16Ty);
}

template <typename DataType>
static void testUpsample3D(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  constexpr dim_t size[4] = {3, 2, 3, 4};
  auto *input =
      createPlaceholderConditionallyQuantized(mod, DTy, size, "input", false);
  bindings.allocate(input)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());

  auto *output = F->createUpsample("Upsample", input, 3);
  auto *save = F->createSave("Save", output);
  bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = bindings.get(save->getPlaceholder())->getHandle<DataType>();
  auto inputH = bindings.get(input)->getHandle<DataType>();

  EXPECT_EQ(resultH.dims()[0], inputH.dims()[0]);
  EXPECT_EQ(resultH.dims()[1], 2 * inputH.dims()[1]);
  EXPECT_EQ(resultH.dims()[2], 2 * inputH.dims()[2]);
  EXPECT_EQ(resultH.dims()[3], 2 * inputH.dims()[3]);
  for (dim_t m = 0; m < size[0]; m++) {
    for (dim_t i = 0; i < size[1]; i++) {
      for (dim_t j = 0; j < size[2]; j++) {
        for (dim_t k = 0; k < size[3]; k++) {
          EXPECT_EQ(resultH.at({m, 2 * i + 0, 2 * j + 0, 2 * k + 0}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 0, 2 * j + 0, 2 * k + 1}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 0, 2 * j + 1, 2 * k + 0}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 0, 2 * j + 1, 2 * k + 1}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 1, 2 * j + 0, 2 * k + 0}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 1, 2 * j + 0, 2 * k + 1}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 1, 2 * j + 1, 2 * k + 0}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
          EXPECT_EQ(resultH.at({m, 2 * i + 1, 2 * j + 1, 2 * k + 1}),
                    static_cast<DataType>(inputH.at({m, i, j, k})));
        }
      }
    }
  }
}

template <typename DataType>
static void testUpsample2D(glow::PlaceholderBindings &bindings,
                           glow::Module &mod, glow::Function *F,
                           glow::ExecutionEngine &EE, ElemKind DTy) {
  constexpr dim_t size[3] = {2, 3, 4};
  auto *input =
      createPlaceholderConditionallyQuantized(mod, DTy, size, "input", false);
  bindings.allocate(input)->getHandle<DataType>().randomize(-10.0, 10.0,
                                                            mod.getPRNG());

  auto *output = F->createUpsample("Upsample", input, 2);
  auto *save = F->createSave("Save", output);
  bindings.allocate(save->getPlaceholder());

  EE.compile(CompilationMode::Infer);
  EE.run(bindings);

  auto resultH = bindings.get(save->getPlaceholder())->getHandle<DataType>();
  auto inputH = bindings.get(input)->getHandle<DataType>();

  EXPECT_EQ(resultH.dims()[0], inputH.dims()[0]);
  EXPECT_EQ(resultH.dims()[1], 2 * inputH.dims()[1]);
  EXPECT_EQ(resultH.dims()[2], 2 * inputH.dims()[2]);
  for (dim_t m = 0; m < size[0]; m++) {
    for (dim_t i = 0; i < size[1]; i++) {
      for (dim_t j = 0; j < size[2]; j++) {
        EXPECT_EQ(resultH.at({m, 2 * i + 0, 2 * j + 0}),
                  static_cast<DataType>(inputH.at({m, i, j})));
        EXPECT_EQ(resultH.at({m, 2 * i + 0, 2 * j + 1}),
                  static_cast<DataType>(inputH.at({m, i, j})));
        EXPECT_EQ(resultH.at({m, 2 * i + 1, 2 * j + 0}),
                  static_cast<DataType>(inputH.at({m, i, j})));
        EXPECT_EQ(resultH.at({m, 2 * i + 1, 2 * j + 1}),
                  static_cast<DataType>(inputH.at({m, i, j})));
      }
    }
  }
}

TEST_P(OperatorTest, Upsample3D_Float) {
  CHECK_IF_ENABLED();
  testUpsample3D<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, Upsample3D_Int8) {
  CHECK_IF_ENABLED();
  testUpsample3D<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

TEST_P(OperatorTest, Upsample2D_Float) {
  CHECK_IF_ENABLED();
  testUpsample2D<float>(bindings_, mod_, F_, EE_, ElemKind::FloatTy);
}

TEST_P(OperatorTest, Upsample2D_Int8) {
  CHECK_IF_ENABLED();
  testUpsample2D<int8_t>(bindings_, mod_, F_, EE_, ElemKind::Int8QTy);
}

INSTANTIATE_BACKEND_TEST(OperatorStatelessTest);
INSTANTIATE_BACKEND_TEST(OperatorTest);
