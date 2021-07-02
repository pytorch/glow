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

#include "glow/Backend/BackendUtils.h"
#include "glow/Backends/Interpreter/Interpreter.h"
#include "glow/Base/TensorSerialization.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/IR/IRBuilder.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Optimizer/IROptimizer/IROptimizer.h"

#include "gtest/gtest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"

#include <future>

using namespace glow;

/// An enum to indicate what type placholder it is.
enum class PlaceholderType {
  InputPlaceholder = 0,
  InputOutputPlaceholder = 1,
  OutputPlaceholder = 2,
  NonePlaceholder = 3
};

class BackendExecTest : public ::testing::TestWithParam<std::string> {
public:
  ExecutionEngine EE_{GetParam()};
};

class BackendExecStatelessTest : public BackendStatelessTest {
public:
  ExecutionEngine EE_{getBackendName()};
};

TEST(Interpreter, profileQuantizationForANetwork) {
  ExecutionEngine EE;
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2f, 0.5f, 1.3f};

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);
  auto *Ex = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "E", false);
  Node *O = F->createFullyConnected(bindings, "fc", A, 4);
  O = F->createRELU("relu", O);
  O = F->createRegression("reg", O, Ex);
  F->createSave("ret", O);

  LoweredInfoMap loweredMap;
  CompilationContext cctx{&bindings, &loweredMap};
  cctx.precisionConfig.quantMode = QuantizationMode::Profile;

  bindings.allocate(A);
  bindings.allocate(Ex);
  EE.compile(cctx);
  bindings.allocate(mod.getPlaceholders());

  // TODO: Verify histogram itself, for now just verify min and max.
  // Run inference first time and capture tensor stats.
  updateInputPlaceholders(bindings, {A}, {&inputs});
  EE.run(bindings);
  // Because we are quantizing the partitioner deleted the original function and
  // created a new one, get the new function.
  F = mod.getFunctions().front();

  QuantizationProfileNode *profile{nullptr};
  // Find QPN for node A.
  for (auto &node : F->getNodes()) {
    if (QuantizationProfileNode *QPN =
            llvm::dyn_cast<QuantizationProfileNode>(&node)) {
      Node *observedNode = QPN->getInput().getNode();
      if (observedNode == A) {
        profile = QPN;
        break;
      }
    }
  }

  EXPECT_TRUE(profile != nullptr);

  auto CI = bindings.get(profile->getComputationInfoPlaceholder())
                ->getHandle<float>();
  float min = CI.raw(0);
  float max = CI.raw(1);
  EXPECT_NEAR(0.5, min, 0.00001);
  EXPECT_NEAR(1.3, max, 0.00001);

  // Run inference for the second time with new min and max.
  inputs.getHandle() = {0.2f, 1.6f, 0.5f, 1.3f};
  updateInputPlaceholders(bindings, {A}, {&inputs});
  EE.run(bindings);
  min = CI.raw(0);
  max = CI.raw(1);
  EXPECT_NEAR(0.2, min, 0.00001);
  EXPECT_NEAR(1.6, max, 0.00001);
}

/// Creates an interpreter with a given \p name and custom instruction handler
/// \p hook. \retruns a newly created custom interpreter.
static Backend *createCustomInterpreter(llvm::StringRef name,
                                        IRInstructionProcessingFn hook) {
  auto interpreter = new Interpreter();
  interpreter->setIRInstructionProcessingHandler(hook);
  interpreter->setName(name);
  return interpreter;
}

#ifdef GLOW_WITH_CPU

/// A couple of counters to check that custom processing has happened.
static unsigned numCustomProcessedSupportedInstructions;
static unsigned numCustomProcessedUnsupportedInstructions;

/// An interceptor to be invoked when executing the interpreter instructions.
static IRInstructionProcessingFn customInterpreterHook =
    [](const Instruction *I, IRInstructionProcessingStage executionStage,
       void *ctx) -> bool {
  // Only handle instructions in the processing stage.
  if (executionStage != IRInstructionProcessingStage::PROCESSING) {
    return false;
  }
  llvm::outs() << "Intercept instruction execution: " << I << "\n";
  // This is an example of handling an instruction that is normally not
  // supported by a vanilla interpreter. This way new backends or tests can
  // extend the functionality of the interpreter and support custom
  // instructions.
  if (llvm::isa<CPUMaxSplatInst>(I)) {
    llvm::outs() << "Apply special processing for an instruction not supported "
                    "by the interpreter: "
                 << I << "\n";
    numCustomProcessedUnsupportedInstructions++;
    // Tell the backend to skip standard processing of this instruction.
    return true;
  }
  // This is an example of implementing a custom handling of an instruction that
  // is supported by a vanilla interpreter. This way new backends or tests can
  // change the behavior of the interpreter for specific instructions.
  if (llvm::isa<ElementSubInst>(I)) {
    llvm::outs() << "Apply special processing instruction: " << I << "\n";
    numCustomProcessedSupportedInstructions++;
    // Tell the backend to skip standard processing of this instruction.
    return true;
  }
  return false;
};

/// Check support for intercepting and customizing the processing of
/// instructions suppored by Interpreter.
TEST(Interpreter, customPrePostAroundProcessing) {
  // Register a custom backend.
  REGISTER_DYNAMIC_GLOW_BACKEND_FACTORY(
      CustomInterpreterFactory, Interpreter, "CustomInterpreter",
      createCustomInterpreter("CustomInterpreter", customInterpreterHook))

  ExecutionEngine EE("CustomInterpreter");
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("test");
  auto *input1 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in1", false);
  auto *input2 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in2", false);
  auto *add = F->createAdd("add", input1, input2);
  auto *sub = F->createSub("sub", add, input1);
  F->createSave("save", sub);
  PlaceholderBindings bindings;
  bindings.allocate({input1, input2});
  EE.compile(CompilationMode::Infer);
  numCustomProcessedSupportedInstructions = 0;
  numCustomProcessedUnsupportedInstructions = 0;
  // Process the function by means of the custom backend.
  EE.run(bindings);
  // Sub operation should have been processed in a custom way.
  EXPECT_EQ(numCustomProcessedSupportedInstructions, 1);
  EXPECT_EQ(numCustomProcessedUnsupportedInstructions, 0);
}

TEST(Interpreter, customHandleUnsupportedInstruction) {
  // Register a custom Interpreter-based backend.
  REGISTER_DYNAMIC_GLOW_BACKEND_FACTORY(
      CustomInterpreterFactory, Interpreter, "CustomInterpreter",
      createCustomInterpreter("CustomInterpreter", customInterpreterHook))
  // Create CPU and custom interpreter backends.
  ExecutionEngine cpuEE("CPU");
  ExecutionEngine customInterpreterEE("CustomInterpreter");
  auto *customInterpreterBackend =
      &customInterpreterEE.getBackend("CustomInterpreter");
  auto *cpuBackend = &cpuEE.getBackend("CPU");
  auto &mod = cpuEE.getModule();
  auto *F = mod.createFunction("test");
  auto *input1 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in1", false);
  auto *splatTy = mod.uniqueType(ElemKind::FloatTy, {1, 10, 10, 3});
  auto *splat = F->createSplat("splat", splatTy, 3);
  auto *maxsplat = F->createMax("max", input1, splat);
  auto *save = F->createSave("save", maxsplat);
  std::unique_ptr<PlaceholderBindings> cpuBindings(new PlaceholderBindings);
  cpuBindings->allocate({input1, save->getPlaceholder()});
  std::unique_ptr<PlaceholderBindings> customInterpreterBindings(
      new PlaceholderBindings);
  customInterpreterBindings->allocate({input1, save->getPlaceholder()});
  CompilationContext cctx;
  cctx.compMode = CompilationMode::Infer;
  FAIL_TEST_IF_ERR(glow::optimizeFunction(F, *cpuBackend, cctx));
  // Generate the low-level IR for the CPU backend.
  std::unique_ptr<IRFunction> cpuIR =
      glow::generateAndOptimizeIR(F, *cpuBackend, false);
  // Clone the low-level IR.
  auto clonedCpuIR = cpuIR->clone("newTest");
  // Compile the cloned IR for the custom Interpreter backend.
  std::unique_ptr<IRFunction> customInterpreterIR(clonedCpuIR);
  auto customInterpreterCompiledF(
      reinterpret_cast<BackendUsingGlowIR *>(customInterpreterBackend)
          ->compileIR(std::move(customInterpreterIR)));
  auto cpuCompiledF(reinterpret_cast<BackendUsingGlowIR *>(cpuBackend)
                        ->compileIR(std::move(cpuIR)));
  ExecutionContext cpuExecCtx(std::move(cpuBindings));
  // Execute on the CPU backend.
  FAIL_TEST_IF_ERR(cpuCompiledF->execute(&cpuExecCtx));
  ExecutionContext customInterpreterExecCtx(
      std::move(customInterpreterBindings));
  numCustomProcessedUnsupportedInstructions = 0;
  // Execute on the custom Interpreter backend. The usual Interpreter backend
  // would not be able to handle some of the custom IR instructions defined by
  // the CPU backend, but the custom interpreter backend can process them.
  numCustomProcessedSupportedInstructions = 0;
  numCustomProcessedUnsupportedInstructions = 0;
  FAIL_TEST_IF_ERR(
      customInterpreterCompiledF->execute(&customInterpreterExecCtx));
  EXPECT_EQ(numCustomProcessedUnsupportedInstructions, 1);
}

#endif

/// An interceptor to be invoked when executing the interpreter instructions.
/// This is similar in spirit to customInterpreterHook.
/// Please remember, the user can choose to do what she wants to with funcImpl
/// -- they can compile and call it (like CUDA), or just invoke it, if it's a
/// handle to an external function. In this case, based on funcImpl being "PLUS"
/// or not, we add the inputs and return the value in the output.
static IRInstructionProcessingFn externFnCallInterpreterHook =
    [](const Instruction *I, IRInstructionProcessingStage executionStage,
       void *ctx) -> bool {
  // Only handle instructions in the processing stage.
  if (executionStage != IRInstructionProcessingStage::PROCESSING) {
    return false;
  }

  if (llvm::isa<ExternalFunctionCallInst>(I)) {
    auto boundInterpFn = reinterpret_cast<BoundInterpreterFunction *>(ctx);
    auto EFCI = llvm::dyn_cast<ExternalFunctionCallInst>(I);
    auto funcImpl = EFCI->getFunctionImpl();

    auto output = EFCI->getDest();
    auto input1 = EFCI->getOperand(1).first;
    auto input2 = EFCI->getOperand(2).first;
    auto out = boundInterpFn->getWeightHandle<float>(output);
    auto in1 = boundInterpFn->getWeightHandle<float>(input1);
    auto in2 = boundInterpFn->getWeightHandle<float>(input2);

    // In this simple test, we check the funcImpl of the instruction is PLUS
    // or MINUS. If so, we return the sum or difference of the inputs. If it's
    // anything else we zero the output. Note that this test shows a simple use
    // of the ExternalFunctionCallInst. The user based on their needs can
    // compile, compile and invoke, or invoke an external function. PLEASE NOTE
    // HERE WE COULD HAVE INVOKED AN EXTERNAL FUNCTION OR COMPILED AND RAN CODE.
    if (funcImpl == "PLUS") {
      for (dim_t i = 0, e = out.size(); i < e; i++) {
        out.raw(i) = in1.raw(i) + in2.raw(i);
      }
    } else if (funcImpl == "MINUS") {
      for (dim_t i = 0, e = out.size(); i < e; i++) {
        out.raw(i) = in1.raw(i) - in2.raw(i);
      }
    } else {
      // Only PLUS and MINUS are supported.
      for (dim_t i = 0, e = out.size(); i < e; i++) {
        out.raw(i) = 0.0;
      }
    }
    // Tell the backend to skip standard processing of this instruction.
    return true;
  }
  return false;
};

TEST(Interpreter, ExternalFunctionCallTest) {
  // Register a custom Interpreter-based backend with a hook for handling the
  // ExternalFunctionCall instructions.
  REGISTER_DYNAMIC_GLOW_BACKEND_FACTORY(
      CustomInterpreterFactory, Interpreter, "CustomInterpreter",
      createCustomInterpreter("CustomInterpreter", externFnCallInterpreterHook))
  // Create a custom interpreter backend.
  ExecutionEngine customInterpreterEE("CustomInterpreter");
  auto *customInterpreterBackend =
      &customInterpreterEE.getBackend("CustomInterpreter");
  auto &mod = customInterpreterEE.getModule();
  auto *F = mod.createFunction("test");

  Tensor inputs(ElemKind::FloatTy, {10});
  inputs.zero();

  auto *input1 = mod.createPlaceholder(ElemKind::FloatTy, {10}, "in1", false);
  auto *input2 = mod.createPlaceholder(ElemKind::FloatTy, {10}, "in2", false);

  // For this test, we send in a toy external function. We call it plus_call.
  // The functionImpl is just a string "PLUS". Based on this string being equal
  // to "PLUS", we compute an add operation with the inputs and store it to the
  // output. PLEASE NOTE: This test is just a toy example. The user can choose
  // to do what she wants to with funcImpl -- they can compile and call it (like
  // CUDA), or just invoke it, if it's a handle to an external function, and use
  // the inputs and outputs as they see fit.

  std::string fnName = "plus_call";
  // This can be source code like OpenCL, CUDA, or a handle to a function.
  std::string fnImplPlus = "PLUS";
  std::string fnImplMinus = "MINUS";
  std::string fnImplMul = "MUL";
  std::string fnKind = "CUSTOM_OP";

  auto *extFnCallPlus = F->createExternalFunctionCall(
      "external_function_call", input1->getType(), {input1, input2}, fnName,
      fnImplPlus, fnKind);
  auto *extFnCallMinus = F->createExternalFunctionCall(
      "external_function_call", input1->getType(), {input1, input2}, fnName,
      fnImplMinus, fnKind);
  auto *extFnCallMul = F->createExternalFunctionCall(
      "external_function_call", input1->getType(), {input1, input2}, fnName,
      fnImplMul, fnKind);
  auto *savePlus = F->createSave("save", extFnCallPlus);
  auto *saveMinus = F->createSave("save", extFnCallMinus);
  auto *saveMul = F->createSave("save", extFnCallMul);

  std::unique_ptr<PlaceholderBindings> customInterpreterBindings(
      new PlaceholderBindings);
  customInterpreterBindings->allocate(
      {input1, input2, savePlus->getPlaceholder(), saveMinus->getPlaceholder(),
       saveMul->getPlaceholder()});

  // Now get the tensors and set their values.
  auto inTensor1 = customInterpreterBindings->get(input1)->getHandle<float>();
  auto inTensor2 = customInterpreterBindings->get(input2)->getHandle<float>();
  for (dim_t i = 0, e = inTensor1.size(); i < e; i++) {
    inTensor1.raw(i) = 5.0;
    inTensor2.raw(i) = 4.0;
  }

  // Generate the IR for the custom backend.
  std::unique_ptr<IRFunction> customInterpreterIR =
      glow::generateAndOptimizeIR(F, *customInterpreterBackend, false);

  auto customInterpreterCompiledF(
      reinterpret_cast<BackendUsingGlowIR *>(customInterpreterBackend)
          ->compileIR(std::move(customInterpreterIR)));

  ExecutionContext customInterpreterExecCtx(
      std::move(customInterpreterBindings));

  FAIL_TEST_IF_ERR(
      customInterpreterCompiledF->execute(&customInterpreterExecCtx));

  // Get bindings, then get the input and output tensors.
  auto *bindings = customInterpreterExecCtx.getPlaceholderBindings();
  auto in1 = bindings->get(input1)->getHandle<float>();
  auto in2 = bindings->get(input2)->getHandle<float>();
  auto outputPlus =
      bindings->get(savePlus->getPlaceholder())->getHandle<float>();
  auto outputMinus =
      bindings->get(saveMinus->getPlaceholder())->getHandle<float>();
  auto outputMul = bindings->get(saveMul->getPlaceholder())->getHandle<float>();

  // Verify the output tensors. Add and Minus should have been processed in the
  // hook. Mul is not supported, and this ouptut should be zero'd.
  for (dim_t i = 0, e = outputPlus.size(); i < e; i++) {
    EXPECT_TRUE(outputPlus.raw(i) == in1.raw(i) + in2.raw(i));
    EXPECT_TRUE(outputMinus.raw(i) == in1.raw(i) - in2.raw(i));
    EXPECT_TRUE(outputMul.raw(i) == 0.0);
  }
}

/// Check that new backends and backend factories can be registered dynamically.
TEST(Interpreter, DynamicBackendFactory) {
  // Use a static variable here, because the macro invocation below creates a
  // new class and C++ does not allow for capturing of local variables.
  static std::string backendName;
  for (unsigned i = 0; i < 16; ++i) {
    {
      backendName = "CustomInterpreter" + std::to_string(i);
      // Dynamically create a new backend factory and register it.
      REGISTER_DYNAMIC_GLOW_BACKEND_FACTORY(CustomInterpreterFactory,
                                            Interpreter, backendName,
                                            []() -> Backend * {
                                              // Dynamically create a backend
                                              // and give it a name.
                                              auto *backend = new Interpreter;
                                              backend->setName(backendName);
                                              return backend;
                                            }())
      ExecutionEngine EE(backendName);
      auto *backend = &EE.getBackend(backendName);
      ASSERT_NE(backend, nullptr);
      // Check that a new backend is registered.
      auto backends = getAvailableBackends();
      EXPECT_NE(std::find(backends.begin(), backends.end(), backendName),
                backends.end());
      // The new backend factory will be destroyed at the end of this scope.
    }
    // Check that a new backend is not registered anymore after its factory was
    // destroyed.
    auto backends = getAvailableBackends();
    EXPECT_EQ(std::find(backends.begin(), backends.end(), backendName),
              backends.end());
  }
}

/// Test that the symbol category for a symbol is properly set.
TEST(RuntimeBundle, BundleSymbolInfo) {

  ExecutionEngine EE;
  auto &mod = EE.getModule();
  PlaceholderBindings bindings;

  Tensor inputs(ElemKind::FloatTy, {1, 10, 10, 3});
  inputs.getHandle().randomize(-2, 2, mod.getPRNG());

  // Create a simple graph that has placeholders, constants, activations, and a
  // tensor_view.
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);

  auto *ex = mod.createConstant(ElemKind::Int64ITy, {1, 1}, "exp");

  auto *FC = F->createFullyConnected(bindings, "FC", input, 30);
  auto *RL = F->createRELU("RL2", FC);
  auto *SM = F->createSoftMax("sm", RL, ex);
  auto *S = F->createSave("ret", SM);
  auto *qp = F->createQuantizationProfile(bindings, "qp", input);

  EE.compile(CompilationMode::Infer);
  runtime::DAG *dag;
  ASSIGN_VALUE_OR_FAIL_TEST(dag, EE.getDAG("main"));
  assert(dag->nodes.size() > 0 && "Empty DAG list");
  auto table = dag->nodes[0]->runtimeBundle->getSymbolTable();

  // Check that placeholders and constants are correctly labelled.
  EXPECT_EQ(
      table.find(S->getPlaceholder()->getName().str())->second.symbolCategory,
      glow::runtime::SymbolCategory::Placeholder);
  EXPECT_EQ(table.find(ex->getName().str())->second.symbolCategory,
            glow::runtime::SymbolCategory::Constant);
  // Check that activations are labelled correctly.
  EXPECT_EQ(table.find("FC_res")->second.symbolCategory,
            glow::runtime::SymbolCategory::Activation);
  // Check that tensor views have the same label as their parent symbol. In this
  // case same as "input".
  EXPECT_EQ(table.find("FC_reshape2D_tensorview")->second.symbolCategory,
            glow::runtime::SymbolCategory::PlaceholderTensorView);

  // Check that placeholders and constants input/output flags are correctly set.
  EXPECT_EQ(table.find(S->getPlaceholder()->getName().str())->second.input,
            false);
  EXPECT_EQ(table.find(S->getPlaceholder()->getName().str())->second.output,
            true);
  EXPECT_EQ(table.find(ex->getName().str())->second.input, false);
  EXPECT_EQ(table.find(ex->getName().str())->second.output, false);
  EXPECT_EQ(table.find(input->getName().str())->second.input, true);
  EXPECT_EQ(table.find(input->getName().str())->second.output, false);
  // HistogramPlaceholder node is not an input node, it is an output node.
  EXPECT_EQ(
      table.find(qp->getHistogramPlaceholder()->getName().str())->second.input,
      false);
  EXPECT_EQ(
      table.find(qp->getHistogramPlaceholder()->getName().str())->second.output,
      true);
  // Check that activations are labelled correctly.
  EXPECT_EQ(table.find("FC_res")->second.input, false);
  EXPECT_EQ(table.find("FC_res")->second.output, false);
  // Check that tensor views are labelled correctly.
  EXPECT_EQ(table.find("FC_reshape2D_tensorview")->second.input, false);
  EXPECT_EQ(table.find("FC_reshape2D_tensorview")->second.output, false);
}

// Test that using a buffer in a TensorView instruction doesn't get it marked
// as an input buffer.
TEST(IR, testInputToTensorView) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  IRBuilder builder(&M);
  auto T0 = mod.uniqueType(ElemKind::FloatTy, {1024, 1024});
  auto T1 = mod.uniqueType(ElemKind::FloatTy, {512, 1024});
  auto *input0 = builder.createWeightVar(T1, "A");
  auto *input1 = builder.createWeightVar(T1, "B");
  auto *output = builder.createWeightVar(T0, "C");
  auto *tvo0 =
      builder.createTensorViewInst("outuput_view0", output, T0, {0, 0});
  auto *tvo1 =
      builder.createTensorViewInst("outuput_view1", output, T0, {512, 0});
  builder.createElementAddInst("add0", tvo0, input0, input1);
  builder.createElementAddInst("add1", tvo1, input0, input1);
  // output0 is only used as an input to a TensorView instruction, The buffer
  // should not be marked as an input buffer since that doesn't include any
  // reads of the buffer.
  EXPECT_EQ(isInput(output), false);
}

// Test the correctness of isInput.
TEST(IR, testIsInput) {
  Module mod;
  Function *F = mod.createFunction("main");
  IRFunction M(F);
  IRBuilder builder(&M);
  auto T0 = mod.uniqueType(ElemKind::FloatTy, {1024, 1024});
  auto T1 = mod.uniqueType(ElemKind::FloatTy, {512, 1024});
  auto *input0 = builder.createWeightVar(T1, "A");
  auto *input1 = builder.createWeightVar(T1, "B");
  auto *output0 = builder.createWeightVar(T0, "C0");
  auto *output1 = builder.createWeightVar(T0, "C1");
  auto *tvo0 =
      builder.createTensorViewInst("output_view0", output0, T0, {0, 0});
  auto *tvo1 =
      builder.createTensorViewInst("output_view1", output1, T0, {512, 0});
  // tv0 is used as src and dest in this instruction. This is a first operation
  // using output0 and it first reads from it. Thus output0 should be reported
  // as input.
  builder.createElementAddInst("add0", tvo0, tvo0, input1);
  // Write into tvo1. This is the first operation touching output1 and it is a
  // write.
  builder.createElementAddInst("add1", tvo1, input0, input1);
  // Read from tvo1 and then write into it.
  builder.createElementAddInst("add2", tvo1, tvo1, input1);
  // output is used as an input to a TensorView instruction tvo0, which doesn't
  // count. But then tvo0 is  used an input and output for the same add
  // instruction. Thus, it is an input.
  EXPECT_EQ(isInput(output0), true);
  // output1 was first written into and then read. Therefore it is not an input.
  EXPECT_EQ(isInput(output1), false);
}

// Test if the placeholders are allocated contiguously as
// Input|InputOutput|Output.
TEST(RuntimeBundle, ContiguousPlaceholder) {
  ExecutionEngine EE;
  PlaceholderBindings bindings;
  auto &mod = EE.getModule();
  Function *F = mod.createFunction("main");
  Tensor inputs(ElemKind::FloatTy, {1, 4});
  inputs.getHandle() = {1, 1.2f, 0.5f, 1.3f};

  auto *A = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "A", false);
  auto *B = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "B", false);
  auto *Ex = mod.createPlaceholder(ElemKind::FloatTy, {1, 4}, "E", false);
  auto *add = F->createAdd("add", A, Ex);
  auto *sub = F->createSub("sub", B, add);
  F->createSave("ret", sub);

  LoweredInfoMap loweredMap;
  CompilationContext cctx{&bindings, &loweredMap};
  cctx.precisionConfig.quantMode = QuantizationMode::Profile;

  bindings.allocate(A);
  bindings.allocate(Ex);
  EE.compile(cctx);
  runtime::DAG *dag;
  ASSIGN_VALUE_OR_FAIL_TEST(dag, EE.getDAG("main"));
  auto &table = dag->nodes[0]->runtimeBundle->getSymbolTable();

  std::vector<glow::runtime::RuntimeSymbolInfo> tableContainer;
  // Only check placeholders.
  for (auto v : table) {
    if (v.second.symbolCategory == glow::runtime::SymbolCategory::Placeholder) {
      tableContainer.push_back(v.second);
    }
  }
  // Sort the placeholders by offset.
  sort(tableContainer.begin(), tableContainer.end(),
       [](const glow::runtime::RuntimeSymbolInfo &a,
          const glow::runtime::RuntimeSymbolInfo &b) {
         return (a.offset < b.offset);
       });

  // Define the order of placeholders.
  auto order = [](glow::runtime::RuntimeSymbolInfo i) -> PlaceholderType {
    if (i.input) {
      if (!i.output) {
        // input only
        return PlaceholderType::InputPlaceholder;
      } else {
        // input & output
        return PlaceholderType::InputOutputPlaceholder;
      }
    } else {
      if (i.output) {
        // output only
        return PlaceholderType::OutputPlaceholder;
      } else {
        // neither
        return PlaceholderType::NonePlaceholder;
      }
    }
  };
  // The order function of placeholders should be increasing.
  PlaceholderType prev = PlaceholderType::InputPlaceholder;
  bool flag = true;
  for (auto v : tableContainer) {
    PlaceholderType tmp = order(v);
    if (tmp > prev) {
      prev = tmp;
    } else if (tmp < prev) {
      flag = false;
      break;
    }
  }

  EXPECT_EQ(flag, true);
}

TEST_P(BackendExecTest, simpleInference) {
  PlaceholderBindings bindings;

  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);
  auto *conv = F->createConv(bindings, "conv", input, 10, 5, 1, 0, 1);
  auto *res = F->createSave("save", conv);

  ::glow::convertPlaceholdersToConstants(F, bindings,
                                         {input, res->getPlaceholder()});
  bindings.allocate(input)->getHandle().randomize(-1.0, 1.0, mod.getPRNG());
  bindings.allocate(res->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings);
}

/// Utility function to create a simple network in which a tensor \p tensor is
/// dumped using the debug instrumentation mechanism using the given \p format
/// and filename \p filename. Note that the backend being tested must inherit
/// from BackendUsingGlowIR and implement the compileIR() function for this test
/// to work.
static void runDebugPrint(ExecutionEngine &EE, std::string backendName,
                          Tensor &tensor, std::string format,
                          std::string filename) {
  auto &mod = EE.getModule();
  auto ctx = glow::make_unique<ExecutionContext>();
  Function *F = mod.createFunction("main");
  auto *IV = mod.createPlaceholder(&tensor.getType(), "tensor", false);
  auto *IVTensor = ctx->getPlaceholderBindings()->allocate(IV);
  IVTensor->assign(&tensor);
  auto *save = F->createSave("save", IV);
  ctx->getPlaceholderBindings()->allocate(save->getPlaceholder());

  std::unique_ptr<Backend> backend(createBackend(backendName));
  auto IR = glow::make_unique<IRFunction>(F);
  IR->generateIR(*backend.get());
  IRBuilder(IR.get()).createDebugPrintInst("print", *IR->getWeights().begin(),
                                           format, filename);

  auto function = reinterpret_cast<BackendUsingGlowIR *>(backend.get())
                      ->compileIR(std::move(IR));

  // Since we are compiling IR by hand we cannot go through the normal EE route.
  // Create and initialize the device.
  auto config =
      glow::make_unique<runtime::DeviceConfig>(backend->getBackendName());
  std::unique_ptr<runtime::DeviceManager> device(
      runtime::DeviceManager::createDeviceManager(*config));
  EXIT_ON_ERR(device->init());
  // Load the function on the device.
  std::string name = "main";
  runtime::FunctionMapTy functionMap;
  functionMap[name] = function.get();

  std::promise<void> addPromise;
  auto fut = addPromise.get_future();
  Error addErr = Error::empty();
  device->addNetwork(&EE.getModule(), std::move(functionMap),
                     [&addPromise, &addErr](const Module *, Error err) {
                       addErr = std::move(err);
                       addPromise.set_value();
                     });
  fut.wait();
  EXIT_ON_ERR(std::move(addErr));
  // Run the function.
  std::promise<void> runPromise;
  fut = runPromise.get_future();
  Error runErr = Error::empty();
  device->runFunction(name, std::move(ctx),
                      [&runPromise, &runErr,
                       &ctx](runtime::RunIdentifierTy, Error err,
                             std::unique_ptr<ExecutionContext> contextPtr) {
                        ctx = std::move(contextPtr);
                        runErr = std::move(err);
                        runPromise.set_value();
                      });
  fut.wait();
  EXIT_ON_ERR(std::move(runErr));
}

/// Utility function to test the debug instrumentation mechanism for a tensor
/// \p tensorRef using the given \p format.
template <typename type>
static void testDebugPrint(ExecutionEngine &EE, std::string backendName,
                           Tensor &tensorRef, std::string format) {
  // Create temporary file.
  llvm::SmallString<64> path;
  auto tempFileRes = llvm::sys::fs::createTemporaryFile("tensor", ".dat", path);
  if (tempFileRes.value() != 0) {
    FAIL() << "Failed to create temp file to write into.";
  }
  // Run debug print.
  runDebugPrint(EE, backendName, tensorRef, format, path.str().str());
  // Read tensor back.
  Tensor tensorTest;
  if (format == "bin") {
    TensorSerializationOptions opts;
    opts.withType = true;
    glow::loadTensorFromBinaryFile(tensorTest, path.str(), opts);
  } else if (format == "txt") {
    TensorSerializationOptions opts;
    opts.withType = true;
    glow::loadTensorFromTextFile(tensorTest, path.str(), opts);
  } else if (format == "rawbin") {
    TensorSerializationOptions opts;
    opts.withType = false;
    tensorTest = Tensor(tensorRef.getType());
    glow::loadTensorFromBinaryFile(tensorTest, path.str(), opts);
  } else if (format == "rawtxt") {
    TensorSerializationOptions opts;
    opts.withType = false;
    tensorTest = Tensor(tensorRef.getType());
    glow::loadTensorFromTextFile(tensorTest, path.str(), opts);
  } else {
    FAIL() << "Invalid DebugPrint format!";
  }
  // Remove temporary file.
  llvm::sys::fs::remove(path);
  // Compare the two tensors.
  EXPECT_EQ(tensorRef.getType(), tensorTest.getType());
  auto handleRef = tensorRef.getHandle<type>();
  auto handleTest = tensorTest.getHandle<type>();
  EXPECT_EQ(handleRef.size(), handleTest.size());
  EXPECT_EQ(handleRef.actualSize(), handleTest.actualSize());
  for (size_t idx = 0; idx < tensorTest.actualSize(); idx++) {
    EXPECT_EQ(handleTest.raw(idx), handleRef.raw(idx));
  }
}

/// Test dumping to console.
TEST_P(BackendExecStatelessTest, DebugPrint_Console) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::FloatTy, {4});
  tensorRef.getHandle<float>() = {1, 2, 3, 4};
  runDebugPrint(EE_, getBackendName(), tensorRef, "console", "");
}

/// Test dumping to file in binary format.
TEST_P(BackendExecStatelessTest, DebugPrint_Bin_FloatTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::FloatTy, {4});
  tensorRef.getHandle<float>() = {1, 2, 3, 4};
  testDebugPrint<float>(EE_, getBackendName(), tensorRef, "bin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Bin_Int8QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int8QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int8_t>() = {1, 2, 3, 4};
  testDebugPrint<int8_t>(EE_, getBackendName(), tensorRef, "bin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Bin_Int16QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int16QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int16_t>() = {1, 2, 3, 4};
  testDebugPrint<int16_t>(EE_, getBackendName(), tensorRef, "bin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Bin_Int32QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "bin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Bin_Int32ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32ITy, {4});
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "bin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Bin_Int64ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int64ITy, {4});
  tensorRef.getHandle<int64_t>() = {1, 2, 3, 4};
  testDebugPrint<int64_t>(EE_, getBackendName(), tensorRef, "bin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Bin_BoolTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::BoolTy, {4});
  tensorRef.getHandle<bool>() = {0, 1, 0, 1};
  testDebugPrint<bool>(EE_, getBackendName(), tensorRef, "bin");
}

/// Test dumping to file in text format.
TEST_P(BackendExecStatelessTest, DebugPrint_Txt_FloatTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::FloatTy, {4});
  tensorRef.getHandle<float>() = {1, 2, 3, 4};
  testDebugPrint<float>(EE_, getBackendName(), tensorRef, "txt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Txt_Int8QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int8QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int8_t>() = {1, 2, 3, 4};
  testDebugPrint<int8_t>(EE_, getBackendName(), tensorRef, "txt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Txt_Int16QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int16QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int16_t>() = {1, 2, 3, 4};
  testDebugPrint<int16_t>(EE_, getBackendName(), tensorRef, "txt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Txt_Int32QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "txt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Txt_Int32ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32ITy, {4});
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "txt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Txt_Int64ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int64ITy, {4});
  tensorRef.getHandle<int64_t>() = {1, 2, 3, 4};
  testDebugPrint<int64_t>(EE_, getBackendName(), tensorRef, "txt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_Txt_BoolTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::BoolTy, {4});
  tensorRef.getHandle<bool>() = {0, 1, 0, 1};
  testDebugPrint<bool>(EE_, getBackendName(), tensorRef, "txt");
}

/// Test dumping to file in raw binary format.
TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_FloatTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::FloatTy, {4});
  tensorRef.getHandle<float>() = {1, 2, 3, 4};
  testDebugPrint<float>(EE_, getBackendName(), tensorRef, "rawbin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_Int8QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int8QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int8_t>() = {1, 2, 3, 4};
  testDebugPrint<int8_t>(EE_, getBackendName(), tensorRef, "rawbin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_Int16QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int16QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int16_t>() = {1, 2, 3, 4};
  testDebugPrint<int16_t>(EE_, getBackendName(), tensorRef, "rawbin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_Int32QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "rawbin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_Int32ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32ITy, {4});
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "rawbin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_Int64ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int64ITy, {4});
  tensorRef.getHandle<int64_t>() = {1, 2, 3, 4};
  testDebugPrint<int64_t>(EE_, getBackendName(), tensorRef, "rawbin");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawBin_BoolTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::BoolTy, {4});
  tensorRef.getHandle<bool>() = {0, 1, 0, 1};
  testDebugPrint<bool>(EE_, getBackendName(), tensorRef, "rawbin");
}

/// Test dumping to file in raw text format.
TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_FloatTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::FloatTy, {4});
  tensorRef.getHandle<float>() = {1, 2, 3, 4};
  testDebugPrint<float>(EE_, getBackendName(), tensorRef, "rawtxt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_Int8QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int8QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int8_t>() = {1, 2, 3, 4};
  testDebugPrint<int8_t>(EE_, getBackendName(), tensorRef, "rawtxt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_Int16QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int16QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int16_t>() = {1, 2, 3, 4};
  testDebugPrint<int16_t>(EE_, getBackendName(), tensorRef, "rawtxt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_Int32QTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32QTy, {4}, 1.0, 0);
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "rawtxt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_Int32ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int32ITy, {4});
  tensorRef.getHandle<int32_t>() = {1, 2, 3, 4};
  testDebugPrint<int32_t>(EE_, getBackendName(), tensorRef, "rawtxt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_Int64ITy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::Int64ITy, {4});
  tensorRef.getHandle<int64_t>() = {1, 2, 3, 4};
  testDebugPrint<int64_t>(EE_, getBackendName(), tensorRef, "rawtxt");
}

TEST_P(BackendExecStatelessTest, DebugPrint_RawTxt_BoolTy) {
  ENABLED_BACKENDS("CPU", "Interpreter");
  Tensor tensorRef(ElemKind::BoolTy, {4});
  tensorRef.getHandle<bool>() = {0, 1, 0, 1};
  testDebugPrint<bool>(EE_, getBackendName(), tensorRef, "rawtxt");
}

/// Test the compile method on the backend completes without error when
/// collectConstants is false.
TEST_P(BackendExecTest, CompileWithoutConstants) {
  Module mod;
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = bindings.allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  bindings.allocate(save->getPlaceholder());
  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  BackendOptions opts;
  opts.collectConstants = false;
  auto function = EXIT_ON_ERR(backend->compile(F, opts));
}

/// Test that the runtimeBundle includes only symbols from its function and not
/// the whole module.
TEST_P(BackendExecTest, BundleFunctionSymbolsOnly) {
  Module mod;
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *X = mod.createConstant(ElemKind::FloatTy, {3}, "X");
  X->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  bindings.allocate(save->getPlaceholder());
  PlaceholderBindings bindings2;
  Function *F2 = mod.createFunction("main2");
  auto *X2 = mod.createConstant(ElemKind::FloatTy, {3}, "X2");
  X2->getHandle() = {1., 2., 3.};
  auto *pow2 = F2->createPow("Pow2", X2, 2.0);
  auto *save2 = F2->createSave("save2", pow2);
  bindings2.allocate(save2->getPlaceholder());

  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  auto function = EXIT_ON_ERR(backend->compile(F));
  auto function2 = EXIT_ON_ERR(backend->compile(F2));
  auto table1 = function->getRuntimeBundle().getSymbolTable();
  auto table2 = function2->getRuntimeBundle().getSymbolTable();
  /// Make sure no symbol in table1 is in table2.
  for (auto sym : table1) {
    auto it = table2.find(sym.first);
    EXPECT_TRUE(it == table2.end());
  }
}

/// Test that a shared constant is in the bundle of both functions.
TEST_P(BackendExecTest, BundleSharedConstant) {
  Module mod;
  PlaceholderBindings bindings;
  Function *F = mod.createFunction("main");
  auto *X = mod.createConstant(ElemKind::FloatTy, {3}, "X");
  X->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  bindings.allocate(save->getPlaceholder());
  PlaceholderBindings bindings2;
  Function *F2 = mod.createFunction("main2");
  auto *pow2 = F2->createPow("Pow2", X, 2.0);
  auto *save2 = F2->createSave("save2", pow2);
  bindings2.allocate(save2->getPlaceholder());

  std::unique_ptr<Backend> backend(createBackend(GetParam()));
  auto function = EXIT_ON_ERR(backend->compile(F));
  auto function2 = EXIT_ON_ERR(backend->compile(F2));
  auto table1 = function->getRuntimeBundle().getSymbolTable();
  auto table2 = function2->getRuntimeBundle().getSymbolTable();
  /// Make sure X is in both tables.
  auto it = table1.find(X->getName().str());
  auto it2 = table2.find(X->getName().str());
  EXPECT_TRUE(it != table1.end());
  EXPECT_TRUE(it2 != table2.end());
}

/// Test compiling a vector of functions completes without error.
TEST_P(BackendExecTest, compileVectorOfFunctions) {
  Module mod;
  std::vector<Function *> functions;
  llvm::StringMap<BackendOptions> optsMap;
  BackendOptions opts;

  for (unsigned int i = 0; i < 3; i++) {
    Function *F = mod.createFunction("function" + std::to_string(i));
    auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3},
                                    "X" + std::to_string(i), false);
    auto *pow = F->createPow("Pow" + std::to_string(i), X, 2.0);
    F->createSave("save" + std::to_string(i), pow);
    functions.push_back(F);
    optsMap.insert({F->getName(), opts});
  }
  std::unique_ptr<Backend> backend(createBackend(GetParam()));

  auto functionOrErr = backend->compileFunctions(functions, optsMap);
  ASSERT_TRUE((bool)functionOrErr);
}

/// This test checks that we can compile a function without depending on the
/// graph representation. We compile some function and then delete the function.
/// Later we execute the code and check that things work.
TEST_P(BackendExecTest, decoupleCodegenFromGraph) {
  auto &mod = EE_.getModule();
  PlaceholderBindings bindings;

  Function *F = mod.createFunction("main");
  auto *X = mod.createPlaceholder(ElemKind::FloatTy, {3}, "X", false);
  auto *XTensor = bindings.allocate(X);
  XTensor->getHandle() = {1., 2., 3.};
  auto *pow = F->createPow("Pow1", X, 2.0);
  auto *save = F->createSave("save", pow);
  auto *saveTensor = bindings.allocate(save->getPlaceholder());
  EE_.compile(CompilationMode::Infer);

  // Erase all of the functions to ensure that the compiled code does not
  // depend on the graph.
  mod.eraseFunctions();

  // We can run the compiled code without having the graph representation
  // around.
  EE_.run(bindings);

  auto HX = saveTensor->getHandle();
  EXPECT_NEAR(HX.at({0}), 1, 1E-5);
  EXPECT_NEAR(HX.at({1}), 4, 1E-5);
  EXPECT_NEAR(HX.at({2}), 9, 1E-5);
}

/// Check that we can pass information to the execution engine using Placeholder
/// variables and read it back using Save nodes (in variables).
TEST_P(BackendExecTest, simplePlaceholderValue) {
  Tensor data{99.0, 35.0, 2.0, 3.0};
  auto &mod = EE_.getModule();
  Function *F = mod.createFunction("main");
  auto *input = mod.createPlaceholder(ElemKind::FloatTy, {4}, "input", false);
  PlaceholderBindings bindings({input}, {&data});
  SaveNode *S = F->createSave("ret", input);
  auto *STensor = bindings.allocate(S->getPlaceholder());

  EE_.compile(CompilationMode::Infer);
  EE_.run(bindings);
  EXPECT_TRUE(STensor->isEqual(data));
}

/// Add and compile a network, then add and compile another so that the first
/// CompiledFunction does not know about every Placeholder in the module.
TEST_P(BackendExecTest, compileThenAddNetwork) {
  PlaceholderBindings bindings1, bindings2;

  auto &mod = EE_.getModule();
  Tensor inputs(ElemKind::FloatTy, {1, 10, 10, 3});
  inputs.getHandle().randomize(-2, 2, mod.getPRNG());

  // Create a simple graph that uses some placeholders.
  Function *F = mod.createFunction("main");
  auto *input =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);

  auto *FC = F->createFullyConnected(bindings1, "FC", input, 30);
  auto *RL = F->createRELU("RL2", FC);
  auto *S = F->createSave("ret", RL);

  Placeholder *FC_weights =
      llvm::dyn_cast<Placeholder>(FC->getWeights().getNode());

  // Recreate that graph in a different Function.
  Function *F2 = mod.createFunction("other");
  auto *input2 =
      mod.createPlaceholder(ElemKind::FloatTy, {1, 10, 10, 3}, "in", false);

  auto *FC2 = F2->createFullyConnected(bindings2, "FC", input2, 30);

  // FC2 now has random weights we replace with FC1's weights so the output is
  // the same.
  Placeholder *FC2_weights =
      llvm::dyn_cast<Placeholder>(FC2->getWeights().getNode());
  bindings2.get(FC2_weights)->assign(bindings1.get(FC_weights));

  auto *RL2 = F2->createRELU("RL2", FC2);
  auto *S2 = F2->createSave("ret", RL2);

  convertPlaceholdersToConstants(F, bindings1, {input, S->getPlaceholder()});
  convertPlaceholdersToConstants(F2, bindings2, {input2, S2->getPlaceholder()});
  EE_.compile(CompilationMode::Infer);

  // Allocate all placeholders.
  bindings1.allocate(mod.getPlaceholders());
  bindings2.allocate(mod.getPlaceholders());
  updateInputPlaceholders(bindings1, {input}, {&inputs});
  updateInputPlaceholders(bindings2, {input2}, {&inputs});

  EE_.run(bindings1, "main");
  EE_.run(bindings2, "other");

  // The graphs were the same so their outputs should be as well.
  EXPECT_TRUE(bindings1.get(S->getPlaceholder())
                  ->isEqual(*bindings2.get(S2->getPlaceholder())));
}

/// Test the basic functionality of the bindings.
TEST(PlaceholderBindings, basicPlaceholderBindingsTest) {
  Module mod;
  TypeRef ty = mod.uniqueType(ElemKind::FloatTy, {1, 32, 32, 3});

  Tensor T1(ty);

  // Create a bindings for some threaded execution.
  PlaceholderBindings C;

  // Create a simple graph, just to have a few placeholders.
  Function *F = mod.createFunction("main");
  auto *input1 = mod.createPlaceholder(ty, "input1", false);
  auto *input2 = mod.createPlaceholder(ty, "input2", false);
  auto *input3 = mod.createPlaceholder(ty, "input3", false);
  auto *add = F->createAdd("add", input1, input2);
  auto *save = F->createSave("ret", add);
  auto *savePlaceholder = save->getPlaceholder();
  C.allocate(savePlaceholder);

  C.insert(input1, std::move(T1));
  Tensor *I2 = C.allocate(input2);

  // Check that the right placeholders are found.
  EXPECT_TRUE(C.count(input1));
  EXPECT_TRUE(C.count(input2));
  EXPECT_TRUE(C.count(savePlaceholder));
  EXPECT_FALSE(C.count(nullptr));

  // Try to fetch some placeholders that exist and some that don't.
  auto *V1 = C.get(input1);
  auto *V2 = C.get(input2);
  auto *V3 = C.get(input3);
  auto *S = C.get(savePlaceholder);
  EXPECT_NE(V1, nullptr);
  EXPECT_NE(V2, nullptr);
  EXPECT_EQ(V3, nullptr);
  EXPECT_NE(S, nullptr);

  // The tensor that we got while allocating T2 is the same one that we got
  // while searching the bindings.
  EXPECT_EQ(I2, V2);

  // Check that all of the placeholders are allocated.
  C.allocate(input3);
  EXPECT_EQ(nullptr, C.getFirstUnallocated(mod.getPlaceholders()));

  // Check that some placeholders are unallocated.
  C.clear();
  EXPECT_NE(nullptr, C.getFirstUnallocated(mod.getPlaceholders()));

  // Check that all of the placeholders are allocated.
  C.allocate(mod.getPlaceholders());
  EXPECT_EQ(nullptr, C.getFirstUnallocated(mod.getPlaceholders()));
}

/// Check if the dump function works for Type.
TEST(BackendExecTest, dumpType) {
  Module mod;
  TypeRef tyA = mod.uniqueType(ElemKind::FloatTy, {1, 32, 32, 3});
  std::string storage;
  llvm::raw_string_ostream os(storage);
  tyA->dump(os);
  std::string mesA = tyA->toString();
  std::string expectA = "float<1 x 32 x 32 x 3>";
  EXPECT_EQ(mesA, expectA);
  EXPECT_EQ(mesA, os.str());
}

INSTANTIATE_BACKEND_TEST(BackendExecTest);
INSTANTIATE_BACKEND_TEST(BackendExecStatelessTest);
