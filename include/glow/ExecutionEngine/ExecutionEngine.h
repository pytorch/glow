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
#ifndef GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H
#define GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H

#include "glow/Backends/Backend.h"
#include "glow/Base/Train.h"
#include "glow/Base/Traits.h"
#include "glow/Optimizer/Optimizer.h"

#include "llvm/ADT/ArrayRef.h"

#include <memory>
#include <unordered_map>

namespace glow {

class Function;
class Node;
class Interpreter;
class Variable;
class Tensor;
class Module;
class Value;

/// This is the ExecutionEngine. It owns the Graph, the IR, and the backends.
/// The Graph, IR, etc in this class are defined as pointers, in order to
/// erase the type and prevent the internal types from leaking out to the
/// users of this class.
class ExecutionEngine final {
  /// The Module that represents the high-level program.
  std::unique_ptr<Module> M_;
  /// The IR function that represents the program.
  std::unique_ptr<IRFunction> IR_;
  /// The network execution backend.
  std::unique_ptr<Backend> backend_;
  /// The training configuration.
  TrainingConfig config_;
  /// The kind of the backend being currently used.
  BackendKind backendKind_;

  /// Optimize the graph, generate IR, and optimize the IR.
  void generateIR(CompilationMode mode, Function *F);

public:
  ExecutionEngine(BackendKind backendKind = BackendKind::Interpreter);

  ~ExecutionEngine();

  // Set the code generator kind to \p backendKind. New code will be generated
  // using this backend.
  void setBackend(BackendKind backendKind);

  /// Reset the execution engine.
  void reset();

  /// \returns the internal IR function.
  IRFunction &getIR() { return *IR_; }

  /// \returns the internal graph.
  Module &getModule() { return *M_; }

  /// \returns whether operation is supported by the underlying backend.
  bool isOpSupported(Kinded::Kind opKind, ElemKind elementTy) const {
    return backend_->isOpSupported(opKind, elementTy);
  }

  /// Optimize the graph, generate IR, optimize IR and compile it for a
  /// specific target. This method should be invoked before the run method.
  void compile(CompilationMode mode, Function *F);

  /// Save a bundle for a standalone execution. This method takes care of
  /// everything when preparing the bundle for saving. There is no need to
  /// invoke the compile method before it.
  void save(CompilationMode mode, Function *F, llvm::StringRef outputDir);

  /// Provides access to the training configuration.
  TrainingConfig &getConfig() { return config_; }

  /// Runs the program in a forward pass. Update the nodes in \p nodes with the
  /// values \p inputs.
  void run(llvm::ArrayRef<Variable *> vars, llvm::ArrayRef<Tensor *> inputs);

  /// Train the network. Perform \p iterations in the training loop. Each
  /// iteration does a full forward and backward pass of a whole batch.
  /// The method updates the variables in \p vars with the tensors \p inputs.
  void runBatch(size_t iterations, llvm::ArrayRef<Variable *> vars,
                llvm::ArrayRef<Tensor *> inputs);

private:
  /// Update the inputs for all variables \p vars with data from the inputs \p
  /// inputs at offset \p sampleIdx. Then perform a run of the network.
  void updateInputsAndRunNetwork(llvm::ArrayRef<Variable *> vars,
                                 llvm::ArrayRef<Tensor *> inputs,
                                 size_t sampleIdx);

  /// Update the content of the tensor \p v with some slices that from \p input.
  /// The data starts at slice \p sampleIdx and wraps around until the
  /// data in \p v is filled. All dimensions, except for the first (batch)
  /// dimension must be identical.
  void loadValueFromTensorSlice(Variable *v, Tensor *input, size_t sampleIdx);

  // Update the content of the tensor \p v with \p input.
  void loadValueFromTensor(Variable *v, Tensor *input);
};

} // namespace glow

#endif // GLOW_EXECUTIONENGINE_EXECUTIONENGINE_H
