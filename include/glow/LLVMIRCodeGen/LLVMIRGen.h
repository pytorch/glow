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
#ifndef GLOW_LLVMIRCODEGEN_LLVMIRGEN_H
#define GLOW_LLVMIRCODEGEN_LLVMIRGEN_H

#include "glow/Base/Tensor.h"
#include "glow/IR/IR.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

namespace glow {

class PlaceholderBindings;
class IRFunction;
class Value;
class Tensor;
class Constant;
class Instruction;
class WeightVar;
class AllocationsInfo;
class LLVMBackendOptions;

/// Different kinds of memory areas used by the emitted LLVM function.
/// The order is important. It should match the order of base addresses
/// arguments passed to "main".
enum MemoryAreaKind {
  ConstWeightsMemoryArea,
  MutableWeightsMemoryArea,
  ActivationsMemoryArea,
  LastMemoryArea
};

/// A POD struct that stores information related to debug info.
struct DebugInfo {
  /// Source file for the main function.
  llvm::DIFile *mainFile_{nullptr};
  /// Debug info for the main function.
  llvm::DISubprogram *mainF_{nullptr};
  /// Line number for the first instruction in the textual representation of
  /// the Glow IR.
  size_t mainFileFirstInstrLineNo_{0};
  /// Debug info for the current compilation unit.
  llvm::DICompileUnit *compilationUnit_{nullptr};
  /// Mapping from LLVM types to DebugInfo types.
  llvm::DenseMap<llvm::Type *, llvm::DIType *> DITypes_;
  /// Names of global variables to hold the bases address of different memory
  /// areas, e.g. activations, constant or mutable weights.
  llvm::SmallVector<llvm::StringRef, LastMemoryArea - 1>
      baseAddressesVariablesNames_;
  /// Maps memory area kinds to the global variables holding the base address of
  /// the corresponding memory area, e.g. activations, constant or mutable
  /// weights area. Used only when producing a debug information. These
  /// variables are required to properly show in the debugger weights and
  /// activations variables, which are expressed in DWARF using a relative
  /// addressing mode with the base addresses stored in these base addresses
  /// variables.
  llvm::SmallVector<llvm::GlobalVariable *, LastMemoryArea - 1>
      baseAddressesVariables_;
};

/// Different kinds of bundle APIs.
enum BundleApiType {
  /// Dynamic bundle API with the following features:
  /// - the weights are exported in a binary file which are assumed
  ///   to be loaded dynamically at run-time.
  /// - the memory layout information (bundle configuration) is only
  ///   available at run-time and therefore allows ONLY dynamic memory
  ///   allocaton.
  Dynamic,
  /// Static bundle API (default) with the following features:
  /// - the weights are exported in a binary file but and also in a
  ///   text file (C array format) suitable to include at compile-time.
  /// - the memory layout information (bundle configuration) is available
  ///   at compile-time through macros printed in the header file and thus
  ///   allows also static memory allocation.
  /// - this API is suitable for low end devices with no file system or OS
  ///   (bare-metal).
  Static,
};

/// This is a class containing a common logic for the generation of the LLVM IR
/// from an IRFunction. The primary clients of this class are JITs and bundlers.
class LLVMIRGen {
protected:
  /// Implementation of emitDataParallelKernel where we bound the number of
  /// inputs to 64.
  /// \param builder IRBuilder to be used for the LLVM IR code emission.
  /// \param bundle set of instructions to be emitted as a data-parallel kernel.
  /// \param argType types of arguments for the data-parallel kernel.
  /// \param bufferToArgNum mapping from a buffer to its argument number in the
  /// data-parallel kernel.
  /// \param buffers buffers used by the data-parallel kernel.
  virtual void
  emitDataParallelKernelImpl(llvm::IRBuilder<> &builder,
                             llvm::ArrayRef<const Instruction *> bundle,
                             llvm::ArrayRef<llvm::Type *> argTypes,
                             llvm::DenseMap<Value *, int> &bufferToArgNum,
                             llvm::ArrayRef<llvm::Value *> buffers);

  /// The IR to generate code for.
  const IRFunction *F_;
  /// LLVM IR function corresponding to F_.
  llvm::Function *llvmF_;
  /// Set of emitted LLVM functions for IR functions.
  llvm::SmallVector<llvm::Function *, 4> emittedLLVMFunctions_;
  /// The LLVM context.
  std::unique_ptr<llvm::LLVMContext> ctx_;
  /// The LLVM IR module.
  std::unique_ptr<llvm::Module> llmodule_{nullptr};
  /// The target machine.
  std::unique_ptr<llvm::TargetMachine> TM_;
  /// Information about allocations.
  AllocationsInfo &allocationsInfo_;
  /// Name of the bundle.
  std::string bundleName_;
  /// Name of the main entry.
  std::string mainEntryName_;
  /// Base name of the saved bundle file, without extension.
  std::string savedBundleName_;
  /// Instruction number for the module.
  std::unique_ptr<InstructionNumbering> instrNumbering_;
  /// Value holding the base address of the activations memory area.
  llvm::Value *baseActivationsAddr_{nullptr};
  /// Value holding the base address of the constant WeightVars memory area.
  llvm::Value *baseConstantWeightVarsAddr_{nullptr};
  /// Value holding the base address of mutable WeightVars memory area.
  llvm::Value *baseMutableWeightVarsAddr_{nullptr};
  /// Value holding the address of the offsets array.
  llvm::Value *offsetsArray_{nullptr};
  /// Maps constant arrays to the constant expressions representing size_t
  /// pointers to these arrays. This is done to ensure the proper uniqueness
  /// semantics of such pointers just like it is done for llvm::Constants.
  llvm::DenseMap<llvm::Constant *, llvm::Value *> constArrayPtrs_;
  /// The IRBuilder used for the code generation.
  std::unique_ptr<llvm::IRBuilder<>> builder_;
  /// Output directory for bundles, debug info files, etc.
  llvm::StringRef outputDir_;
  /// Debug info emission support.
  DebugInfo dbgInfo_;
  /// Debug info builder.
  std::unique_ptr<llvm::DIBuilder> DIBuilder_;

  /// A set that contains all of the argument that we request from the
  /// specializer not to specialize.
  llvm::DenseSet<llvm::Value *> dontSpecializeArgsSet_;

  /// Bitcode of the libjit. Contains the starting address and the length of
  /// the bitcode.
  llvm::StringRef libjitBC_;

  /// Array with raw objects which can be optionally used by the code generator
  /// to archive additional object code into the bundle.
  llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry_;

  /// Array with the names of the additional objects which will be archived
  /// into the bundle. The objects must be registered in \ref objectRegistry_.
  std::vector<std::string> bundleObjects_;

  /// Whether to print the IR instrumentation callback API.
  bool printInstrumentIR_{false};

  /// Generates LLVM IR that computes the address of \p val using \p builder.
  /// The address type is specified by \p ptrTy.
  virtual llvm::Value *emitValueAddress(llvm::IRBuilder<> &builder,
                                        const glow::Value *val);
  /// Emit the address of the buffer \p v inside a data-parallel kernel \p
  /// kernel using the mapping provided by \p bufferToArgNum.
  llvm::Value *emitBufferAddress(llvm::IRBuilder<> &builder, Value *val,
                                 llvm::Function *kernel,
                                 llvm::DenseMap<Value *, int> &bufferToArgNum);
  /// Generates LLVM IR that computes the size of the tensor of \p val using
  /// \p builder. The size type is native to the machine (size_t).
  virtual llvm::Value *emitValueSize(llvm::IRBuilder<> &builder,
                                     const glow::Value *val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstF32(llvm::IRBuilder<> &builder, float val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstI32(llvm::IRBuilder<> &builder, int32_t val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstI16(llvm::IRBuilder<> &builder, int16_t val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstI8(llvm::IRBuilder<> &builder, int8_t val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstI1(llvm::IRBuilder<> &builder, bool val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstSizeT(llvm::IRBuilder<> &builder, size_t val);
  /// Generates LLVM IR that materializes the constant \p val.
  llvm::Value *emitConstDimT(llvm::IRBuilder<> &builder, dim_t val);
  /// Generates LLVM IR that materializes the constant \p val as a constant of
  /// the type specified by \p kind.
  llvm::Value *emitConst(llvm::IRBuilder<> &builder, float val,
                         glow::ElemKind kind);
  /// Generates LLVM IR that materializes the constant array \p vals. Note that
  /// it will cast non-size_t types T into size_t.
  template <typename T>
  llvm::Value *emitConstSizeTArray(llvm::IRBuilder<> &builder,
                                   llvm::ArrayRef<T> vals);

  /// Generates LLVM IR that materializes the constant array \p vals. Note that
  /// it will cast non-dim_t types T into dim_t.
  template <typename T>
  llvm::Value *emitConstDimTArray(llvm::IRBuilder<> &builder,
                                  llvm::ArrayRef<T> vals);

  /// Generates LLVM IR that materializes the constant array \p vals. Note that
  /// int32 data type is accepted.
  llvm::Value *emitConstI32Array(llvm::IRBuilder<> &builder,
                                 llvm::ArrayRef<int32_t> vals);

  /// Generates LLVM IR that materializes the constant array \p vals. Note that
  /// float data type is accepted.
  llvm::Value *emitConstFloatArray(llvm::IRBuilder<> &builder,
                                   llvm::ArrayRef<float> vals);

  /// Generates LLVM IR that materializes the constant array \p vals. Elements
  /// of vals have the type \p elemTy.
  llvm::Value *emitConstArray(llvm::IRBuilder<> &builder,
                              llvm::ArrayRef<llvm::Constant *> vals,
                              llvm::Type *elemTy);

  /// Generates LLVM IR to store all the LLVM IR values \p vals consecutively
  /// starting with the base pointer given by \p basePtr and the relative base
  /// index \p baseIdx. The LLVM IR values \p vals must have same type T and the
  /// type of the base pointer must be T*.
  void emitArrayStore(llvm::IRBuilder<> &builder,
                      llvm::ArrayRef<llvm::Value *> vals, llvm::Value *basePtr,
                      unsigned baseIdx = 0);

  /// Generates LLVM IR that computes the dimensions of \p val using \p builder.
  /// The result type is "size_t*".
  virtual llvm::Value *emitValueDims(llvm::IRBuilder<> &builder,
                                     const glow::Value *val);

  /// Generates LLVM IR that materializes the float activation parameters for
  /// the instruction \p I.
  template <class InstructionTy>
  llvm::Value *emitConstFloatActivationArgs(llvm::IRBuilder<> &builder,
                                            const InstructionTy *I);

  /// Generates LLVM IR that materializes the quantized activation parameters
  /// for the instruction \p I.
  template <class InstructionTy>
  llvm::Value *emitConstQuantActivationArgs(llvm::IRBuilder<> &builder,
                                            const InstructionTy *I);

  /// Load base addresses of different memory areas (activations, const
  /// weightvars, mutable weight vars) so that they can be reused inside the
  /// body of the function.
  virtual void loadBaseAddresses(llvm::IRBuilder<> &builder);
  /// Create a function representing a stacked kernel for instructions provided
  /// in \p stackedInstrs.
  virtual void
  emitDataParallelKernel(llvm::IRBuilder<> &builder,
                         llvm::ArrayRef<const Instruction *> stackedInstrs);
  /// Emit IR for the data parallel instruction \p I which is invoked inside the
  /// stacked \p kernel. The current loop count is described by \p loopCount.
  /// The \p bufferToArgNum map can be used to find the required buffers, which
  /// are provided as arguments to the stacked \p kernel.
  /// Derived classes may want to override this function to implement a
  /// backend-specific LLVM IR generation logic for some intructions.
  virtual void generateLLVMIRForDataParallelInstr(
      llvm::IRBuilder<> &builder, const glow::Instruction *I,
      llvm::Function *kernel, llvm::DenseMap<Value *, int> &bufferToArgNum,
      llvm::Value *loopCount);
  /// \returns the llvm type of the glow vale \p val.
  llvm::Type *getElementType(llvm::IRBuilder<> &builder, const Value *val);
  /// Create a debug information for a given LLVM type \p ty.
  llvm::DIType *getDebugType(llvm::IRBuilder<> &builder, llvm::Type *ty);
  /// Init the generation of debug information.
  virtual void initDebugInfo();
  /// Generate debug information for the current function.
  virtual void generateFunctionDebugInfo();
  /// Generate debug information for the whole module.
  virtual void generateModuleDebugInfo();
  /// Set the debug location for the \p builder, so that it corresponds to the
  /// instruction \p I in the textual representation of the Glow IR.
  void setCurrentDebugLocation(llvm::IRBuilder<> &builder,
                               const glow::Instruction *I);
  /// Get or create a debug information for a given LLVM function.
  llvm::DISubprogram *getOrCreateFunctionDebugInfo(llvm::Function *F,
                                                   llvm::DIScope *scope,
                                                   llvm::DIFile *file,
                                                   unsigned lineNo);
  /// Emit a debug info for the logical global variable representing a weight or
  /// an activation described by \p val. This allows for inspecting the values
  /// of weights and activations when using a debugger. Logical global variables
  /// are not materialized and do not require any additional memory to be
  /// reserved or allocated. Instead, they reside at offsets described by
  /// AllocationsInfo inside the memory blocks dynamically allocated by clients
  /// for weights and activations, but behave like regular global variables from
  /// the debugger's perspective.
  void emitDebugGlobalVariableForValue(const Value *val);

  /// Create LLVM IR for the for loop with a loop count specified by the only
  /// parameter of the enclosing function.
  /// \returns a pair of basic blocks. The first BB is the BB of the loop body,
  /// the second BB is the loop exit BB.
  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoop(llvm::IRBuilder<> &builder, llvm::LLVMContext &ctx,
             llvm::Value *numElements) const;

  /// \returns the backing tensor associated to the IR constant value \p value.
  Tensor getTensorForConstantValue(Value *value);

public:
  /// Destructor
  virtual ~LLVMIRGen() {}
  /// Ctor.
  /// \param M IRFunction to be converted into LLVM IR.
  /// \param allocationsInfo information about allocation of weights and
  /// activations.
  /// \param mainEntryName Name of the main entry.
  /// \param libjitBC bitcode of the backend's libjit library.
  explicit LLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName, llvm::StringRef libjitBC);

  /// Ctor.
  /// \param M IRFunction to be converted into LLVM IR.
  /// \param allocationsInfo information about allocation of weights and
  /// activations.
  /// \param mainEntryName Name of the main entry.
  /// \param libjitBC bitcode of the backend's libjit library.
  /// \param objectRegistry array with additional objects for code generation.
  explicit LLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                     std::string mainEntryName, llvm::StringRef libjitBC,
                     llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry);

  /// Init the TargetMachine using settings provided by \p llvmBackend.
  virtual void initTargetMachine(const LLVMBackendOptions &opts);

  /// Init the TargetOptions \p targetOpts in a backend-specific way. Use \p
  /// backendOpts as input if necessary.
  virtual void initTargetOptions(llvm::TargetOptions &targetOpts,
                                 const LLVMBackendOptions &backendOpts);

  /// Emit LLVM-IR for the instruction \p I, using the builder \p builder.
  /// Derived classes may want to override this function to implement a
  /// backend-specific LLVM IR generation logic for some intructions.
  /// \param builder IRBuilder to be used to emit LLVM IR code.
  /// \param I IR instruction which should be compiled into LLVM IR.
  virtual void generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                      const glow::Instruction *I);
  /// Emit LLVM-IR for the whole IRFunction.
  virtual void generateLLVMIRForModule(llvm::IRBuilder<> &builder);
  /// Helper function to create a new CallInst, with the specified \p builder,
  /// \p callee, and \p args. Verifies that the function signature is correct,
  /// and then creates and \returns the CallInst.
  /// \param builder the IR builder to be used for creating the Call
  /// instruction. \param callee the function to be called. \param args
  /// arguments to be passed in this call. If \p checked is set, this helper
  /// emits checks for the int result of the call and returns from the caller if
  /// it is non-zero. If callee does not return an int result, \p checked has no
  /// effect. \returns generated Call instruction
  virtual llvm::CallInst *createCall(llvm::IRBuilder<> &builder,
                                     llvm::Function *callee,
                                     llvm::ArrayRef<llvm::Value *> args,
                                     bool checked = false);
  /// The checked version of createCall.
  virtual llvm::CallInst *createCheckedCall(llvm::IRBuilder<> &builder,
                                            llvm::Function *callee,
                                            llvm::ArrayRef<llvm::Value *> args);
  /// The unchecked version of createCall.
  virtual llvm::CallInst *
  createUncheckedCall(llvm::IRBuilder<> &builder, llvm::Function *callee,
                      llvm::ArrayRef<llvm::Value *> args);
  /// \returns a libjit API function by name.
  virtual llvm::Function *getFunction(const std::string &name);
  /// \returns a libjit API function by name and tensor element type.
  virtual llvm::Function *getFunction(const std::string &name,
                                      glow::ElemKind elemTy);
  /// \returns a libjit API function by name and tensor element type.
  virtual llvm::Function *
  getFunction(const std::string &name,
              llvm::ArrayRef<glow::ElemKind> elemTyArray);
  /// \returns current LLVM function.
  virtual llvm::Function *getLLVMFunction();
  /// Optimize the function \p F and the module that owns it. Use the target
  /// information from the \p TM target machine.
  virtual void optimizeLLVMModule(llvm::Module *M, llvm::TargetMachine &TM);
  /// Performs specialization of operations based on constant parameters.
  virtual void performSpecialization();
  /// Insert debug traces in appropriate places.
  virtual void performDebugInstrumentation();
  /// \returns allocations info.
  virtual AllocationsInfo &getAllocationsInfo() { return allocationsInfo_; }
  /// \returns the name of the bundle, to be used for filename when saving.
  llvm::StringRef getBundleName() const;
  /// Set the name of the bundle (name is automatically legalized).
  void setBundleName(const std::string &name);
  /// \returns the base name of the saved bundle file to be used by a
  /// BundleSaver.
  llvm::StringRef getSavedBundleName() const;
  /// Set the base name of the saved bundle file.
  void setSavedBundleName(const std::string &name);
  /// \returns the name of the main entry point.
  /// When JITting, it will be "main". In case of bundling it will be the name
  /// of the bundle.
  std::string getMainEntryName() const;
  /// Set the name of the main entry point (name is automatically legalized).
  void setMainEntryName(std::string name);
  /// Creates an LLVM module, the entry function, etc.
  virtual void initCodeGen();
  /// Emits the code of the entry function, performs optimizations, etc.
  virtual void performCodeGen();
  /// Finish the LLVM IR code generation.
  virtual void finishCodeGen();
  /// \returns the current builder.
  llvm::IRBuilder<> &getBuilder() { return *builder_; }
  /// \returns the target machine description.
  llvm::TargetMachine &getTargetMachine() { return *TM_; }
  /// Takes the target machine for further processing, e.g. by a JIT.
  /// The target machine cannot be used by the LLVMIRGen afterwards.
  std::unique_ptr<llvm::TargetMachine> takeTargetMachine() {
    return std::move(TM_);
  }
  /// \returns the LLVMContext being used.
  llvm::LLVMContext &getLLVMContext() { return *ctx_; }
  /// Takes the LLVM Context for further processing, e.g. by a JIT.
  /// The context cannot be used by the LLVMIRGen afterwards.
  std::unique_ptr<llvm::LLVMContext> takeLLVMContext() {
    return std::move(ctx_);
  }
  /// Borrows the LLVM module for further processing, e.g. by a JIT.
  /// The module cannot be used by the LLVMIRGen afterwards.
  std::unique_ptr<llvm::Module> borrowModule() { return std::move(llmodule_); }
  /// \returns current LLVM module.
  llvm::Module &getModule() const { return *llmodule_; }
  /// \returns the IR function.
  const IRFunction *getIRFunction() { return F_; }
  /// Set IRFunction to be processed next.
  void setIRFunction(const IRFunction *F) { F_ = F; }
  /// Set output directory for bundles, debug info files, etc.
  void setOutputDir(llvm::StringRef outputDir) { outputDir_ = outputDir; }
  /// Get output directory for bundles, debug info files, etc.
  llvm::StringRef getOutputDir() const { return outputDir_; }
  /// Emit the array of constant offsets as provided by the \p allocationsInfo.
  virtual llvm::Value *
  emitConstOffsetsArray(llvm::IRBuilder<> &builder,
                        const AllocationsInfo &allocationsInfo);
  /// Generate debug info for a LLVM function \p F.
  virtual void generateFunctionDebugInfo(llvm::Function *F);
  /// Generates LLVM IR that materializes the string literal \p str.
  virtual llvm::Value *emitStringConst(llvm::IRBuilder<> &builder,
                                       llvm::StringRef str);
  /// Emit symbols to JIT to allow it to use host side file printing.
  void generateJITFileWriter();
  /// Register \p val as an argument that should not be specialized.
  virtual void markArgAsUnspecialized(llvm::Value *val);
  /// \returns bit-width of the target size_t.
  virtual unsigned getTargetSizeTWidth() const;
  /// \returns the sizeof(size_t) of the actual target-specific size_t type that
  /// was used to compile libjit into LLVM bitcode.
  unsigned getLibjitSizeTWidth() const;
  /// \returns the sizeof(int) of the actual target-specific int type that
  /// was used to compile libjit into LLVM bitcode.
  unsigned getLibjitIntWidth() const;
  /// \returns true if a call is eligible for specialization.
  virtual bool isEligibleForSpecialization(const llvm::CallInst *call);
  /// \returns true if a global symbol \p GV needs to be preserved in the module
  /// and not interalized during optimizations.
  virtual bool preserveSymbol(const llvm::GlobalValue &GV);
  /// \returns inlining mode to be used for a function \p F.
  virtual llvm::Attribute::AttrKind
  getInlinineAttr(const llvm::Function *F) const;
  /// Configures a provided PassManagerBuilder \p PMB in a backend-specific way
  virtual void populatePassManagerBuilderOptions(llvm::PassManagerBuilder &PMB);
  /// Update inline attributes of functions in the module \p M using a
  /// backend-specific logic.
  virtual void updateInlineAttributes(llvm::Module *M);
  /// \returns true if an instruction \p I can be part of a data parallel
  /// kernel. This gives backends a possibility to provide a custom logic to
  /// decide on a per-instruction basis what can be part of data parallel
  /// kernels. Typically an instruction which is isDataParallel() can be part of
  /// a data parallel kernel. But a backend may decide that a specific
  /// instruction \p I cannot be part of data-parallel kernels, because there is
  /// no support for this functionality in this backend yet.
  virtual bool canBePartOfDataParallelKernel(const glow::Instruction *I) const;
  /// \returns a string which is printed at the end of the bundle header file
  /// following the standard content produced by the bundle saver.
  virtual std::string getBundleHeaderExtra() const;
  /// \returns the object registry for this code generator instance.
  virtual llvm::ArrayRef<llvm::MemoryBufferRef> getObjectRegistry() const;
  /// Set the object registry for this code generator instance.
  virtual void
  setObjectRegistry(llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry);
  /// \returns the names of the objects which are archived into the bundle.
  virtual std::vector<std::string> getBundleObjects() const;
  /// Add a bundle object \p objectName to be archived to the bundle. The object
  /// must be registered in the \ref objectRegistry_ otherwise error is thrown.
  virtual void addBundleObject(llvm::StringRef objectName);
};

template <typename T>
llvm::Value *LLVMIRGen::emitConstSizeTArray(llvm::IRBuilder<> &builder,
                                            llvm::ArrayRef<T> vals) {
  assert(std::is_integral<T>() && "Can only convert integral type to size_t.");
  auto SizeTType = builder.getIntNTy(getLibjitSizeTWidth());
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    assert(I >= 0 && "Only allow casting positive values into size_t.");
    assert(I <= std::numeric_limits<size_t>::max() &&
           "Do not allow overflow of size_t.");
    elems.push_back(llvm::ConstantInt::get(SizeTType, (size_t)I));
  }
  return emitConstArray(builder, elems, SizeTType);
}

template <typename T>
llvm::Value *LLVMIRGen::emitConstDimTArray(llvm::IRBuilder<> &builder,
                                           llvm::ArrayRef<T> vals) {
  assert(std::is_integral<T>() && "Can only convert integral type to dim_t.");
  auto DimTType = builder.getIntNTy(sizeof(dim_t) * 8);
  std::vector<llvm::Constant *> elems;
  for (auto I : vals) {
    assert(I >= 0 && "Only allow casting positive values into size_t.");
    assert(I <= std::numeric_limits<dim_t>::max() &&
           "Do not allow overflow of size_t.");
    elems.push_back(llvm::ConstantInt::get(DimTType, (dim_t)I));
  }
  return emitConstArray(builder, elems, DimTType);
}

} // namespace glow

#endif // GLOW_LLVMIRCODEGEN_LLVMIRGEN_H
