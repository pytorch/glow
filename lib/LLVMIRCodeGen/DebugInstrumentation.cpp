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

#include "glow/LLVMIRCodeGen/CommandLine.h"
#include "glow/LLVMIRCodeGen/LLVMIRGen.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/Regex.h"

#define DEBUG_TYPE "debug-instrumentation"

using namespace glow;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace {

/// Supported types of instrumentations.
enum FunctionInstrumentationType {
  // Every single LLVM IR instruction
  // of selected functions will be instrumented.
  BODY = 1,
  // Function calls will be wrapped in traces.
  CALL = 2,
};

/// Metadata about instrumentation.
struct InstrumentationMetaInformation {
  // Regular expression of function.
  llvm::Regex funcReg;
  // Instrumentation style.
  FunctionInstrumentationType style;
  // If instrumentation style is CALL, then
  // it allows to specify funciton to call before
  llvm::Function *callBefore;
  // and after.
  llvm::Function *callAfter;
};

/// Perform code instrumentation for selected functions.
/// The syntax is array of sections separated by coma ",".
/// Every section starts with function regex, then body/call clause.
/// PE Ex: "func1*:body,func2:call[:before_func2[:after_func2]];..."
static llvm::cl::list<std::string> llvmIrInstrumentation(
    "llvm-code-debug-trace-instrumentation",
    llvm::cl::desc(
        "Create trace instructions for functions bodies or function calls"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
    llvm::cl::cat(getLLVMBackendCat()));

static llvm::cl::opt<std::string> llvmIrInstrPrintoutFuncName(
    "llvm-debug-trace-print-function-name",
    llvm::cl::desc("Select function that will do prints, function must be with "
                   "signature int f(const char *)"),
    llvm::cl::init("printf"), llvm::cl::cat(getLLVMBackendCat()));

/// Inserts code to generate logs or traces into the generated LLVM IR to make
/// debugging easier.
class DebugInstrumentation {
public:
  explicit DebugInstrumentation(LLVMIRGen &irgen) : irgen_{irgen} {
    // Bail if there is nothing to be instrumented.
    if (llvmIrInstrumentation.empty()) {
      return;
    }

    formatInstrArgD_ = irgen_.emitStringConst(
        irgen_.getBuilder(), "Instruction number %u ,stored value %d\n");
    formatInstrArgP_ = irgen_.emitStringConst(
        irgen_.getBuilder(), "Instruction number %u ,stored value %p\n");
    formatFuncInArg_ =
        irgen_.emitStringConst(irgen_.getBuilder(), "Function called %s\n");
    formatFuncOutArg_ =
        irgen_.emitStringConst(irgen_.getBuilder(), "Function exited %s\n");

    // Parse input llvm-code-instrumentation option and convert it to
    // InstrumentationMetaInformation vector.
    for (auto &section : llvmIrInstrumentation) {
      llvm::SmallVector<llvm::StringRef, 4> elements;
      llvm::StringRef(section).split(elements, ":");
      InstrumentationMetaInformation funcToInstrument{
          llvm::Regex(elements[0]), (elements[1] == "body" ? BODY : CALL),
          nullptr, nullptr};

      if (funcToInstrument.style == CALL) {
        if (elements.size() >= 3) {
          funcToInstrument.callBefore =
              irgen_.getModule().getFunction(elements[3]);
          CHECK(funcToInstrument.callBefore)
              << "Cannot find " << elements[3].data() << " function";
        }
        if (elements.size() >= 4) {
          funcToInstrument.callAfter =
              irgen_.getModule().getFunction(elements[3]);
          CHECK(funcToInstrument.callAfter)
              << "Cannot find " << elements[3].data() << " function";
        }
      }
      funcsToInstrument_.emplace_back(std::move(funcToInstrument));
    }
  }

  void run() {
    // Bail if there is nothing to be instrumented.
    if (llvmIrInstrumentation.empty()) {
      return;
    }

    auto *printfF =
        irgen_.getModule().getFunction(llvmIrInstrPrintoutFuncName.getValue());
    CHECK(printfF) << "Cannot find " << llvmIrInstrPrintoutFuncName.getValue()
                   << " function";

    int64_t traceCounter = 0;
    // Iterating over all functions in the module.
    for (auto &F : irgen_.getModule().functions()) {
      bool instrumentFunctionBody = false;
      // Checking if function's body is requested to be instrumented.
      for (auto &funcToInstrument : funcsToInstrument_) {
        if (!funcToInstrument.funcReg.match(F.getName()) ||
            funcToInstrument.style != BODY) {
          continue;
        }
        instrumentFunctionBody = true;
      }

      // Getting down to LLVM IR instruction to insert
      // instrumentation traces.
      for (auto &BB : F) {
        for (auto &I : BB) {
          // Skipping instruction that doesn't change memory.
          if (I.getOpcode() == llvm::Instruction::Alloca ||
              I.getOpcode() == llvm::Instruction::Ret ||
              I.getOpcode() == llvm::Instruction::Unreachable ||
              I.getOpcode() == llvm::Instruction::Br) {
            continue;
          }
          // If function body matched to be instrumented above
          // then we iterated over its body adding traces otherwise
          // checking if current instuction is a call and matching
          // on of the function calls to be instrumented.
          if (instrumentFunctionBody && !isa<llvm::CallInst>(&I)) {
            llvm::IRBuilder<> builder(&I);
            if (isa<llvm::StoreInst>(&I)) {
              builder.SetInsertPoint(&BB, ++builder.GetInsertPoint());
              auto *traceCounterValue =
                  builder.getInt64(static_cast<int64_t>(traceCounter++));
              builder.CreateCall(
                  printfF->getFunctionType(), printfF,
                  {formatInstrArgP_, traceCounterValue,
                   llvm::cast<llvm::StoreInst>(&I)->getPointerOperand()});
            } else {
              builder.SetInsertPoint(&BB, ++builder.GetInsertPoint());
              auto *traceCounterValue =
                  builder.getInt64(static_cast<int64_t>(traceCounter++));
              builder.CreateCall(printfF->getFunctionType(), printfF,
                                 {formatInstrArgD_, traceCounterValue,
                                  llvm::cast<llvm::Value>(&I)});
            }
          } else if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
            // If current LLVM IR instruction is not a call or the
            // instruction/call name doesn't match requested set of function
            // calls to be instrumented, skipping instrumentation, otherwise
            // wrap function call in traces.
            auto funcToInstrument = std::find_if(
                funcsToInstrument_.begin(), funcsToInstrument_.end(),
                [&](auto &fi) {
                  if (CI->getCalledFunction() == nullptr ||
                      !fi.funcReg.match(CI->getCalledFunction()->getName()) ||
                      fi.style != CALL) {
                    return false;
                  }
                  return true;
                });

            // If function is in the list to be instumented and
            // custom functions before (and after specified) use them.
            if (funcToInstrument != funcsToInstrument_.end() &&
                funcToInstrument->callBefore) {
              llvm::IRBuilder<> builder(CI);

              // Args to be used for calling the specialized function.
              llvm::SmallVector<llvm::Value *, 16> argsForInstr;
              for (auto &arg : CI->arg_operands()) {
                argsForInstr.push_back(arg);
              }

              builder.CreateCall(
                  funcToInstrument->callBefore->getFunctionType(),
                  funcToInstrument->callBefore, argsForInstr);
              if (funcToInstrument->callAfter) {
                builder.SetInsertPoint(&BB, ++builder.GetInsertPoint());
                builder.CreateCall(
                    funcToInstrument->callAfter->getFunctionType(),
                    funcToInstrument->callAfter, argsForInstr);
              }
              // Otherwise check if function body instrumentation is going
              // or default wrappers for function requested, wrap the
              // function.
            } else if (instrumentFunctionBody ||
                       funcToInstrument != funcsToInstrument_.end()) {
              instrumentFuntionCall(CI, printfF);
            }
          }
        }
      }
    }

    DEBUG_GLOW(llvm::outs() << "LLVM module after instrumentation:\n");
    DEBUG_GLOW(irgen_.getModule().print(llvm::outs(), nullptr));
  }

private:
  // Prints function input parameter values.
  void instrumentFuntionCall(llvm::CallInst *CI, llvm::Function *printfF) {
    if (CI->getCalledFunction() == nullptr ||
        (CI->getCalledFunction()->getName() == printfF->getName())) {
      return;
    }

    llvm::IRBuilder<> builder(CI);

    auto *functionName = irgen_.emitStringConst(
        irgen_.getBuilder(), CI->getCalledFunction()->getName());
    builder.CreateCall(printfF->getFunctionType(), printfF,
                       {formatFuncInArg_, functionName});

    // Iterating over all function args. First agr is function output
    // if function signature is not void.
#if LLVM_VERSION_MAJOR >= 8
    for (auto &op : CI->args()) {
#else
    for (auto &op : CI->arg_operands()) {
#endif
      llvm::Value *argFormat = nullptr;
      auto type = op.get()->getType();

      // Printing interger values with %d symbol.
      if (type->isIntegerTy() ||
          (type->isPointerTy() && dyn_cast<llvm::PointerType>(type)
                                      ->getElementType()
                                      ->isIntegerTy())) {
        auto *value = op.get();
        // If arg is a pointer to integer value get value and print it.
        if (type->isPointerTy()) {
          value = builder.CreateLoad(op.get());
        }

        argFormat = irgen_.emitStringConst(irgen_.getBuilder(), "\targ: %d\n");
        builder.CreateCall(printfF->getFunctionType(), printfF,
                           {argFormat, value});
        continue;
      }

      // Processing pointers to structure values.
      if (type->isPointerTy()) {
        auto *structType = dyn_cast<llvm::StructType>(
            dyn_cast<llvm::PointerType>(type)->getElementType());
        if (structType) {
          std::string printStructFuncName = "pretty_print_";
          llvm::StringRef structName;
          // Structure types usually start either from struct. or class.
          // stripping it from a name.
          if (structType->getName().startswith("struct.")) {
            structName = structType->getName().slice(
                std::string("struct.").length(), structType->getName().size());
          } else if (structType->getName().startswith("class.")) {
            structName = structType->getName().slice(
                std::string("class.").length(), structType->getName().size());
          }
          printStructFuncName += structName;
          // Checking if module has print_<Struct\Class name> function to
          // print arg. If not then just jumpring to default printout - ...
          if (auto *printStruct =
                  irgen_.getModule().getFunction(printStructFuncName)) {
            builder.CreateCall(printStruct->getFunctionType(), printStruct,
                               {op.get()});
            continue;
          }
        }
      }

      argFormat = irgen_.emitStringConst(irgen_.getBuilder(), "\targ: ...\n");
      builder.CreateCall(printfF->getFunctionType(), printfF, {argFormat});
    }

    builder.SetInsertPoint(CI->getParent(), ++builder.GetInsertPoint());
    builder.CreateCall(printfF->getFunctionType(), printfF,
                       {formatFuncOutArg_, functionName});
  }

  // Parsed metadata about instrumentation types.
  std::vector<InstrumentationMetaInformation> funcsToInstrument_;

  /// LLVMIRGen to be used.
  LLVMIRGen &irgen_;

  // Format string for simple line instumentation.
  llvm::Value *formatInstrArgD_ = nullptr;
  // Format string for simple line instumentation.
  llvm::Value *formatInstrArgP_ = nullptr;
  // Format string for trace before function call.
  llvm::Value *formatFuncInArg_ = nullptr;
  // Format string for simple after function call.
  llvm::Value *formatFuncOutArg_ = nullptr;
};

} // namespace

void LLVMIRGen::performDebugInstrumentation() {
  DebugInstrumentation debugInstrumentation(*this);
  debugInstrumentation.run();
}
