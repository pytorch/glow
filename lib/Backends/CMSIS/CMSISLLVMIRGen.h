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
#ifndef GLOW_BACKENDS_CMSIS_CMSISLLVMIRGEN_H
#define GLOW_BACKENDS_CMSIS_CMSISLLVMIRGEN_H

#include "../CPU/CPULLVMIRGen.h"

namespace glow {

class CMSISLLVMIRGen : public CPULLVMIRGen {

public:
  virtual ~CMSISLLVMIRGen() = default;

  
  explicit CMSISLLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                          std::string mainEntryName, llvm::StringRef libjitBC);

  explicit CMSISLLVMIRGen(const IRFunction *M, AllocationsInfo &allocationsInfo,
                          std::string mainEntryName, llvm::StringRef libjitBC,
			  llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry);

  virtual void generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                      const glow::Instruction *I) override;
};

} // namespace glow

#endif // GLOW_BACKENDS_CMSIS_CMSISLLVMIRGEN_H
