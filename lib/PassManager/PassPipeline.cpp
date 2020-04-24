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
#include "glow/PassManager/Pipeline.h"

using namespace glow;

void PassPipelineBase::dump(llvm::raw_ostream &os) const {
  os << "Pipeline contains:\n";
  for (size_t i = 0, e = size(); i < e; i++) {
    const auto &passConfig = elementAt(i);
    os << "FunctionPassIdx " << i << ": {\n";
    auto passName = passConfig.getNameOfPass();
    passConfig.dump(os, passName);
    os << "}\n";
  }
}
